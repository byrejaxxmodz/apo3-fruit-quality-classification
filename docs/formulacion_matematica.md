# Formulación matemática (evidencia PI3)

> Borrador para el informe. Las ecuaciones están en LaTeX, que GitHub renderiza. Hay que completar los valores reales cuando estén los resultados.

## 1. Formulación del problema

Sea una imagen RGB $I \in [0,255]^{H \times W \times 3}$ con **una** fruta sobre un fondo uniforme. Se buscan dos funciones:

$$
f_q : I \mapsto \hat{y}_q \in \mathcal{Q} = \{\text{good}, \text{regular}, \text{bad}\}, \qquad
f_s : I \mapsto \hat{y}_s \in \{\text{pequeño}, \text{mediano}, \text{grande}\}
$$

**Variables principales.**

| Símbolo | Significado |
|---------|-------------|
| $M \in \{0,1\}^{H\times W}$ | máscara de segmentación de la fruta |
| $A = \sum M$ | área en píxeles |
| $d_{norm}$ | diámetro equivalente normalizado |
| $\mathbf{x} \in \mathbb{R}^{62}$ | vector de características manuales |
| $\theta$ | parámetros del modelo; $\lambda$ hiperparámetros |

**Pipeline formal:**

$$
I \xrightarrow{\text{resize}} I' \xrightarrow{\text{segmentación}} (M, A) \xrightarrow{\text{recorte}} I_c \in \mathbb{R}^{128\times128\times3}
\begin{cases}
\xrightarrow{\phi} \mathbf{x} \xrightarrow{h_\theta} \hat{y}_q & \text{(ML clásico)}\\
\xrightarrow{g_\theta} \hat{y}_q & \text{(CNN)}
\end{cases}
$$

**¿Por qué es un problema complejo de ingeniería?**
- Hay mucha variación dentro de cada clase: forma, color según la variedad y la madurez, iluminación, cámara.
- Las fronteras entre clases son difusas y subjetivas, sobre todo entre *regular* y las otras dos (ruido en las etiquetas).
- Hay pocos datos y están desbalanceados, y las fuentes tienen dominios distintos (Kaggle vs. celular).
- El tamaño es una medida física, pero se estima desde una proyección 2D sin conocer la escala.
- Se necesita inferencia en tiempo real con hardware modesto.

## 2. Segmentación (Otsu)

Sobre el canal de saturación $S$, Otsu elige el umbral $t^*$ que maximiza la varianza **entre** clases:

$$
t^* = \arg\max_t \; \omega_0(t)\,\omega_1(t)\,[\mu_0(t) - \mu_1(t)]^2
$$

donde $\omega_i$ son las proporciones de píxeles a cada lado del umbral y $\mu_i$ sus medias. Esto equivale a minimizar la varianza **dentro** de cada clase. Funciona bien porque el fondo neutro tiene $S \approx 0$ y la fruta tiene $S$ alta, así que el histograma es bimodal. Después se aplica apertura ($\ominus$ y luego $\oplus$) para quitar ruido y cierre para rellenar los brillos.

## 3. Estimación de tamaño

$$
d_{eq} = \sqrt{\frac{4A}{\pi}}, \qquad d_{norm} = \frac{d_{eq}}{\min(H, W)} \in [0, 1]
$$

$d_{eq}$ es el diámetro del círculo que tiene la misma área que la fruta. Es más robusto que el ancho de la caja porque no depende de la rotación. La clase se asigna con los terciles $P_{33}, P_{66}$ de $d_{norm}$ **por fruta**. **Limitación:** $d_{norm}$ es proporcional a $\frac{D_{real} \cdot f}{Z}$ (modelo de cámara pinhole), así que depende de la distancia $Z$. Por eso hay que fijar la distancia de captura o usar un objeto de referencia.

Descriptores de forma: circularidad $C = 4\pi A / P^2$ (vale 1 para un círculo) y solidez $= A / A_{hull}$.

## 4. Características manuales $\phi(I_c)$

- **Histogramas HSV normalizados:** $h_k = n_k / \sum_j n_j$, es decir, una distribución de probabilidad del color. Se usa HSV porque separa el tono (madurez) de la iluminación (V).
- **Momentos en Lab:** $\mu$, $\sigma$ y la asimetría $\gamma = E[(x-\mu)^3]/\sigma^3$. Las manchas oscuras sesgan la distribución de L.
- **LBP uniforme:** $LBP_{P,R} = \sum_{p=0}^{P-1} s(g_p - g_c)\, 2^p$, que describe la micro-textura.
- **GLCM:** contraste $= \sum_{i,j} (i-j)^2 p(i,j)$, homogeneidad $= \sum p(i,j)/(1+|i-j|)$, energía $= \sum p(i,j)^2$.
- **Estandarización:** $z = (x - \mu_{train}) / \sigma_{train}$. Las medias y desviaciones se calculan **solo con train**, dentro del pipeline, para evitar fuga de información.

## 5. Modelos

**Regresión logística multinomial (línea base lineal):**
$$ P(y=k\mid\mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}{\sum_j e^{\mathbf{w}_j^\top \mathbf{x} + b_j}}, \quad \min_\theta -\sum_i \log P(y_i \mid \mathbf{x}_i) + \frac{1}{2C}\|\mathbf{w}\|^2 $$

**SVM con kernel RBF:**
$$ \min_{\mathbf{w}, b, \xi} \tfrac12\|\mathbf{w}\|^2 + C\sum_i \xi_i \;\; \text{s.a.}\;\; y_i(\mathbf{w}^\top\varphi(\mathbf{x}_i)+b) \ge 1-\xi_i, \qquad K(\mathbf{x},\mathbf{x}') = e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2} $$
$C$ controla el equilibrio entre margen y error, y $\gamma$ el radio de influencia de cada vector de soporte.

**Random Forest:** $\hat{y} = \text{moda}\{T_b(\mathbf{x})\}_{b=1}^{B}$. Cada árbol se entrena sobre una muestra bootstrap y divide los nodos minimizando la impureza de Gini $G = 1 - \sum_k p_k^2$. Promediar árboles poco correlacionados reduce la varianza.

**CNN:** convolución $(X * W)_{u,v} = \sum_{c,i,j} W_{c,i,j} X_{c,u+i,v+j} + b$, seguida de ReLU $\max(0,z)$, max-pooling y global average pooling. Se entrena con **entropía cruzada ponderada**:
$$ \mathcal{L} = -\frac{1}{N}\sum_i w_{y_i} \log \frac{e^{z_{i,y_i}}}{\sum_k e^{z_{i,k}}}, \qquad w_k = \frac{N}{K\,N_k} $$
y con Adam más weight decay ($+\lambda\|\theta\|^2$).

## 6. Selección y métricas

- **Validación cruzada estratificada (k = 5):** $\hat{\lambda} = \arg\max_\lambda \frac{1}{k}\sum_{j=1}^{k} F1_{macro}(h_{\theta(\lambda)}^{(-j)}, D_j)$
- $P_k = \frac{TP_k}{TP_k+FP_k}$, $R_k = \frac{TP_k}{TP_k+FN_k}$, $F1_k = \frac{2P_kR_k}{P_k+R_k}$, $F1_{macro} = \frac1K\sum_k F1_k$
- **Sobreajuste:** brecha $\Delta = F1_{train} - F1_{val}$. Se registra en `resumen_cv_clasicos.csv` y en `curvas_cnn.svg`.
- **Acuerdo entre anotadores:** $\kappa = (p_o - p_e)/(1-p_e)$.

## 7. Alternativas consideradas y descartadas

| Alternativa | Motivo para descartarla |
|-------------|-------------------------|
| Transfer learning (ResNet/MobileNet) | Lo permite la consigna, pero recomienda entrenar desde cero. Se puede dejar como experimento de comparación. |
| Detección de objetos (YOLO) | La entrada tiene una sola fruta: no hace falta localizar varias, y requiere anotar cajas. |
| Segmentación con U-Net | Exige máscaras anotadas a mano; Otsu basta con un fondo uniforme. |
| Regresión del diámetro en cm | Sin escala conocida no se puede; se usa $d_{norm}$ más clases. |
