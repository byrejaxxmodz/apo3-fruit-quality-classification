# Formulación matemática (evidencia PI3)

> Versión extendida de la formulación del informe (`report/ieee/informe.tex`, sección II-A). Las ecuaciones están en LaTeX, que GitHub renderiza.

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

### 4.1 Visualización del espacio de características (`src/evaluation/feature_space.py`)

Cada fruta es un punto $\mathbf{x} \in \mathbb{R}^{62}$. Para verlo se proyecta a 2D:

- **PCA** (no supervisado). Con $Z$ estandarizada con la media y la desviación de train, la covarianza es $\Sigma = Z^\top Z/(n-1)$ y sus vectores propios $\Sigma\mathbf{v}_j = \lambda_j \mathbf{v}_j$ ($\lambda_1 \ge \lambda_2 \ge \dots$) son las direcciones de **máxima varianza**. La proyección es $\mathbf{t} = Z[\mathbf{v}_1\ \mathbf{v}_2]$ y la fracción de varianza que conserva la componente $j$ es $\lambda_j / \sum_i \lambda_i$.
- **LDA** (supervisado, criterio de Fisher). Busca las direcciones que **separan las clases**: $\max_{\mathbf{w}} J(\mathbf{w}) = \dfrac{\mathbf{w}^\top S_B \mathbf{w}}{\mathbf{w}^\top S_W \mathbf{w}}$, con $S_B$ la dispersión entre clases y $S_W$ la dispersión dentro de cada clase. Con $K = 3$ clases hay como máximo $K - 1 = 2$ direcciones.
- **Detector de dominio.** Una regresión logística que intenta predecir la fuente (Kaggle o propia) a partir de $\mathbf{x}$, evaluada con validación cruzada agrupada. Un AUC cercano a 1 significa que las dos fuentes ocupan regiones distintas del espacio: es la condición que permite el atajo «foto propia ⇒ *regular*».

## 5. Modelos

**Regresión logística multinomial (línea base lineal):**
$$ P(y=k\mid\mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}{\sum_j e^{\mathbf{w}_j^\top \mathbf{x} + b_j}}, \quad \min_\theta -\sum_i \log P(y_i \mid \mathbf{x}_i) + \frac{1}{2C}\|\mathbf{w}\|^2 $$

**SVM con kernel RBF:**
$$ \min_{\mathbf{w}, b, \xi} \tfrac12\|\mathbf{w}\|^2 + C\sum_i \xi_i \;\; \text{s.a.}\;\; y_i(\mathbf{w}^\top\varphi(\mathbf{x}_i)+b) \ge 1-\xi_i, \qquad K(\mathbf{x},\mathbf{x}') = e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2} $$
$C$ controla el equilibrio entre margen y error, y $\gamma$ el radio de influencia de cada vector de soporte.

**Random Forest:** $\hat{y} = \text{moda}\{T_b(\mathbf{x})\}_{b=1}^{B}$. Cada árbol se entrena sobre una muestra bootstrap y divide los nodos minimizando la impureza de Gini $G = 1 - \sum_k p_k^2$. Promediar árboles poco correlacionados reduce la varianza.

**XGBoost (boosting de gradiente):** el ensamble se construye de forma **aditiva**, $\hat{\mathbf{z}}^{(m)}(\mathbf{x}) = \hat{\mathbf{z}}^{(m-1)}(\mathbf{x}) + \eta\, f_m(\mathbf{x})$, donde cada árbol $f_m$ se ajusta al gradiente (y la curvatura) de la pérdida del ensamble anterior. Minimiza un objetivo **regularizado**:
$$ \mathcal{L} = \sum_i w_i\, \ell\big(y_i, \text{softmax}(\hat{\mathbf{z}}_i)\big) + \sum_m \Big(\gamma T_m + \tfrac12 \lambda \|\mathbf{w}^{(m)}\|^2\Big) $$
con $\ell$ la entropía cruzada, $T_m$ el número de hojas y $\mathbf{w}^{(m)}$ los valores de las hojas. A diferencia de Random Forest (que **promedia** árboles independientes para bajar la varianza), el boosting **suma** árboles débiles para bajar el sesgo. Como `XGBClassifier` no tiene `class_weight`, se usan pesos por muestra $w_i = N / (K\, N_{y_i})$, los mismos de `class_weight="balanced"`.

**CNN:** convolución $(X * W)_{u,v} = \sum_{c,i,j} W_{c,i,j} X_{c,u+i,v+j} + b$, seguida de ReLU $\max(0,z)$, max-pooling y global average pooling. Se entrena con **entropía cruzada ponderada**:
$$ \mathcal{L} = -\frac{1}{N}\sum_i w_{y_i} \log \frac{e^{z_{i,y_i}}}{\sum_k e^{z_{i,k}}}, \qquad w_k = \frac{N}{K\,N_k} $$
y con Adam más weight decay ($+\lambda\|\theta\|^2$).

## 6. Selección y métricas

- **Validación cruzada estratificada y agrupada (`StratifiedGroupKFold`, k = 5)**, para que las fotos de una misma fruta queden en el mismo pliegue: $\hat{\lambda} = \arg\max_\lambda \frac{1}{k}\sum_{j=1}^{k} F1_{macro}(h_{\theta(\lambda)}^{(-j)}, D_j)$
- $P_k = \frac{TP_k}{TP_k+FP_k}$, $R_k = \frac{TP_k}{TP_k+FN_k}$, $F1_k = \frac{2P_kR_k}{P_k+R_k}$, $F1_{macro} = \frac1K\sum_k F1_k$
- **Sobreajuste:** brecha $\Delta = F1_{train} - F1_{val}$. Se registra en `resumen_cv_clasicos.csv` y en `curvas_cnn.svg`.
- **Acuerdo entre anotadores:** $\kappa = (p_o - p_e)/(1-p_e)$.

### 6.1 Incertidumbre de las métricas (`src/evaluation/intervals.py`)

Una métrica medida sobre $n$ imágenes de test es una **estimación**, no el valor real. Con las 605 fotos iniciales había solo $n_{bad} = 5$ fotos *bad* propias en test, así que un acierto más o menos cambiaba el recall de *bad* en $1/5 = 0.20$; con las 850 fotos actuales y la partición corregida hay $n_{bad} = 29$.

- **Error estándar binomial** de una proporción $\hat p = k/n$ (exactitud o recall): $SE(\hat p) = \sqrt{\hat p(1-\hat p)/n}$. Es la medida que usa la guía de transfer learning del curso para comparar estrategias.
- **Intervalo de Wilson al 95 %** ($z = 1.96$). Se prefiere a $\hat p \pm z\,SE$ porque no colapsa cuando $\hat p \in \{0, 1\}$ o $n$ es pequeño:
$$ \frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \;\pm\; \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}} $$
  Ejemplo: con 5 *bad* propias el SVM acertaba 3 ($\hat p = 0.60$, IC 95 % $[0.23,\ 0.88]$); con 29 acierta 21 ($\hat p = 0.72$, IC 95 % $[0.54,\ 0.85]$). Multiplicar $n$ por 5.8 redujo el ancho del intervalo de 0.65 a 0.31, cerca de lo que predice $SE \propto 1/\sqrt{n}$ (0.65/$\sqrt{5.8}$ ≈ 0.27).
- **Bootstrap percentil para el F1 macro**, que no tiene fórmula cerrada de varianza: se remuestrea el test con reemplazo $B = 2000$ veces, $F1^{*}_b = F1(\text{muestra}_b)$, y el IC es $[F1^{*}_{(2.5\%)},\ F1^{*}_{(97.5\%)}]$. Cuando se conoce el grupo de casi duplicados se remuestrean **grupos** completos (bootstrap por conglomerados), porque las fotos de una misma fruta no son independientes.
- **Prueba exacta de McNemar** para decidir si dos modelos $A$ y $B$ difieren sobre las mismas imágenes. Solo cuentan los desacuerdos: $b$ = imágenes que solo $A$ falla, $c$ = las que solo $B$ falla. Bajo $H_0$ (ambos se equivocan igual), $b \sim \text{Binomial}(b+c, \tfrac12)$:
$$ p = \min\!\Big(1,\; 2\sum_{i=0}^{\min(b,c)} \binom{b+c}{i} 2^{-(b+c)}\Big) $$
- **Corrección de Holm** para $m$ comparaciones simultáneas: con los p-valores ordenados $p_{(1)} \le \dots \le p_{(m)}$, $\tilde p_{(j)} = \max_{i \le j} \min\{1, (m - i + 1)\,p_{(i)}\}$. Sin ella, con 15 pares de modelos, la probabilidad de declarar al menos una diferencia falsa con $\alpha = 0.05$ sería de hasta $1 - 0.95^{15} \approx 0.54$.

## 7. Alternativas consideradas y descartadas

| Alternativa | Motivo para descartarla |
|-------------|-------------------------|
| Transfer learning (ResNet/MobileNet) | Lo permite la consigna, pero recomienda entrenar desde cero. Se puede dejar como experimento de comparación. |
| Detección de objetos (YOLO) | La entrada tiene una sola fruta: no hace falta localizar varias, y requiere anotar cajas. |
| Segmentación con U-Net | Exige máscaras anotadas a mano; Otsu basta con un fondo uniforme. |
| Regresión del diámetro en cm | Sin escala conocida no se puede; se usa $d_{norm}$ más clases. |
