# Predicción de sabores con imágenes

Este proyecto estima la intensidad de cuatro sabores en una imagen de comida: salado, dulce, amargo y picante. Cada resultado es un valor entre 0 y 1.

Las imágenes se representan con 1,280 características de MobileNetV2 y 68 características de color HSV. El modelo final es un SVR con kernel RBF y una salida independiente para cada sabor.

## Entorno

El proyecto utiliza Python 3.10 dentro de un entorno de Anaconda.

```bash
conda create -n sabor_ml python=3.10 -y
conda activate sabor_ml
conda install pandas numpy scikit-learn pillow -y
```

En una Mac con Apple Silicon instala TensorFlow con soporte para Metal.

```bash
pip install tensorflow-macos tensorflow-metal
```

En otros equipos instala TensorFlow estándar.

```bash
pip install tensorflow
```

## Estructura

```text
sabor-ml/
├── train/
├── test/
├── model/
├── color.py
├── etiquetado.py
├── modelo.py
└── prueba.py
```

`train` contiene las imágenes de entrenamiento organizadas por sabor y alimento. `test` contiene las imágenes de evaluación y su CSV de respuestas. `model` almacena las características extraídas y el modelo entrenado.

## Etiquetas

`etiquetado.py` recorre `train`, asigna el perfil de cada alimento y genera `etiquetas_comida.csv`.

```bash
python etiquetado.py
```

Cada fila del CSV contiene la ruta de la imagen, su categoría, el alimento y sus valores de salado, dulce, amargo y picante.

## Características de color

`color.py` calcula histogramas normalizados de matiz, saturación y valor.

```text
36 columnas de matiz
16 columnas de saturación
16 columnas de valor
```

El matiz se pondera por saturación para dar más importancia a colores intensos. Estas 68 columnas se combinan con las características de MobileNetV2.

## Entrenamiento

`modelo.py` extrae las características, entrena el SVR con kernel RBF y guarda el modelo en `model/modelo_sabores.joblib`.

```bash
python modelo.py
```

Las características se guardan en caché para evitar procesar nuevamente todas las imágenes.

```text
model/features_mobilenetv2_hsv.npy
model/features_mobilenetv2_hsv_rutas.txt
model/features_mobilenetv2_hsv_columnas.txt
```

Para reconstruir la caché completa ejecuta:

```bash
python modelo.py --recalcular
```

El tamaño del lote puede ajustarse según la memoria disponible.

```bash
python modelo.py --lote 16
```

## Probar una imagen

Coloca la imagen dentro de `test` y ejecuta:

```bash
python prueba.py
```

El programa solicitará el nombre de la imagen y mostrará los cuatro valores predichos.

```text
Ingresa la ruta de la imagen a probar: prueba_001.jpg
```

## Evaluar la carpeta de prueba

`prueba.py` puede evaluar todas las imágenes de `test` usando `test/respuestas_prueba.csv`.

```bash
python prueba.py --carpeta test
```

El CSV debe contener las columnas `archivo`, `clase` y `alimento`.

```csv
archivo,clase,alimento
prueba_001.jpg,amarga,berenjenas
prueba_002.jpg,picante,ramen_picante
```

La evaluación muestra el MAE de cada sabor, el MAE promedio, la exactitud del sabor dominante y la matriz de confusión.

También se puede indicar el CSV de manera explícita.

```bash
python prueba.py --carpeta test --csv test/respuestas_prueba.csv
```

## Flujo completo

Cuando cambie el dataset o los perfiles de sabor, ejecuta:

```bash
python etiquetado.py
python modelo.py --recalcular
python prueba.py --carpeta test
```
