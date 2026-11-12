# Predicción de sabores en imágenes de comida

Este proyecto estima la intensidad de cuatro sabores en una imagen de comida: salado, dulce, amargo y picante. Las predicciones se expresan con valores entre 0 y 1.

El proyecto utiliza MobileNetV2 para extraer características visuales y RidgeCV para aprender los perfiles de sabor del dataset.

## Preparar el entorno

El proyecto utiliza Python 3.10 y puede ejecutarse dentro de un entorno de Anaconda.

```bash
conda create -n sabor_ml python=3.10 -y
conda activate sabor_ml
conda install pandas numpy scikit-learn pillow -y
```

En una Mac con Apple Silicon instala TensorFlow con soporte para Metal.

```bash
pip install tensorflow-macos tensorflow-metal
```

En otros equipos instala la versión estándar de TensorFlow.

```bash
pip install tensorflow
```

## Dataset

Las imágenes de entrenamiento se encuentran en `train`. Cada carpeta final representa un tipo de comida y las carpetas principales indican su sabor predominante.

```text
train/
├── amarga/
├── dulce/
├── picante/
└── salada/
```

El dataset actual contiene 66 tipos de comida y 2,105 imágenes.

## Generar las etiquetas

`etiquetado.py` recorre el dataset y crea `etiquetas_comida.csv`. Cada comida recibe un perfil independiente de salado, dulce, amargo y picante.

```bash
python etiquetado.py
```

También se pueden indicar rutas diferentes para el dataset y el archivo resultante.

```bash
python etiquetado.py --dataset train --salida etiquetas_comida.csv
```

## Entrenar el modelo

`modelo.py` procesa las imágenes por lotes, entrena el regresor y guarda los archivos generados dentro de `model`.

```bash
python modelo.py
```

Las características visuales se guardan en caché. Las siguientes ejecuciones reutilizan esta caché y no vuelven a procesar todas las imágenes.

Para ignorar la caché y extraer nuevamente las características ejecuta:

```bash
python modelo.py --recalcular
```

El tamaño del lote se puede modificar si el equipo tiene más o menos memoria disponible.

```bash
python modelo.py --lote 16
```

El modelo entrenado se guarda en:

```text
model/modelo_sabores.joblib
```

## Probar una imagen

Coloca una imagen nueva dentro de la carpeta `test` y ejecuta `prueba.py`.

```bash
python prueba.py
```

El programa solicitará el nombre del archivo.

```text
Ingresa la ruta de la imagen a probar: comida.jpg
```

La salida muestra la intensidad estimada de cada sabor.

```text
Resultados para: ./test/comida.jpg
--------------------------------
Salado    : 0.72  (72.0%)
Dulce     : 0.08  (8.0%)
Amargo    : 0.15  (15.0%)
Picante   : 0.81  (81.0%)
```

## Evaluar una carpeta

`prueba.py` también puede evaluar todas las imágenes de `test` usando las respuestas de `test/respuestas_prueba.csv`. El resultado muestra el MAE por sabor, la exactitud del sabor dominante y una matriz de confusión.

```bash
python prueba.py --carpeta test
```

El CSV debe contener una fila por imagen y las columnas `archivo`, `clase` y `alimento`.

```csv
archivo,clase,alimento
prueba_001.jpg,amarga,berenjenas
prueba_002.jpg,picante,ramen_picante
```

Cuando el CSV tiene otro nombre o se encuentra en otra ruta se puede indicar de forma explícita.

```bash
python prueba.py --carpeta test --csv test/respuestas_prueba.csv
```

El MAE compara los valores estimados con el perfil configurado para cada alimento. La matriz de confusión utiliza la columna `clase` como respuesta real y la compara con el sabor dominante predicho.

El tamaño del lote puede ajustarse durante la evaluación.

```bash
python prueba.py --carpeta test --lote 16
```

## Flujo completo

Ejecuta los scripts en este orden cuando cambies el dataset o los perfiles de sabor.

```bash
python etiquetado.py
python modelo.py --recalcular
python prueba.py
```
