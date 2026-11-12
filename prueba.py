import os
import warnings

# Debe configurarse antes de importar TensorFlow. Oculta los mensajes internos
# de TensorFlow/Metal, pero no los resultados ni los errores del programa.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# 1. Cargar el modelo clásico guardado y el extractor visual
modelo = joblib.load('./model/modelo_sabores.joblib')
extractor = MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3),
)

def predecir_sabor(ruta_imagen):
    # Preparar la imagen igual que durante el entrenamiento
    img = image.load_img(ruta_imagen, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    # Extraer las 1,280 características visuales
    caracteristicas = extractor.predict(arr, verbose=0).flatten().reshape(1, -1)
    
    # Predecir y limitar los resultados estrictamente entre 0 y 1
    prediccion = np.clip(modelo.predict(caracteristicas)[0], 0, 1)
    
    # Mostrar resultados ordenados
    sabores = ['Salado', 'Dulce', 'Amargo', 'Picante']
    print(f"\nResultados para: {ruta_imagen}")
    print("-" * 32)
    for sabor, valor in zip(sabores, prediccion):
        print(f"{sabor:<10}: {valor:.2f}  ({valor * 100:.1f}%)")

# --- PRUEBA TU MODELO AQUÍ ---
# Coloca una foto nueva en tu carpeta y escribe su nombre exacto:
prompt_img = input('Ingresa la ruta de la imagen a probar: ')
prompt_img = './test/' + prompt_img
predecir_sabor(prompt_img)
