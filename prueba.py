"""Predice sabores de una imagen o evalua una carpeta completa."""

import argparse
import os
from pathlib import Path
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

from etiquetado import PERFILES_COMIDA, normalizar


SABORES = ("Salado", "Dulce", "Amargo", "Picante")
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}
RUTA_MODELO = Path("model/modelo_sabores.joblib")


def cargar_modelos():
    modelo = joblib.load(RUTA_MODELO)
    extractor = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    return modelo, extractor


def cargar_imagen(ruta: Path) -> np.ndarray:
    img = image.load_img(ruta, target_size=(224, 224), color_mode="rgb")
    return image.img_to_array(img)


def predecir_rutas(modelo, extractor, rutas: list[Path], lote: int) -> np.ndarray:
    predicciones = []
    total_lotes = (len(rutas) + lote - 1) // lote

    for numero, inicio in enumerate(range(0, len(rutas), lote), start=1):
        rutas_lote = rutas[inicio:inicio + lote]
        imagenes = []

        for ruta in rutas_lote:
            try:
                imagenes.append(cargar_imagen(ruta))
            except Exception as error:
                raise RuntimeError(f"No se pudo cargar la imagen: {ruta}") from error

        entradas = preprocess_input(np.stack(imagenes, axis=0))
        caracteristicas = extractor.predict_on_batch(entradas)
        predicciones.append(np.clip(modelo.predict(caracteristicas), 0, 1))

        if len(rutas) > lote:
            print(
                f"\rProcesando: {numero}/{total_lotes} lotes "
                f"({inicio + len(rutas_lote)}/{len(rutas)} imagenes)",
                end="",
                flush=True,
            )

    if len(rutas) > lote:
        print()
    return np.vstack(predicciones)


def predecir_imagen(modelo, extractor, ruta: Path) -> None:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")

    prediccion = predecir_rutas(modelo, extractor, [ruta], lote=1)[0]
    print(f"\nResultados para: {ruta}")
    print("-" * 32)
    for sabor, valor in zip(SABORES, prediccion):
        print(f"{sabor:<10}: {valor:.2f}  ({valor * 100:.1f}%)")


def evaluar_carpeta(modelo, extractor, carpeta: Path, lote: int) -> None:
    if not carpeta.is_dir():
        raise NotADirectoryError(f"No existe la carpeta: {carpeta}")

    rutas = sorted(
        ruta
        for ruta in carpeta.rglob("*")
        if ruta.is_file() and ruta.suffix.casefold() in EXTENSIONES
    )
    if not rutas:
        raise ValueError(f"No se encontraron imagenes compatibles en: {carpeta}")

    comidas = [normalizar(ruta.parent.name) for ruta in rutas]
    desconocidas = sorted(set(comidas) - set(PERFILES_COMIDA))
    if desconocidas:
        nombres = ", ".join(desconocidas)
        raise ValueError(
            "No se pueden calcular metricas porque estas carpetas no tienen "
            f"un perfil real definido: {nombres}"
        )

    valores_reales = np.asarray(
        [PERFILES_COMIDA[comida] for comida in comidas], dtype=np.float32
    )
    predicciones = predecir_rutas(modelo, extractor, rutas, lote)
    errores = mean_absolute_error(
        valores_reales, predicciones, multioutput="raw_values"
    )

    clases_reales = np.argmax(valores_reales, axis=1)
    clases_predichas = np.argmax(predicciones, axis=1)
    matriz = confusion_matrix(clases_reales, clases_predichas, labels=range(4))
    exactitud = np.mean(clases_reales == clases_predichas)

    print(f"\nEvaluacion de {len(rutas)} imagenes en: {carpeta}")
    print("\nMAE por sabor:")
    for sabor, error in zip(SABORES, errores):
        print(f"  {sabor:<8}: {error:.3f}")
    print(f"  {'Promedio':<8}: {errores.mean():.3f}")
    print(f"\nExactitud del sabor dominante: {exactitud:.1%}")

    print("\nMatriz de confusion (filas=reales, columnas=predichas):")
    print(f"{'Real/Pred.':<12}" + "".join(f"{sabor:<10}" for sabor in SABORES))
    for sabor, fila in zip(SABORES, matriz):
        print(f"{sabor:<12}" + "".join(f"{valor:<10}" for valor in fila))


def resolver_ruta_interactiva(entrada: str) -> Path:
    ruta = Path(entrada).expanduser()
    if ruta.is_absolute() or ruta.is_file():
        return ruta
    return Path("test") / ruta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predice una imagen o evalua una carpeta etiquetada."
    )
    parser.add_argument(
        "--carpeta",
        type=Path,
        help="Carpeta con subcarpetas nombradas por tipo de comida.",
    )
    parser.add_argument(
        "--lote",
        type=int,
        default=32,
        help="Imagenes procesadas simultaneamente (default: 32).",
    )
    args = parser.parse_args()
    if args.lote < 1:
        parser.error("--lote debe ser mayor que cero")

    modelo, extractor = cargar_modelos()
    if args.carpeta:
        evaluar_carpeta(modelo, extractor, args.carpeta, args.lote)
        return

    entrada = input("Ingresa la ruta de la imagen a probar: ").strip()
    predecir_imagen(modelo, extractor, resolver_ruta_interactiva(entrada))


if __name__ == "__main__":
    main()
