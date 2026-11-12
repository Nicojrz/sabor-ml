"""Predice sabores de una imagen o evalua una carpeta completa."""

import argparse
import os
from pathlib import Path
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

from color import histograma_hsv
from etiquetado import PERFILES_COMIDA, normalizar


SABORES = ("Salado", "Dulce", "Amargo", "Picante")
CLASES = ("salado", "dulce", "amargo", "picante")
ALIAS_CLASES = {
    "salado": "salado",
    "salada": "salado",
    "dulce": "dulce",
    "amargo": "amargo",
    "amarga": "amargo",
    "picante": "picante",
}
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


def cargar_imagen(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    img = image.load_img(ruta, target_size=(224, 224), color_mode="rgb")
    return image.img_to_array(img), histograma_hsv(img)


def predecir_rutas(modelo, extractor, rutas: list[Path], lote: int) -> np.ndarray:
    predicciones = []
    total_lotes = (len(rutas) + lote - 1) // lote

    for numero, inicio in enumerate(range(0, len(rutas), lote), start=1):
        rutas_lote = rutas[inicio:inicio + lote]
        imagenes = []
        features_hsv = []

        for ruta in rutas_lote:
            try:
                imagen_rgb, histograma = cargar_imagen(ruta)
                imagenes.append(imagen_rgb)
                features_hsv.append(histograma)
            except Exception as error:
                raise RuntimeError(f"No se pudo cargar la imagen: {ruta}") from error

        entradas = preprocess_input(np.stack(imagenes, axis=0))
        features_cnn = extractor.predict_on_batch(entradas)
        caracteristicas = np.concatenate(
            (features_cnn, np.stack(features_hsv)), axis=1
        )
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


def mostrar_metricas(
    predicciones: np.ndarray,
    valores_reales: np.ndarray,
    clases_reales: np.ndarray,
    descripcion: str,
) -> None:
    errores = mean_absolute_error(
        valores_reales, predicciones, multioutput="raw_values"
    )
    clases_predichas = np.argmax(predicciones, axis=1)
    matriz = confusion_matrix(clases_reales, clases_predichas, labels=range(4))
    exactitud = np.mean(clases_reales == clases_predichas)

    print(f"\nEvaluacion de {len(predicciones)} imagenes en: {descripcion}")
    print("\nMAE por sabor:")
    for sabor, error in zip(SABORES, errores):
        print(f"  {sabor:<8}: {error:.3f}")
    print(f"  {'Promedio':<8}: {errores.mean():.3f}")
    print(f"\nExactitud del sabor dominante: {exactitud:.1%}")

    print("\nMatriz de confusion (filas=reales, columnas=predichas):")
    print(f"{'Real/Pred.':<12}" + "".join(f"{sabor:<10}" for sabor in SABORES))
    for sabor, fila in zip(SABORES, matriz):
        print(f"{sabor:<12}" + "".join(f"{valor:<10}" for valor in fila))


def evaluar_con_csv(
    modelo,
    extractor,
    carpeta: Path,
    archivo_csv: Path,
    lote: int,
) -> None:
    if not archivo_csv.is_file():
        raise FileNotFoundError(f"No existe el CSV de respuestas: {archivo_csv}")

    df = pd.read_csv(archivo_csv)
    columnas = {"archivo", "clase", "alimento"}
    faltantes = columnas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas en {archivo_csv}: {', '.join(sorted(faltantes))}"
        )
    if df.empty:
        raise ValueError(f"El CSV no contiene respuestas: {archivo_csv}")
    if df[list(columnas)].isna().any().any():
        raise ValueError("El CSV contiene valores vacios en archivo, clase o alimento")

    df = df.copy()
    for columna in columnas:
        df[columna] = df[columna].astype(str).str.strip()
    duplicados = df.loc[df["archivo"].duplicated(), "archivo"].unique().tolist()
    if duplicados:
        raise ValueError(f"Hay archivos duplicados en el CSV: {', '.join(duplicados[:5])}")

    rutas = [carpeta / nombre for nombre in df["archivo"]]
    inexistentes = [str(ruta) for ruta in rutas if not ruta.is_file()]
    if inexistentes:
        raise FileNotFoundError(
            "Hay imagenes del CSV que no existen, por ejemplo: "
            + ", ".join(inexistentes[:5])
        )

    comidas = [normalizar(nombre) for nombre in df["alimento"]]
    desconocidas = sorted(set(comidas) - set(PERFILES_COMIDA))
    if desconocidas:
        raise ValueError(
            "Estos alimentos no tienen perfil en etiquetado.py: "
            + ", ".join(desconocidas)
        )

    clases_texto = [ALIAS_CLASES.get(normalizar(clase)) for clase in df["clase"]]
    clases_desconocidas = sorted(
        {clase for clase, valor in zip(df["clase"], clases_texto) if valor is None}
    )
    if clases_desconocidas:
        raise ValueError(
            "Clases desconocidas en el CSV: " + ", ".join(clases_desconocidas)
        )

    valores_reales = np.asarray(
        [PERFILES_COMIDA[comida] for comida in comidas], dtype=np.float32
    )
    indices_clase = np.asarray([CLASES.index(clase) for clase in clases_texto])
    predicciones = predecir_rutas(modelo, extractor, rutas, lote)
    mostrar_metricas(
        predicciones,
        valores_reales,
        indices_clase,
        f"{carpeta} usando {archivo_csv}",
    )


def evaluar_por_subcarpetas(modelo, extractor, carpeta: Path, lote: int) -> None:
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
        raise ValueError(
            "Estas carpetas no tienen un perfil en etiquetado.py: "
            + ", ".join(desconocidas)
        )

    valores_reales = np.asarray(
        [PERFILES_COMIDA[comida] for comida in comidas], dtype=np.float32
    )
    clases_reales = np.argmax(valores_reales, axis=1)
    predicciones = predecir_rutas(modelo, extractor, rutas, lote)
    mostrar_metricas(
        predicciones, valores_reales, clases_reales, str(carpeta)
    )


def evaluar_carpeta(
    modelo,
    extractor,
    carpeta: Path,
    archivo_csv: Path | None,
    lote: int,
) -> None:
    if not carpeta.is_dir():
        raise NotADirectoryError(f"No existe la carpeta: {carpeta}")

    csv_predeterminado = carpeta / "respuestas_prueba.csv"
    if archivo_csv is not None:
        evaluar_con_csv(modelo, extractor, carpeta, archivo_csv, lote)
    elif csv_predeterminado.is_file():
        evaluar_con_csv(modelo, extractor, carpeta, csv_predeterminado, lote)
    else:
        evaluar_por_subcarpetas(modelo, extractor, carpeta, lote)


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
        help="Carpeta que contiene las imagenes de evaluacion.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV con las columnas archivo, clase y alimento.",
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
        evaluar_carpeta(modelo, extractor, args.carpeta, args.csv, args.lote)
        return
    if args.csv:
        parser.error("--csv requiere tambien --carpeta")

    entrada = input("Ingresa la ruta de la imagen a probar: ").strip()
    predecir_imagen(modelo, extractor, resolver_ruta_interactiva(entrada))


if __name__ == "__main__":
    main()
