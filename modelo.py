import argparse
import os
from pathlib import Path
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

SABORES = ("salado", "dulce", "amargo", "picante")
ARCHIVO_ETIQUETAS = Path("etiquetas_comida.csv")
CARPETA_MODELO = Path("model")
ARCHIVO_FEATURES = CARPETA_MODELO / "features_mobilenetv2.npy"
ARCHIVO_RUTAS = CARPETA_MODELO / "features_mobilenetv2_rutas.txt"
ARCHIVO_MODELO = CARPETA_MODELO / "modelo_sabores.joblib"
FEATURES_ANTERIORES = Path("features_500.npy")
DIMENSION_FEATURES = 1280

def cache_valida(archivo: Path, rutas: list[str]) -> bool:
    if not archivo.exists():
        return False

    try:
        features = np.load(archivo, mmap_mode="r")
    except (OSError, ValueError):
        return False

    if features.shape != (len(rutas), DIMENSION_FEATURES):
        return False

    if archivo == ARCHIVO_FEATURES and ARCHIVO_RUTAS.exists():
        rutas_cache = ARCHIVO_RUTAS.read_text(encoding="utf-8").splitlines()
        return rutas_cache == rutas
    
    return archivo.stat().st_mtime >= ARCHIVO_ETIQUETAS.stat().st_mtime

def extraer_features(rutas: list[str], tamano_lote: int) -> np.ndarray:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.utils import img_to_array, load_img

    extractor = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    features = np.empty((len(rutas), DIMENSION_FEATURES), dtype=np.float32)
    total_lotes = (len(rutas) + tamano_lote - 1) // tamano_lote

    for numero_lote, inicio in enumerate(range(0, len(rutas), tamano_lote), start=1):
        rutas_lote = rutas[inicio:inicio + tamano_lote]
        imagenes = []

        for ruta in rutas_lote:
            try:
                imagen = load_img(ruta, target_size=(224, 224), color_mode="rgb")
                imagenes.append(img_to_array(imagen))
            except Exception as error:
                raise RuntimeError(f"No se pudo cargar la imagen: {ruta}") from error

        lote = preprocess_input(np.stack(imagenes, axis=0))
        features[inicio:inicio + len(rutas_lote)] = extractor.predict_on_batch(lote)
        print(
            f"\rExtrayendo features: {numero_lote}/{total_lotes} lotes "
            f"({inicio + len(rutas_lote)}/{len(rutas)} imagenes)",
            end="",
            flush=True,
        )

    print()
    guardar_cache(features, rutas)
    return features

def guardar_cache(features: np.ndarray, rutas: list[str]) -> None:
    CARPETA_MODELO.mkdir(parents=True, exist_ok=True)
    np.save(ARCHIVO_FEATURES, features)
    ARCHIVO_RUTAS.write_text("\n".join(rutas) + "\n", encoding="utf-8")
    print(f"Features guardadas en {ARCHIVO_FEATURES}.")

def obtener_features(
    rutas: list[str], tamano_lote: int, recalcular: bool
) -> np.ndarray:
    if not recalcular and cache_valida(ARCHIVO_FEATURES, rutas):
        print(f"Reutilizando {ARCHIVO_FEATURES}; no se procesaran las imagenes.")
        return np.load(ARCHIVO_FEATURES)

    if not recalcular and cache_valida(FEATURES_ANTERIORES, rutas):
        print(f"Migrando la cache anterior {FEATURES_ANTERIORES}...")
        features = np.load(FEATURES_ANTERIORES)
        guardar_cache(features, rutas)
        return features

    return extraer_features(rutas, tamano_lote)

def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de sabores.")
    parser.add_argument(
        "--lote",
        type=int,
        default=32,
        help="Imagenes procesadas a la vez por MobileNetV2 (default: 32).",
    )
    parser.add_argument(
        "--recalcular",
        action="store_true",
        help="Ignora la cache y vuelve a extraer todas las features.",
    )
    args = parser.parse_args()
    if args.lote < 1:
        parser.error("--lote debe ser mayor que cero")

    df = pd.read_csv(ARCHIVO_ETIQUETAS)
    columnas_requeridas = {"ruta_imagen", *SABORES}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(sorted(faltantes))}")

    rutas = df["ruta_imagen"].astype(str).tolist()
    rutas_inexistentes = [ruta for ruta in rutas if not Path(ruta).is_file()]
    if rutas_inexistentes:
        ejemplo = ", ".join(rutas_inexistentes[:3])
        raise FileNotFoundError(f"Hay rutas de imagen inexistentes, por ejemplo: {ejemplo}")

    y = df[list(SABORES)].to_numpy(dtype=np.float32)
    X = obtener_features(rutas, args.lote, args.recalcular)

    estratificar = df["comida"] if "comida" in df.columns else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=estratificar,
    )

    print(f"Entrenando RidgeCV con {len(X_train)} imagenes...")
    modelo = RidgeCV(alphas=[1.0, 10.0, 50.0, 100.0, 500.0], cv=5)
    modelo.fit(X_train, y_train)

    predicciones = np.clip(modelo.predict(X_test), 0, 1)
    errores = mean_absolute_error(y_test, predicciones, multioutput="raw_values")
    print("\nEvaluacion del modelo:")
    for sabor, error in zip(SABORES, errores):
        print(f"  {sabor.capitalize():<8}: MAE +/-{error:.3f}")
    print(f"  Alpha seleccionado: {modelo.alpha_}")

    CARPETA_MODELO.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ARCHIVO_MODELO)
    print(f"\nModelo guardado en {ARCHIVO_MODELO}.")

if __name__ == "__main__":
    main()
