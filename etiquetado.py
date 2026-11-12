"""Genera las etiquetas de sabor para todas las comidas de ``train``.

Cada puntuacion representa una intensidad independiente entre 0 y 1; por eso
las cuatro columnas no tienen que sumar 1.
"""

from pathlib import Path
import argparse
import unicodedata

import pandas as pd


EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}
COLUMNAS_SABOR = ("salado", "dulce", "amargo", "picante")


def normalizar(nombre: str) -> str:
    """Uniforma nombres con acentos para poder compararlos."""
    nombre = unicodedata.normalize("NFKD", nombre.casefold())
    return "".join(caracter for caracter in nombre if not unicodedata.combining(caracter))


# Formato de cada perfil: (salado, dulce, amargo, picante).
# Las claves corresponden a cada carpeta final presente en train/.
PERFILES_COMIDA = {
    # Amargas
    "alcachofa": (0.20, 0.05, 0.75, 0.00),
    "americano": (0.00, 0.05, 0.90, 0.00),
    "apio": (0.15, 0.05, 0.55, 0.00),
    "berenjenas": (0.15, 0.05, 0.60, 0.00),
    "browni": (0.05, 0.45, 0.70, 0.00),
    "cacao_polvo": (0.00, 0.10, 0.95, 0.00),
    "cafe": (0.00, 0.05, 0.90, 0.00),
    "cerveza": (0.05, 0.05, 0.70, 0.00),
    "chocolate_amargo": (0.00, 0.35, 0.90, 0.00),
    "col_brucelas": (0.15, 0.05, 0.65, 0.00),
    "espresso": (0.00, 0.05, 1.00, 0.00),
    "te_verde": (0.00, 0.05, 0.75, 0.00),
    "toronja": (0.00, 0.20, 0.80, 0.00),
    "vino_tinto": (0.00, 0.10, 0.70, 0.00),

    # Dulces
    "algodon_azucar": (0.00, 1.00, 0.00, 0.00),
    "arroz_con_leche": (0.05, 0.90, 0.05, 0.00),
    "brownie": (0.05, 0.95, 0.25, 0.00),
    "cheesecake": (0.10, 0.90, 0.05, 0.00),
    "churros": (0.05, 0.90, 0.00, 0.00),
    "concha": (0.05, 0.85, 0.00, 0.00),
    "crepas": (0.05, 0.80, 0.00, 0.00),
    "cupcakes": (0.05, 0.95, 0.00, 0.00),
    "donas": (0.05, 0.95, 0.00, 0.00),
    "flan": (0.05, 0.90, 0.10, 0.00),
    "gomitas": (0.00, 0.95, 0.00, 0.00),
    "helado": (0.05, 0.90, 0.00, 0.00),
    "macarons": (0.00, 0.95, 0.00, 0.00),
    "malvaviscos": (0.00, 1.00, 0.00, 0.00),
    "miel": (0.00, 1.00, 0.05, 0.00),
    "pastel_fresa": (0.05, 0.95, 0.00, 0.00),
    "sandia": (0.00, 0.75, 0.00, 0.00),
    "tiramisu": (0.05, 0.85, 0.30, 0.00),
    "tostadas_francesas": (0.10, 0.85, 0.00, 0.00),
    "waffles": (0.10, 0.85, 0.00, 0.00),

    # Picantes
    "aguachile": (0.65, 0.05, 0.05, 0.85),
    "alitas_bufalo": (0.75, 0.05, 0.05, 0.85),
    "chilaquiles_rojos": (0.70, 0.05, 0.05, 0.75),
    "chile_arbol": (0.05, 0.00, 0.15, 1.00),
    "chile_en_polvo": (0.20, 0.00, 0.10, 0.95),
    "chile_habanero": (0.05, 0.05, 0.10, 1.00),
    "chile_jalapeno": (0.10, 0.05, 0.10, 0.80),
    "chili_con_carne": (0.70, 0.10, 0.10, 0.80),
    "curry_picante": (0.55, 0.10, 0.15, 0.85),
    "kimchi": (0.70, 0.05, 0.10, 0.80),
    "mole_negro": (0.45, 0.25, 0.30, 0.70),
    "ramen_picante": (0.80, 0.05, 0.10, 0.90),
    "salsa_roja": (0.35, 0.05, 0.05, 0.85),
    "salsa_verde": (0.35, 0.05, 0.05, 0.75),
    "tteokbokki": (0.65, 0.25, 0.05, 0.85),
    "wasabi": (0.05, 0.00, 0.20, 1.00),

    # Saladas
    "aceitunas": (0.90, 0.05, 0.15, 0.00),
    "anchoas": (1.00, 0.00, 0.10, 0.00),
    "cacahuates_salados": (0.90, 0.05, 0.05, 0.00),
    "chicharron": (0.95, 0.00, 0.05, 0.00),
    "galletas_saladas": (0.85, 0.05, 0.00, 0.00),
    "jamon_serrano": (0.95, 0.05, 0.05, 0.00),
    "lasana": (0.80, 0.10, 0.05, 0.00),
    "nachos_queso": (0.90, 0.05, 0.05, 0.10),
    "palomitas": (0.85, 0.05, 0.00, 0.00),
    "papas_chips": (0.90, 0.05, 0.00, 0.00),
    "papas_francesas": (0.85, 0.05, 0.00, 0.00),
    "pepitas": (0.80, 0.05, 0.05, 0.00),
    "pretzels": (0.90, 0.05, 0.00, 0.00),
    "sal": (1.00, 0.00, 0.00, 0.00),
    "salami": (0.95, 0.05, 0.05, 0.05),
    "tocino": (0.95, 0.10, 0.05, 0.00),
}


def generar_etiquetas(carpeta_raiz: Path) -> pd.DataFrame:
    if not carpeta_raiz.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {carpeta_raiz}")

    imagenes = sorted(
        ruta
        for ruta in carpeta_raiz.rglob("*")
        if ruta.is_file() and ruta.suffix.casefold() in EXTENSIONES
    )
    clases_dataset = {normalizar(ruta.parent.name) for ruta in imagenes}
    clases_configuradas = set(PERFILES_COMIDA)

    faltantes = clases_dataset - clases_configuradas
    if faltantes:
        nombres = ", ".join(sorted(faltantes))
        raise ValueError(f"Faltan perfiles para estas comidas: {nombres}")

    sin_imagenes = clases_configuradas - clases_dataset
    if sin_imagenes:
        nombres = ", ".join(sorted(sin_imagenes))
        print(f"Aviso: perfiles sin imagenes en el dataset: {nombres}")

    filas = []
    for ruta in imagenes:
        comida = normalizar(ruta.parent.name)
        categoria = ruta.relative_to(carpeta_raiz).parts[0].casefold()
        if categoria == "salada":
            categoria = "salado"

        filas.append({
            "ruta_imagen": ruta.as_posix(),
            "categoria": categoria,
            "comida": comida,
            **dict(zip(COLUMNAS_SABOR, PERFILES_COMIDA[comida])),
        })

    return pd.DataFrame(
        filas,
        columns=("ruta_imagen", "categoria", "comida", *COLUMNAS_SABOR),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea el CSV de sabores del dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("train"))
    parser.add_argument("--salida", type=Path, default=Path("etiquetas_comida.csv"))
    args = parser.parse_args()

    df = generar_etiquetas(args.dataset)
    df.to_csv(args.salida, index=False)

    print(f"CSV generado en {args.salida} con {len(df)} imagenes.")
    print(f"Comidas etiquetadas: {df['comida'].nunique()}")
    print(df.groupby("categoria").size().to_string())


if __name__ == "__main__":
    main()
