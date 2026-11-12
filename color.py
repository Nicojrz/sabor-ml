import numpy as np

BINS_MATIZ = 36
BINS_SATURACION = 16
BINS_VALOR = 16
DIMENSION_HSV = BINS_MATIZ + BINS_SATURACION + BINS_VALOR

NOMBRES_HSV = (
    tuple(f"hsv_matiz_{indice:02d}" for indice in range(BINS_MATIZ))
    + tuple(f"hsv_saturacion_{indice:02d}" for indice in range(BINS_SATURACION))
    + tuple(f"hsv_valor_{indice:02d}" for indice in range(BINS_VALOR))
)

def _normalizar_histograma(histograma: np.ndarray) -> np.ndarray:
    total = histograma.sum()
    if total > 0:
        histograma = histograma / total
    return histograma.astype(np.float32)


def histograma_hsv(imagen) -> np.ndarray:
    """Devuelve histogramas normalizados H, S y V de una imagen PIL."""
    hsv = np.asarray(imagen.convert("HSV"), dtype=np.float32)
    matiz = hsv[..., 0]
    saturacion = hsv[..., 1]
    valor = hsv[..., 2]

    histograma_matiz = np.histogram(
        matiz,
        bins=BINS_MATIZ,
        range=(0, 256),
        weights=saturacion / 255.0,
    )[0]
    histograma_saturacion = np.histogram(
        saturacion, bins=BINS_SATURACION, range=(0, 256)
    )[0]
    histograma_valor = np.histogram(
        valor, bins=BINS_VALOR, range=(0, 256)
    )[0]

    return np.concatenate(
        (
            _normalizar_histograma(histograma_matiz),
            _normalizar_histograma(histograma_saturacion),
            _normalizar_histograma(histograma_valor),
        )
    )
