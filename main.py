import cv2
import numpy as np


# Images URLs

# Laica 1863
# https://www.loc.gov/pictures/resource/prok.00336/?co=prok

# San Nicolas 01368
# https://www.loc.gov/pictures/collection/prok/item/2018680847/

# Soldier 01317
# https://www.loc.gov/pictures/collection/prok/item/2018681235/

def alinear_canales(canal_a_mover, canal_base, margen_busqueda=15):
    """
    Busca el mejor desplazamiento (dx, dy) para alinear 'canal_a_mover'
    con 'canal_base' minimizando la diferencia entre los píxeles.
    """
    mejor_desplazamiento = (0, 0)
    min_diferencia = float('inf')

    # Recortamos los bordes (15%)
    alto, ancho = canal_base.shape
    c_alto, c_ancho = int(alto * 0.15), int(ancho * 0.15)

    canal_base_recortado = canal_base[c_alto:-c_alto, c_ancho:-c_ancho]

    # Probamos todas las combinaciones posibles de movimiento dentro del margen
    for dy in range(-margen_busqueda, margen_busqueda + 1):
        for dx in range(-margen_busqueda, margen_busqueda + 1):

            # Desplazamos la imagen
            canal_movido = np.roll(canal_a_mover, dy, axis=0)
            canal_movido = np.roll(canal_movido, dx, axis=1)

            # Recortamos la imagen movida con los mismos márgenes
            canal_movido_recortado = canal_movido[c_alto:-c_alto, c_ancho:-c_ancho]

            # Calculamos el error (Suma de Diferencias al Cuadrado)
            # Cuanto menor sea el número, más parecidas son las imágenes
            diferencia = np.sum((np.float32(canal_base_recortado) - np.float32(canal_movido_recortado)) ** 2)

            if diferencia < min_diferencia:
                min_diferencia = diferencia
                mejor_desplazamiento = (dx, dy)

    return mejor_desplazamiento


def corregir_canal_fotometricamente(canal, percentil_bajo=5, percentil_alto=95):
    """
    Ajusta el contraste del canal recortando los valores extremos y
    reescalando el resultado al rango [0, 255].
    """
    canal_float = canal.astype(np.float32)

    minimo = np.percentile(canal_float, percentil_bajo)
    maximo = np.percentile(canal_float, percentil_alto)

    if maximo <= minimo:
        return canal.copy()

    canal_recortado = np.clip(canal_float, minimo, maximo)
    canal_normalizado = (canal_recortado - minimo) / (maximo - minimo)
    canal_normalizado = canal_normalizado * 255.0

    return canal_normalizado.astype(np.uint8)


def aplicar_correcciones_fotometricas(b, g, r):
    """
    Aplica una corrección fotométrica independiente a cada canal.
    """
    b_corregido = corregir_canal_fotometricamente(b)
    g_corregido = corregir_canal_fotometricamente(g)
    r_corregido = corregir_canal_fotometricamente(r)

    return b_corregido, g_corregido, r_corregido


def eliminar_defectos(imagen_color, umbral=30, area_maxima=4):
    """
    Elimina solo defectos pequeños y aislados sin degradar la textura general.

    Estrategia:
    - se calcula una mediana local en escala de grises
    - se detectan píxeles muy distintos de su entorno
    - se conservan solo componentes muy pequeñas
    - solo en esas zonas se sustituye por una versión filtrada
    """
    gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
    gris_mediana = cv2.medianBlur(gris, 3)

    diferencia = cv2.absdiff(gris, gris_mediana)
    mascara = (diferencia > umbral).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)

    mascara_pequena = np.zeros_like(mascara)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= area_maxima:
            mascara_pequena[labels == i] = 255

    imagen_filtrada = cv2.medianBlur(imagen_color, 3)

    imagen_limpia = imagen_color.copy()
    imagen_limpia[mascara_pequena == 255] = imagen_filtrada[mascara_pequena == 255]

    return imagen_limpia


def procesar_imagen_prokudin(ruta_imagen):
    # 1. Leer la imagen original en escala de grises
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error al cargar la imagen: {ruta_imagen}")
        return None, None, None

    # 2. Dividir la placa en tres partes iguales.
    # El orden en las placas digitalizadas es, de arriba a abajo: azul, verde y rojo
    alto = img.shape[0]
    h_tercio = alto // 3

    b = img[0:h_tercio, :]
    g = img[h_tercio:2 * h_tercio, :]
    r = img[2 * h_tercio:3 * h_tercio, :]

    # 3. Puesta en correspondencia de los canales
    # Usamos el canal Azul (B) como referencia fija y alineamos G y R hacia él.
    print(f"Procesando {ruta_imagen}...")

    dx_g, dy_g = alinear_canales(g, b)
    g_alineado = np.roll(g, dy_g, axis=0)
    g_alineado = np.roll(g_alineado, dx_g, axis=1)

    dx_r, dy_r = alinear_canales(r, b)
    r_alineado = np.roll(r, dy_r, axis=0)
    r_alineado = np.roll(r_alineado, dx_r, axis=1)

    # 4. Aglutinar la solución en una imagen a color
    # OpenCV utiliza el formato BGR por defecto para las imágenes a color
    imagen_color = cv2.merge([b, g_alineado, r_alineado])

    # 5. Generar una segunda versión con corrección fotométrica
    b_corregido, g_corregido, r_corregido = aplicar_correcciones_fotometricas(b, g_alineado, r_alineado)
    imagen_color_corregida = cv2.merge([b_corregido, g_corregido, r_corregido])

    # 6. Generar una tercera versión combinando corrección fotométrica y eliminación de defectos
    imagen_color_corregida_sin_defectos = eliminar_defectos(imagen_color_corregida)

    return imagen_color, imagen_color_corregida, imagen_color_corregida_sin_defectos


if __name__ == "__main__":
    ruta = 'laica-small.jpg'
    resultado_original, resultado_corregido, resultado_corregido_sin_defectos = procesar_imagen_prokudin(ruta)

    if resultado_original is not None and resultado_corregido is not None and resultado_corregido_sin_defectos is not None:
        cv2.imwrite('laica_color.jpg', resultado_original)
        print("Imagen original guardada como 'laica_color.jpg'.")

        cv2.imwrite('laica_color_corregida.jpg', resultado_corregido)
        print("Imagen con corrección fotométrica guardada como 'laica_color_corregida.jpg'.")

        cv2.imwrite('laica_color_corregida_sin_defectos.jpg', resultado_corregido_sin_defectos)
        print(
            "Imagen con corrección fotométrica y eliminación de defectos guardada como 'laica_color_corregida_sin_defectos.jpg'.")
