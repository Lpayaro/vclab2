import cv2
import numpy as np
import time
import os

# Images URLs

# Laica 1863
# https://www.loc.gov/pictures/resource/prok.00336/?co=prok

# San Nicolas 01368
# https://www.loc.gov/pictures/collection/prok/item/2018680847/

# Soldier 01317
# https://www.loc.gov/pictures/collection/prok/item/2018681235/

def alinear_canales_avanzado(canal_a_mover, canal_base, metodo='ncc', margen_busqueda=15):
    """
    Busca el mejor desplazamiento (dx, dy) utilizando uno de los 4 métodos de correlación.
    """
    # Recortamos los bordes (15%) para evitar que los marcos negros afecten la correlación
    alto, ancho = canal_base.shape
    c_alto, c_ancho = int(alto * 0.15), int(ancho * 0.15)
    
    base_crop = canal_base[c_alto:-c_alto, c_ancho:-c_ancho].astype(np.float32)
    mover_crop = canal_a_mover[c_alto:-c_alto, c_ancho:-c_ancho].astype(np.float32)

    mejor_desplazamiento = (0, 0)

    if metodo in ['espacial', 'ncc']:
        # Extraemos una plantilla más pequeña del centro del canal a mover
        template = mover_crop[margen_busqueda:-margen_busqueda, margen_busqueda:-margen_busqueda]
        
        if metodo == 'espacial':
            # Correlación basada en convolución pura en el espacio (TM_CCORR)
            resultado = cv2.matchTemplate(base_crop, template, cv2.TM_CCORR)
        else:
            # Correlación Normalizada (TM_CCORR_NORMED)
            resultado = cv2.matchTemplate(base_crop, template, cv2.TM_CCORR_NORMED)
            
        _, _, _, max_loc = cv2.minMaxLoc(resultado)
        # Ajustamos el resultado restando el margen para permitir valores negativos
        dx = max_loc[0] - margen_busqueda
        dy = max_loc[1] - margen_busqueda
        mejor_desplazamiento = (dx, dy)

    elif metodo in ['fourier', 'fase']:
        # Aplicamos una ventana de Hanning para mitigar los artefactos en los bordes por la FFT
        hanning_y = np.hanning(base_crop.shape[0])
        hanning_x = np.hanning(base_crop.shape[1])
        ventana = np.outer(hanning_y, hanning_x)

        base_w = base_crop * ventana
        mover_w = mover_crop * ventana

        # Pasamos al dominio de Fourier
        F_base = np.fft.fft2(base_w)
        F_mover = np.fft.fft2(mover_w)

        # Correlación cruzada = F(base) * conjugado( F(mover) )
        cross_power = F_base * np.conj(F_mover)

        if metodo == 'fase':
            # Correlación de fase: normalizamos por la magnitud
            cross_power /= (np.abs(cross_power) + 1e-5)

        # Volvemos al dominio espacial
        correlacion = np.real(np.fft.ifft2(cross_power))
        correlacion = np.fft.fftshift(correlacion) # Ponemos el (0,0) en el centro

        # Buscamos el pico de correlación
        H_c, W_c = correlacion.shape
        y_max, x_max = np.unravel_index(np.argmax(correlacion), correlacion.shape)

        dy = y_max - H_c // 2
        dx = x_max - W_c // 2
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
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error al cargar la imagen: {ruta_imagen}")
        return None, None, None

    alto = img.shape[0]
    h_tercio = alto // 3

    b = img[0:h_tercio, :]
    g = img[h_tercio:2 * h_tercio, :]
    r = img[2 * h_tercio:3 * h_tercio, :]

    print(f"\nProcesando {ruta_imagen}...")
    print("-" * 50)
    
    # Benchmarking de los métodos
    metodos = ['espacial', 'ncc', 'fourier', 'fase']
    mejores_tiempos = {}
    
    print("Midiendo tiempos de alineación para el canal Verde (G):")
    for m in metodos:
        inicio = time.time()
        dx, dy = alinear_canales_avanzado(g, b, metodo=m, margen_busqueda=15)
        fin = time.time()
        tiempo_total = fin - inicio
        mejores_tiempos[m] = (tiempo_total, dx, dy)
        print(f" - {m.capitalize():10s}: {tiempo_total:.4f} seg | Desplazamiento: (dx={dx:3d}, dy={dy:3d})")

    # Escoger el método más rápido (excluyendo el 'espacial' puro si es muy errático)
    # Por lo general, NCC o Fase darán los mejores resultados reales. Usaremos 'ncc' por defecto
    # como punto de equilibrio perfecto entre velocidad y precisión.
    metodo_elegido = 'ncc' 
    print(f"\nUsando '{metodo_elegido}' para el alineado final...")

    # Alineación G
    tiempo_g, dx_g, dy_g = mejores_tiempos[metodo_elegido]
    g_alineado = np.roll(g, dy_g, axis=0)
    g_alineado = np.roll(g_alineado, dx_g, axis=1)

    # Alineación R
    dx_r, dy_r = alinear_canales_avanzado(r, b, metodo=metodo_elegido, margen_busqueda=15)
    r_alineado = np.roll(r, dy_r, axis=0)
    r_alineado = np.roll(r_alineado, dx_r, axis=1)

    imagen_color = cv2.merge([b, g_alineado, r_alineado])

    b_corregido, g_corregido, r_corregido = aplicar_correcciones_fotometricas(b, g_alineado, r_alineado)
    imagen_color_corregida = cv2.merge([b_corregido, g_corregido, r_corregido])

    imagen_color_corregida_sin_defectos = eliminar_defectos(imagen_color_corregida)

    return imagen_color, imagen_color_corregida, imagen_color_corregida_sin_defectos


if __name__ == "__main__":
    # ruta = 'laica-big.jpg'
    # ruta = 'san-nicolas-big.jpg'
    ruta = 'soldier-big.jpg'
    nombre_base = os.path.splitext(os.path.basename(ruta))[0]

    salida_original = f'{nombre_base}_color.jpg'
    salida_corregida = f'{nombre_base}_color_corregida.jpg'
    salida_corregida_sin_defectos = f'{nombre_base}_color_corregida_sin_defectos.jpg'

    resultado_original, resultado_corregido, resultado_corregido_sin_defectos = procesar_imagen_prokudin(ruta)

    if resultado_original is not None and resultado_corregido is not None and resultado_corregido_sin_defectos is not None:
        cv2.imwrite(salida_original, resultado_original)
        print(f"Imagen original guardada como '{salida_original}'.")

        cv2.imwrite(salida_corregida, resultado_corregido)
        print(f"Imagen con corrección fotométrica guardada como '{salida_corregida}'.")

        cv2.imwrite(salida_corregida_sin_defectos, resultado_corregido_sin_defectos)
        print(
            f"Imagen con corrección fotométrica y eliminación de defectos guardada como '{salida_corregida_sin_defectos}'.")
