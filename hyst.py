import threading
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk, ImageEnhance
import time
from numba import jit  # Numba importálása
import cv2
from concurrent.futures import ThreadPoolExecutor


# Globális változók a kép és a fényerő kezeléséhez
original_image = Image.open('photo.jfif')  # Kép fájl neve
brightness_value = 1  # Kezdeti fényerő érték (1 = eredeti)
contrast_value = 1  # Kezdeti fényerő érték (1 = eredeti)

bui_avail = 1
sec_avail = 1
simd_avail = 1
multi_avail = 1

brg_sec_times = []
brg_bui_times = []
brg_simd_times = []  
brg_multi_times = []

con_sec_times = []
con_bui_times = []
con_simd_times = []  
con_multi_times = []

sharp_sec_times = []
sharp_bui_times = []
sharp_simd_times = []  
sharp_multi_times = []

########################
### Brightness ###
def adjust_brightness_builtin(image, brightness):
    # A fényerő módosítása beépített függvénnyel
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness)

def adjust_brightness_sec(image, brightness):
    # A fényerő módosítása pixel szinten
    image_array = np.array(image)
    # A fényerő módosítása: az új pixelértékek
    adjusted_image_array = np.clip(image_array + (brightness), 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted_image_array)

@jit(nopython=True)  # Numba JIT alkalmazása az SIMD hatékonyság érdekében
def adjust_brightness_simd_array(image_array, brightness):
    # A fényerő módosítása SIMD-szerű működéssel közvetlenül a NumPy tömbön
    return np.clip(image_array * brightness, 0, 255).astype(np.uint8)

def adjust_brightness_simd(image, brightness):
    image_array = np.array(image)
    adjusted_image_array = adjust_brightness_simd_array(image_array, brightness)
    return Image.fromarray(adjusted_image_array)


def adjust_brightness_multi(image, value):
    image_array = np.array(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("A bemeneti képnek RGB formátumban kell lennie.")

    height, width, _ = image_array.shape
    new_image_array = np.empty_like(image_array)

    # Function to process a block of pixels
    def process_block(start_row, end_row):
        block_result = np.empty((end_row - start_row, width, 3), dtype=np.uint8)
        for y in range(start_row, end_row):
            for x in range(width):
                r, g, b = image_array[y, x]
                new_r = min(int(r * value), 255)
                new_g = min(int(g * value), 255)
                new_b = min(int(b * value), 255)
                block_result[y - start_row, x] = (new_r, new_g, new_b)
        return block_result

    # Determine number of threads to use
    num_threads = min(8, height)  # Limit to a reasonable number of threads
    rows_per_thread = height // num_threads
    blocks = [(i * rows_per_thread, (i + 1) * rows_per_thread if i < num_threads - 1 else height) for i in range(num_threads)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda block: process_block(*block), blocks))

    # Combine results into the final image array
    for i, (start_row, end_row) in enumerate(blocks):
        new_image_array[start_row:end_row] = results[i]

    return Image.fromarray(new_image_array)
#######################
### Contrast ###
def adjust_contrast_builtin(image, contrast):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast)

def adjust_contrast_sec(image, contrast):
    image_array = np.array(image).astype(np.float32)  # Típus konverzió lebegőpontos számokra
    midpoint = 128
    adjusted_image_array = np.clip(midpoint + contrast * (image_array - midpoint), 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted_image_array)

@jit(nopython=True)  # JIT fordítás Numba-val
def adjust_contrast_simd_array(image_array, contrast):
    midpoint = 128
    adjusted_image_array = np.empty_like(image_array, dtype=np.uint8)
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            adjusted_value = midpoint + contrast * (image_array[i, j] - midpoint)
            adjusted_image_array[i, j] = np.clip(adjusted_value, 0, 255)

    return adjusted_image_array

def adjust_contrast_simd(image, contrast):
    image_array = np.array(image)
    adjusted_image_array = adjust_contrast_simd_array(image_array, contrast)
    return Image.fromarray(adjusted_image_array)

def adjust_contrast_segment(segment, contrast, midpoint):
    """A kép egy részének kontrasztját módosítja."""
    adjusted_segment = np.empty_like(segment, dtype=np.uint8)
    for i in range(segment.shape[0]):
        for j in range(segment.shape[1]):
            adjusted_value = midpoint + contrast * (segment[i, j] - midpoint)
            adjusted_segment[i, j] = np.clip(adjusted_value, 0, 255)
    return adjusted_segment

def adjust_contrast_multi(image, contrast):
    """A kép kontrasztját multithreading segítségével módosítja."""
    # A kép NumPy tömbbé konvertálása
    image_array = np.array(image).astype(np.float32)  # Típus konverzió lebegőpontos számokra
    midpoint = 128  # Kép középértéke

    # A kép felosztása sávokra
    height, width = image_array.shape[:2]
    num_segments = 4  # A szegmensek száma
    segments = np.array_split(image_array, num_segments)  # Felosztás a szegmensekre

    # Párhuzamos feldolgozás a ThreadPoolExecutor segítségével
    def process_segment(segment):
        """Feldolgozza a szegmenst a kontraszt beállításához."""
        adjusted_segment = np.empty_like(segment, dtype=np.uint8)
        for i in range(segment.shape[0]):
            for j in range(segment.shape[1]):
                adjusted_value = midpoint + contrast * (segment[i, j] - midpoint)
                adjusted_segment[i, j] = np.clip(adjusted_value, 0, 255)
        return adjusted_segment

    with ThreadPoolExecutor() as executor:
        # Minden szegmenst párhuzamosan dolgozunk fel
        futures = [executor.submit(process_segment, segment) for segment in segments]
        adjusted_segments = [future.result() for future in futures]

    # Az összes feldolgozott szegmenst egyesítjük
    adjusted_image_array = np.vstack(adjusted_segments)

    # A módosított NumPy tömböt visszaalakítjuk képpé
    return Image.fromarray(adjusted_image_array)

#######################
### Vignette ###

#######################
### Sharpness ###
def adjust_sharpness_builtin(image, sharpness):
    # Az élesség módosítása beépített függvénnyel
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(sharpness)

def adjust_sharpness_sec(image, sharpness_factor):
    # Kép megnyitása és átalakítása numpy tömbbé
    img_array = np.array(image)

    # Élesség kernel definiálása
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + sharpness_factor, -1],
                       [0, -1, 0]])

    # Kernel alkalmazása a kép minden csatornájára
    img_sharpened = np.zeros_like(img_array)
    for i in range(3):  # RGB csatornák
        img_sharpened[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, kernel)

    # Sharpness factor skálázása a nagyobb hatás érdekében
    if sharpness_factor < 100:
        factor = (sharpness_factor / 100) * 1.5  # Tompítás erősebb hatással (1.5-tel skálázva)
    else:
        factor = ((sharpness_factor - 100) / 100) * 5000  # Élesítés erősebb hatással (5-tel skálázva)

    # Keverjük az eredeti képet és az élesített képet a sharpness_factor alapján
    output_array = cv2.addWeighted(img_array, 1 - factor, img_sharpened, factor, 0)

    # Kép visszaalakítása és mentése
    image_sharpened = Image.fromarray(np.uint8(output_array))
    return image_sharpened







@jit(nopython=True)
def clip(value, min_value, max_value):
    # A bemeneti értékek klippelése a megadott tartományon belül
    clipped = np.empty_like(value)
    for i in range(value.size):
        if value[i] < min_value:
            clipped[i] = min_value
        elif value[i] > max_value:
            clipped[i] = max_value
        else:
            clipped[i] = value[i]
    return clipped

@jit(nopython=True)
def sharpen_kernel(img_array, kernel, output_array):
    height, width, channels = img_array.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for c in range(channels):
                # Kernel alkalmazása
                pixel_value = (
                    kernel[0, 0] * img_array[i - 1, j - 1, c] + kernel[0, 1] * img_array[i - 1, j, c] + kernel[0, 2] * img_array[i - 1, j + 1, c] +
                    kernel[1, 0] * img_array[i, j - 1, c] + kernel[1, 1] * img_array[i, j, c] + kernel[1, 2] * img_array[i, j + 1, c] +
                    kernel[2, 0] * img_array[i + 1, j - 1, c] + kernel[2, 1] * img_array[i + 1, j, c] + kernel[2, 2] * img_array[i + 1, j + 1, c]
                )
                output_array[i, j, c] = clip(np.array([pixel_value]), 0, 255)[0]

def adjust_sharpness_simd(image, sharpness_factor):
    img_array = np.array(image, dtype=np.float32)

    # Élesség kernel definiálása
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + sharpness_factor, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Élesített kép inicializálása
    img_sharpened = np.zeros_like(img_array)

    # Kernel alkalmazása
    sharpen_kernel(img_array, kernel, img_sharpened)

    # Sharpness factor skálázása
    if sharpness_factor < 100:
        factor = (sharpness_factor / 100) * 1.5  # Tompítás erősebb hatással
    else:
        factor = ((sharpness_factor - 100) / 100) * 5  # Élesítés erősebb hatással

    # Keverjük az eredeti képet és az élesített képet
    output_array = np.zeros_like(img_array)
    for c in range(img_array.shape[2]):
        # Készítjük a kimeneti képet
        combined = img_array[:, :, c] * (1 - factor) + img_sharpened[:, :, c] * factor
        output_array[:, :, c] = clip(combined, 0, 255)

    return Image.fromarray(output_array.astype(np.uint8))


def adjust_sharpness_multi(image, sharpness_value):
    import numpy as np

    # Az élesség értéke alapján egy szűrőt definiálunk
    sharpness_factor = sharpness_value / 100.0
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + sharpness_factor, -1],
                       [0, -1, 0]])

    # Kép konvertálása numpy tömbbé
    image_array = np.array(image)

    # A képméret
    height, width, channels = image_array.shape
    new_image = np.zeros((height, width, channels), dtype=np.float64)  # float64 típusú tömb

    # Kép élesítése
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # A konvolúció kiszámítása
            for k in range(-1, 2):
                for l in range(-1, 2):
                    new_image[i, j] += image_array[i + k, j + l] * kernel[k + 1, l + 1]
                    
            # Határértékek beállítása 0-255 között
            new_image[i, j] = np.clip(new_image[i, j], 0, 255)

    return Image.fromarray(new_image.astype('uint8'))



#######################
def update_brightness():
    global brightness_value
    # A fényerő beállítása a csúszka értéke alapján
    brightness_value = brightness_slider.get() / 100.0  # Normálás 0 és 2 közötti értékre (1 = alap)

    # Beépített függvény kép
    if bui_avail > 0:
        start_time_builtin = time.time()
        builtin_image = adjust_brightness_builtin(original_image, brightness_value)
        builtin_time = time.time() - start_time_builtin
        brg_bui_times.append(builtin_time)
        display_image(builtin_image, builtin_canvas)
        update_histogram(builtin_image, builtin_hist_canvas)

    # Szekvenciális fényerő állítás
    if sec_avail > 0:
        start_time_sec = time.time()
        sequential_image = adjust_brightness_sec(original_image, brightness_value)
        sequential_time = time.time() - start_time_sec
        brg_sec_times.append(sequential_time)
        display_image(sequential_image, sequential_canvas)
        update_histogram(sequential_image, sequential_hist_canvas)

    # SIMD fényerő állítás (Numba JIT segítségével)
    if simd_avail > 0:
        start_time_simd = time.time()
        simd_image = adjust_brightness_simd(original_image, brightness_value)
        simd_time = time.time() - start_time_simd
        brg_simd_times.append(simd_time)
        display_image(simd_image, simd_canvas)
        update_histogram(simd_image, simd_hist_canvas)
        
    if multi_avail > 0:
        start_time_multi = time.time()
        multi_image = adjust_brightness_multi(original_image, brightness_value)
        multi_time = time.time() - start_time_multi
        brg_multi_times.append(multi_time)
        display_image(multi_image, multi_canvas)
        update_histogram(multi_image, multi_hist_canvas)
    
    display_image(original_image, original_canvas)
    update_histogram(original_image, original_hist_canvas)


    # Táblázat frissítése a feldolgozási idővel
    update_table()

def update_contrast():
    global contrast_value
    # A kontraszt beállítása a csúszka értéke alapján
    contrast_value = contrast_slider.get() / 100.0  # Normálás 0 és 2 közötti értékre (1 = alap)

    # Beépített függvény kép
    if bui_avail > 0:
        start_time_builtin = time.time()
        builtin_image = adjust_contrast_builtin(original_image, contrast_value)
        builtin_time = time.time() - start_time_builtin
        con_bui_times.append(builtin_time)
        display_image(builtin_image, builtin_canvas)
        update_histogram(builtin_image, builtin_hist_canvas)

    # Szekvenciális fényerő állítás
    if sec_avail > 0:
        start_time_sec = time.time()
        sequential_image = adjust_contrast_sec(original_image, contrast_value)
        sequential_time = time.time() - start_time_sec
        con_sec_times.append(sequential_time)
        display_image(sequential_image, sequential_canvas)
        update_histogram(sequential_image, sequential_hist_canvas)

    # SIMD fényerő állítás (Numba JIT segítségével)
    if simd_avail > 0:
        start_time_simd = time.time()
        simd_image = adjust_contrast_simd(original_image, contrast_value)
        simd_time = time.time() - start_time_simd
        con_simd_times.append(simd_time)
        display_image(simd_image, simd_canvas)
        update_histogram(simd_image, simd_hist_canvas)
        
    if multi_avail > 0:
        start_time_multi = time.time()
        multi_image = adjust_contrast_multi(original_image, contrast_value)
        multi_time = time.time() - start_time_multi
        con_multi_times.append(multi_time)
        display_image(multi_image, multi_canvas)
        update_histogram(multi_image, multi_hist_canvas)
    
    display_image(original_image, original_canvas)
    update_histogram(original_image, original_hist_canvas)

    update_table()

def update_sharpness():
    global sharpness_value
    # A kontraszt beállítása a csúszka értéke alapján
    sharpness_value = sharpness_slider.get() / 100.0  # Normálás 0 és 2 közötti értékre (1 = alap)

    # Beépített függvény kép
    if bui_avail > 0:
        start_time_builtin = time.time()
        builtin_image = adjust_sharpness_builtin(original_image, sharpness_value)
        builtin_time = time.time() - start_time_builtin
        sharp_bui_times.append(builtin_time)
        display_image(builtin_image, builtin_canvas)
        update_histogram(builtin_image, builtin_hist_canvas)
    

    # Szekvenciális fényerő állítás
    if sec_avail > 0:
        start_time_sec = time.time()
        sequential_image = adjust_sharpness_sec(original_image, sharpness_value)
        sequential_time = time.time() - start_time_sec
        sharp_sec_times.append(sequential_time)
        display_image(sequential_image, sequential_canvas)
        update_histogram(sequential_image, sequential_hist_canvas)

    """    

    # SIMD fényerő állítás (Numba JIT segítségével)
    if simd_avail > 0:
        start_time_simd = time.time()
        simd_image = adjust_sharpness_simd(original_image, sharpness_value)
        simd_time = time.time() - start_time_simd
        sharp_simd_times.append(simd_time)
        display_image(simd_image, simd_canvas)
        update_histogram(simd_image, simd_hist_canvas)
    """
        
    if multi_avail > 0:
        start_time_multi = time.time()
        multi_image = adjust_sharpness_multi(original_image, sharpness_value)
        multi_time = time.time() - start_time_multi
        sharp_multi_times.append(multi_time)
        display_image(multi_image, multi_canvas)
        update_histogram(multi_image, multi_hist_canvas)


    display_image(original_image, original_canvas)
    update_histogram(original_image, original_hist_canvas)

    update_table()

def display_image(image, canvas):
    image_tk = ImageTk.PhotoImage(image.resize((240, 240)))  # Kisebb méret (240x240)
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    canvas.image = image_tk

def update_histogram(image, canvas):
    # Hisztogram generálása
    canvas.delete("all")  # Törli a korábbi hisztogramot
    histogram = image.histogram()
    colors = ('red', 'green', 'blue')
    
    # Hisztogram háttér színének beállítása
    canvas.configure(bg='lightgrey')  # Világosszürke háttér

    for i, color in enumerate(colors):
        hist_data = histogram[i * 256:(i + 1) * 256]
        
        if not hist_data:  # If hist_data is empty, skip to avoid errors
            continue

        max_value = max(hist_data)
        if max_value == 0:  # Avoid division by zero
            max_value = 1
        
        # Hisztogram vonalak rajzolása
        canvas.create_line([(x, 100 - y / max_value * 100) for x, y in enumerate(hist_data)], fill=color)

def update_table():
    for row in table.get_children():
        table.delete(row)

    # Fényerő    
    # Átlagos idő kiszámítása
    avg_time_brg_bui = sum(brg_bui_times[2:]) / len(brg_bui_times[2:]) if len(brg_bui_times) > 2 else 0
    avg_time_brg_sec = sum(brg_sec_times[2:]) / len(brg_sec_times[2:]) if len(brg_sec_times) > 2 else 0
    avg_time_brg_simd = sum(brg_simd_times[2:]) / len(brg_simd_times[2:]) if len(brg_simd_times) > 2 else 0
    avg_time_brg_multi = sum(brg_multi_times[2:]) / len(brg_multi_times[2:]) if len(brg_multi_times) > 2 else 0

    if len(brg_bui_times) > 2:
        table.insert('', 'end', values=("Fényerő beépített", f"{brg_bui_times[-1]:.6f} mp", f"{avg_time_brg_bui:.6f} mp"))
    if len(brg_sec_times) > 2:
        table.insert('', 'end', values=("Fényerő szekvenciális", f"{brg_sec_times[-1]:.6f} mp", f"{avg_time_brg_sec:.6f} mp"))
    if len(brg_simd_times) > 2:
        table.insert('', 'end', values=("Fényerő SIMD", f"{brg_simd_times[-1]:.6f} mp", f"{avg_time_brg_simd:.6f} mp"))
    if len(brg_multi_times) > 2:
        table.insert('', 'end', values=("Fényerő multi", f"{brg_multi_times[-1]:.6f} mp", f"{avg_time_brg_multi:.6f} mp"))


    # Kontraszt
    # Átlagos idő kiszámítása
    avg_time_con_bui = sum(con_bui_times[2:]) / len(con_bui_times[2:]) if len(con_bui_times) > 2 else 0
    avg_time_con_sec = sum(con_sec_times[2:]) / len(con_sec_times[2:]) if len(con_sec_times) > 2 else 0
    avg_time_con_simd = sum(con_simd_times[2:]) / len(con_simd_times[2:]) if len(con_simd_times) > 2 else 0
    avg_time_con_multi = sum(con_multi_times[2:]) / len(con_multi_times[2:]) if len(con_multi_times) > 2 else 0

    if len(con_bui_times) > 2:
        table.insert('', 'end', values=("Kontraszt beépített", f"{con_bui_times[-1]:.6f} mp", f"{avg_time_con_bui:.6f} mp"))
    if len(con_sec_times) > 2:
        table.insert('', 'end', values=("Kontraszt szekvenciális", f"{con_sec_times[-1]:.6f} mp", f"{avg_time_con_sec:.6f} mp"))
    if len(con_simd_times) > 2:
        table.insert('', 'end', values=("Kontraszt SIMD", f"{con_simd_times[-1]:.6f} mp", f"{avg_time_con_simd:.6f} mp"))
    if len(con_multi_times) > 2:
        table.insert('', 'end', values=("Kontraszt multi", f"{con_multi_times[-1]:.6f} mp", f"{avg_time_con_multi:.6f} mp"))
   
    # Sharpness
    # Átlagos idő kiszámítása
    avg_time_sharp_bui = sum(sharp_bui_times[2:]) / len(sharp_bui_times[2:]) if len(sharp_bui_times) > 2 else 0
    avg_time_sharp_sec = sum(sharp_sec_times[2:]) / len(sharp_sec_times[2:]) if len(sharp_sec_times) > 2 else 0
    avg_time_sharp_simd = sum(sharp_simd_times[2:]) / len(sharp_simd_times[2:]) if len(sharp_simd_times) > 2 else 0
    avg_time_sharp_multi = sum(sharp_multi_times[2:]) / len(sharp_multi_times[2:]) if len(sharp_multi_times) > 2 else 0

    if len(sharp_bui_times) > 2:
        table.insert('', 'end', values=("Élesség beépített", f"{sharp_bui_times[-1]:.6f} mp", f"{avg_time_sharp_bui:.6f} mp"))
    if len(sharp_sec_times) > 2:
        table.insert('', 'end', values=("Élesség szekvenciális", f"{sharp_sec_times[-1]:.6f} mp", f"{avg_time_sharp_sec:.6f} mp"))
    if len(sharp_simd_times) > 2:
        table.insert('', 'end', values=("Élesség SIMD", f"{sharp_simd_times[-1]:.6f} mp", f"{avg_time_sharp_simd:.6f} mp"))
    if len(sharp_multi_times) > 2:
        table.insert('', 'end', values=("Élesség multi", f"{sharp_multi_times[-1]:.6f} mp", f"{avg_time_sharp_multi:.6f} mp"))
 


# Főablak létrehozása
root = Tk()
root.title("Kép szerkesztés")

# Teljes képernyős mód bekapcsolása
root.attributes('-fullscreen', True)

# Bezárás ESC gomb vagy Ctrl+C megnyomásával
root.bind("<Escape>", lambda e: root.destroy())
root.bind("<Control-c>", lambda e: root.destroy())

# Háttérszín beállítása sötétszürkére
root.configure(bg='#2e2e2e')

# Eredeti kép megjelenítése
original_canvas = Canvas(root, width=240, height=240, bg='#2e2e2e', highlightthickness=0)
original_canvas.grid(row=0, column=0, padx=20, pady=20)
original_hist_canvas = Canvas(root, width=240, height=100, bg='lightgrey', highlightthickness=0)
original_hist_canvas.grid(row=1, column=0, padx=20, pady=20)

# Beépített függvény kép megjelenítése
builtin_canvas = Canvas(root, width=240, height=240, bg='#2e2e2e', highlightthickness=0)
builtin_canvas.grid(row=0, column=1, padx=20, pady=20)
builtin_hist_canvas = Canvas(root, width=240, height=100, bg='lightgrey', highlightthickness=0)
builtin_hist_canvas.grid(row=1, column=1, padx=20, pady=20)

# Szekvenciális kép megjelenítése
sequential_canvas = Canvas(root, width=240, height=240, bg='#2e2e2e', highlightthickness=0)
sequential_canvas.grid(row=0, column=2, padx=20, pady=20)
sequential_hist_canvas = Canvas(root, width=240, height=100, bg='lightgrey', highlightthickness=0)
sequential_hist_canvas.grid(row=1, column=2, padx=20, pady=20)

# SIMD kép megjelenítése
simd_canvas = Canvas(root, width=240, height=240, bg='#2e2e2e', highlightthickness=0)
simd_canvas.grid(row=0, column=3, padx=20, pady=20)
simd_hist_canvas = Canvas(root, width=240, height=100, bg='lightgrey', highlightthickness=0)
simd_hist_canvas.grid(row=1, column=3, padx=20, pady=20)

# Multithreading kép megjelenítése
multi_canvas = Canvas(root, width=240, height=240, bg='#2e2e2e', highlightthickness=0)
multi_canvas.grid(row=0, column=4, padx=20, pady=20)
multi_hist_canvas = Canvas(root, width=240, height=100, bg='lightgrey', highlightthickness=0)
multi_hist_canvas.grid(row=1, column=4, padx=20, pady=20)

# Csúszka a fényerő állításához
brightness_slider = Scale(root, from_=0, to=200, orient=HORIZONTAL, command=lambda _: update_brightness(), label="Fényerő")
brightness_slider.set(100)
brightness_slider.grid(row=4, column=1, columnspan=5, pady=20, padx=80, sticky=W)

# Kontraszt csúszka hozzáadása
contrast_slider = Scale(root, from_=0, to=200, orient=HORIZONTAL, command=lambda _: update_contrast(), label="Kontraszt")
contrast_slider.set(100)
contrast_slider.grid(row=4, column=2, columnspan=5, pady=20, padx=80, sticky=W)

# Élesség csúszka hozzáadása
sharpness_slider = Scale(root, from_=0, to=200, orient=HORIZONTAL, command=lambda _: update_sharpness(), label="Élesség")
sharpness_slider.set(100)
sharpness_slider.grid(row=4, column=3, columnspan=5, pady=20, padx=80, sticky=W)


# Táblázat a feldolgozási idők megjelenítéséhez
table_frame = Frame(root, bg='#2e2e2e')
table_frame.grid(row=6, column=0, columnspan=6, pady=20)
table = ttk.Treeview(table_frame, columns=("Algoritmus", "Idő", "Átlagos idő"), show="headings", height=10)
table.heading("Algoritmus", text="Algoritmus")
table.heading("Idő", text="Idő (mp)")
table.heading("Átlagos idő", text="Átlagos idő (mp)")
table.pack()

Label(root, text="Eredeti", bg='#2e2e2e', fg='white').grid(row=2, column=0)
Label(root, text="Beépített", bg='#2e2e2e', fg='white').grid(row=2, column=1)
Label(root, text="Szekvenciális", bg='#2e2e2e', fg='white').grid(row=2, column=2)
Label(root, text="SIMD", bg='#2e2e2e', fg='white').grid(row=2, column=3)
Label(root, text="Multi", bg='#2e2e2e', fg='white').grid(row=2, column=4)


def update_button_texts():
    builtin_text = f"Beépített {'kikapcsolva' if bui_avail < 0 else 'bekapcsolva'}"
    sequential_text = f"Szekvenciális {'kikapcsolva' if sec_avail < 0 else 'bekapcsolva'}"
    multi_text = f"Multi {'kikapcsolva' if multi_avail < 0 else 'bekapcsolva'}"
    simd_text = f"SIMD {'kikapcsolva' if simd_avail < 0 else 'bekapcsolva'}"

    # Gombok szövegének frissítése
    builtin_button.config(text=builtin_text)
    sequential_button.config(text=sequential_text)
    simd_button.config(text=simd_text)
    multi_button.config(text=multi_text)

def toggle_builtin():
    global bui_avail
    bui_avail *= -1  # Ki-/bekapcsolás
    update_button_texts()
    update_brightness()  # Frissítés szükséges a kapcsolt állapot alapján

def toggle_sequential():
    global sec_avail
    sec_avail *= -1  # Ki-/bekapcsolás
    update_button_texts()
    update_brightness()  # Frissítés szükséges a kapcsolt állapot alapján

def toggle_simd():
    global simd_avail
    simd_avail *= -1  # Ki-/bekapcsolás
    update_button_texts()
    update_brightness()  # Frissítés szükséges a kapcsolt állapot alapján

def toggle_multi():
    global multi_avail
    multi_avail *= -1  # Ki-/bekapcsolás
    update_button_texts()
    update_brightness()  # Frissítés szükséges a kapcsolt állapot alapján

button_frame = Frame(root, bg='#2e2e2e')
button_frame.grid(row=3, column=1, columnspan=5, pady=20)

builtin_button = Button(button_frame, text="Beépített bekapcsolva", command=toggle_builtin)
builtin_button.pack(side=LEFT, padx=80)

sequential_button = Button(button_frame, text="Szekvenciális bekapcsolva", command=toggle_sequential)
sequential_button.pack(side=LEFT, padx=80)

simd_button = Button(button_frame, text="SIMD bekapcsolva", command=toggle_simd)
simd_button.pack(side=LEFT, padx=80)

multi_button = Button(button_frame, text="Multi bekapcsolva", command=toggle_multi)
multi_button.pack(side=LEFT, padx=80)

# Első frissítés
update_brightness()
update_contrast()
update_sharpness()

# Fő program futtatása
root.mainloop()
