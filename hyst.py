import time
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import Scale, Text, StringVar, OptionMenu
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Kép betöltése
def load_image(file_path):
    img = Image.open(file_path).convert('RGB')
    return img

# Fényerő módosítás szekvenciálisan
def adjust_brightness_sequential(img, brightness):
    pixels = np.array(img)
    factor = brightness / 100.0
    pixels = np.clip(pixels * (1 + factor), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

# Kontraszt módosítás szekvenciálisan
def adjust_contrast_sequential(img, contrast):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(1 + contrast / 100.0)

# Fényerő módosítás SIMD-vel (NumPy)
def adjust_brightness_simd(img, brightness):
    pixels = np.array(img)
    factor = brightness / 100.0
    pixels = np.clip(pixels * (1 + factor), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

# Kontraszt módosítás SIMD-vel
def adjust_contrast_simd(img, contrast):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(1 + contrast / 100.0)

# Multithreading fényerő és kontraszt módosításokhoz
def adjust_brightness_multithreading(img, brightness):
    pixels = np.array(img)
    factor = brightness / 100.0

    def process_chunk(start, end):
        pixels[start:end] = np.clip(pixels[start:end] * (1 + factor), 0, 255)

    chunk_size = len(pixels) // 4  # Négyszálas feldolgozás
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(pixels), chunk_size):
            executor.submit(process_chunk, i, min(i + chunk_size, len(pixels)))

    return Image.fromarray(pixels.astype(np.uint8))

def adjust_contrast_multithreading(img, contrast):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(1 + contrast / 100.0)

def update_display(fig, canvas, original, sequential, simd, multithreading):
    fig.clear()
    ax1, ax2, ax3, ax4 = fig.subplots(1, 4)

    ax1.imshow(original)
    ax1.set_title("Original")
    ax1.axis('off')

    ax2.imshow(sequential)
    ax2.set_title("Sequential")
    ax2.axis('off')

    ax3.imshow(simd)
    ax3.set_title("SIMD")
    ax3.axis('off')

    ax4.imshow(multithreading)
    ax4.set_title("Multithreading")
    ax4.axis('off')

    canvas.draw()

def measure_time(func, *args, repeats=10):
    start = time.perf_counter()
    for _ in range(repeats):
        func(*args)
    end = time.perf_counter()
    return (end - start) / repeats

def brightness_slider():
    root = tk.Tk()
    root.title("Brightness and Contrast Adjustment")

    fig = plt.Figure(figsize=(10, 3), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    time_text = Text(root, height=15, width=70)
    time_text.pack()

    # Változók a jelenlegi fényerő és kontraszt értékek tárolásához
    current_brightness = 0
    current_contrast = 0

    # Kép betöltése
    image_options = ["photo.jpg", "photo.jfif", "photo.png"]
    selected_image = StringVar(root)
    selected_image.set(image_options[0])

    img = load_image(selected_image.get())

    # Képek tárolása a módosítások után
    sequential_img_brightness = img
    simd_img_brightness = img
    multithreading_img_brightness = img

    sequential_img_contrast = img
    simd_img_contrast = img
    multithreading_img_contrast = img

    # Fényerő módosító függvény
    def adjust_brightness_only(value):
        nonlocal current_brightness, current_contrast
        nonlocal sequential_img_brightness, simd_img_brightness, multithreading_img_brightness
        nonlocal sequential_img_contrast, simd_img_contrast, multithreading_img_contrast
        current_brightness = slider_brightness.get()

        # Fényerő módosítás és időmérés
        seq_time_brightness = measure_time(lambda: adjust_brightness_sequential(img, current_brightness))
        sequential_img_brightness = adjust_brightness_sequential(img, current_brightness)

        simd_time_brightness = measure_time(lambda: adjust_brightness_simd(img, current_brightness))
        simd_img_brightness = adjust_brightness_simd(img, current_brightness)

        mt_time_brightness = measure_time(lambda: adjust_brightness_multithreading(img, current_brightness))
        multithreading_img_brightness = adjust_brightness_multithreading(img, current_brightness)

        # Ha van kontraszt módosítás, azt alkalmazzuk a fényerő módosított képekre
        if current_contrast != 0:
            sequential_img_contrast = adjust_contrast_sequential(sequential_img_brightness, current_contrast)
            simd_img_contrast = adjust_contrast_simd(simd_img_brightness, current_contrast)
            multithreading_img_contrast = adjust_contrast_multithreading(multithreading_img_brightness, current_contrast)
        else:
            sequential_img_contrast = sequential_img_brightness
            simd_img_contrast = simd_img_brightness
            multithreading_img_contrast = multithreading_img_brightness

        # Képfrissítés
        update_display(fig, canvas, img, sequential_img_contrast, simd_img_contrast, multithreading_img_contrast)

        # Időeredmények kiírása
        table_text = (
            f"{'Method':<25}{'Current (s)':<15}\n"
            f"{'-'*40}\n"
            f"{'Brightness Sequential':<25}{seq_time_brightness:.6f}\n"
            f"{'Brightness SIMD':<25}{simd_time_brightness:.6f}\n"
            f"{'Brightness Multithreading':<25}{mt_time_brightness:.6f}\n"
        )

        time_text.delete(1.0, tk.END)
        time_text.insert(tk.END, table_text)

    # Kontraszt módosító függvény
    def adjust_contrast_only(value):
        nonlocal current_contrast
        nonlocal sequential_img_brightness, simd_img_brightness, multithreading_img_brightness
        nonlocal sequential_img_contrast, simd_img_contrast, multithreading_img_contrast
        current_contrast = slider_contrast.get()

        # Kontraszt módosítás és időmérés
        seq_time_contrast = measure_time(lambda: adjust_contrast_sequential(sequential_img_brightness, current_contrast))
        sequential_img_contrast = adjust_contrast_sequential(sequential_img_brightness, current_contrast)

        simd_time_contrast = measure_time(lambda: adjust_contrast_simd(simd_img_brightness, current_contrast))
        simd_img_contrast = adjust_contrast_simd(simd_img_brightness, current_contrast)

        mt_time_contrast = measure_time(lambda: adjust_contrast_multithreading(multithreading_img_brightness, current_contrast))
        multithreading_img_contrast = adjust_contrast_multithreading(multithreading_img_brightness, current_contrast)

        # Képfrissítés
        update_display(fig, canvas, img, sequential_img_contrast, simd_img_contrast, multithreading_img_contrast)

        # Időeredmények kiírása
        table_text = (
            f"{'Method':<25}{'Current (s)':<15}\n"
            f"{'-'*40}\n"
            f"{'Contrast Sequential':<25}{seq_time_contrast:.6f}\n"
            f"{'Contrast SIMD':<25}{simd_time_contrast:.6f}\n"
            f"{'Contrast Multithreading':<25}{mt_time_contrast:.6f}\n"
        )

        time_text.delete(1.0, tk.END)
        time_text.insert(tk.END, table_text)

    # Fényerő csúszka
    slider_brightness = Scale(root, from_=-100, to=100, orient="horizontal", label="Brightness", command=adjust_brightness_only)
    slider_brightness.set(0)
    slider_brightness.pack()

    # Kontraszt csúszka
    slider_contrast = Scale(root, from_=-100, to=100, orient="horizontal", label="Contrast", command=adjust_contrast_only)
    slider_contrast.set(0)
    slider_contrast.pack()

    # Képválasztó legördülő menü
    def on_image_change(*args):
        nonlocal img, current_brightness, current_contrast
        nonlocal sequential_img_brightness, simd_img_brightness, multithreading_img_brightness
        nonlocal sequential_img_contrast, simd_img_contrast, multithreading_img_contrast
        img = load_image(selected_image.get())
        slider_brightness.set(0)
        slider_contrast.set(0)
        current_brightness = 0
        current_contrast = 0

        # Képek alaphelyzetbe állítása
        sequential_img_brightness = img
        simd_img_brightness = img
        multithreading_img_brightness = img

        sequential_img_contrast = img
        simd_img_contrast = img
        multithreading_img_contrast = img

        # Képfrissítés
        update_display(fig, canvas, img, img, img, img)
        time_text.delete(1.0, tk.END)

    image_menu = OptionMenu(root, selected_image, *image_options, command=on_image_change)
    image_menu.pack()

    # Kezdeti kép megjelenítése
    update_display(fig, canvas, img, img, img, img)

    root.mainloop()

if __name__ == "__main__":
    brightness_slider()
