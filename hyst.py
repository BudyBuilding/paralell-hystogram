import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from numba import njit

current_brightness = 0

def adjust_brightness_sequential(img, brightness):
    pixels = np.array(img)
    factor =  brightness / 100.0
    pixels = np.clip(pixels * (1 + factor), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

def adjust_brightness_simd(img, brightness):
    pixels = np.array(img, dtype=np.float32)
    factor = brightness / 100.0
    pixels *= (1 + factor)
    np.clip(pixels, 0, 255, out=pixels)
    return Image.fromarray(pixels.astype(np.uint8))

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

def calculate_contrast(img):
    pixels = np.array(img.convert('L'))  # Convert to grayscale
    mean_brightness = np.mean(pixels)
    contrast = np.std(pixels)
    return contrast

def adjust_contrast_sequential(img, contrast):
    pixels = np.array(img, dtype=np.float32)
    mean_brightness = np.mean(pixels)
    factor = contrast / 100.0
    pixels = np.clip((pixels - mean_brightness) * (1 + factor) + mean_brightness, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

def adjust_contrast_simd(img, contrast):
    pixels = np.array(img, dtype=np.float32)
    mean_brightness = np.mean(pixels)
    factor = contrast / 100.0
    pixels = np.clip((pixels - mean_brightness) * (1 + factor) + mean_brightness, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

def adjust_contrast_multithreading(img, contrast):
    pixels = np.array(img, dtype=np.float32)
    mean_brightness = np.mean(pixels)
    factor = contrast / 100.0

    def process_chunk(start, end):
        pixels[start:end] = np.clip((pixels[start:end] - mean_brightness) * (1 + factor) + mean_brightness, 0, 255)

    chunk_size = len(pixels) // 4  # Négyszálas feldolgozás
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(pixels), chunk_size):
            executor.submit(process_chunk, i, min(i + chunk_size, len(pixels)))

    return Image.fromarray(pixels.astype(np.uint8))

def adjust_vignette_sequential(img, vignette_strength):
    width, height = img.size
    vignette_mask = Image.new("L", (width, height), 0)
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)

    for y in range(height):
        for x in range(width):
            # Távolság a középponttól
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Maszk értékének számítása: középtől távolodva nő
            vignette_value = (distance / max_distance) * 255
            
            # Vignette erősség alkalmazása: -100 esetén fekete, +100 esetén fehér
            vignette_value = np.clip(vignette_value * (vignette_strength / 100), 0, 255)
            vignette_mask.putpixel((x, y), int(vignette_value))

    # Alkalmazzuk a vignette maszkot a képre
    vignette_mask = vignette_mask.resize(img.size)
    return Image.composite(img, Image.new("RGB", img.size, "black"), vignette_mask)

def adjust_vignette_simd(img, vignette_strength):
    width, height = img.size
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Létrehozunk egy 2D numpy tömböt a távolságokkal
    y, x = np.ogrid[:height, :width]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Vignette maszk számítása
    vignette_values = (distances / max_distance) * 255
    
    # Vignette erősség alkalmazása
    vignette_values = np.clip(vignette_values * (vignette_strength / 100), 0, 255).astype(np.uint8)
    
    vignette_mask = Image.fromarray(vignette_values, mode="L")
    return Image.composite(img, Image.new("RGB", img.size, "black"), vignette_mask)

def adjust_vignette_multithreading(img, vignette_strength):
    width, height = img.size
    vignette_mask = Image.new("L", (width, height), 0)
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    pixels = np.array(vignette_mask)

    def process_chunk(start, end):
        for y in range(start, end):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                vignette_value = (distance / max_distance) * 255
                
                # Vignette erősség alkalmazása
                vignette_value = np.clip(vignette_value * (vignette_strength / 100), 0, 255)
                pixels[y, x] = int(vignette_value)

    chunk_size = height // 4  # Feldolgozás 4 szálon
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, height, chunk_size):
            executor.submit(process_chunk, i, min(i + chunk_size, height))

    vignette_mask = Image.fromarray(pixels)
    return Image.composite(img, Image.new("RGB", img.size, "black"), vignette_mask)

def adjust_sharpnes_sequential(img, intensity):
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + intensity / 20, -1],  # Intensity adjustment
                       [0, -1, 0]])
    
    img_array = np.array(img)
    kernel_height, kernel_width = kernel.shape
    height, width = img_array.shape[:2]
    new_img_array = np.zeros((height, width, 3), dtype=np.uint8)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_img = np.pad(img_array, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='edge')

    for y in range(height):
        for x in range(width):
            region = padded_img[y:y + kernel_height, x:x + kernel_width]
            new_img_array[y, x] = np.clip(np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1)), 0, 255)

    return Image.fromarray(new_img_array)

@njit
def adjust_sharpness_simd(img, intensity):
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + intensity / 20, -1],  # Intensity adjustment
                       [0, -1, 0]])
    img_array = img.copy()
    kernel_height, kernel_width = kernel.shape
    height, width = img_array.shape[:2]
    new_img_array = np.zeros((height, width, 3), dtype=np.uint8)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_img = np.pad(img_array, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='edge')

    for y in range(height):
        for x in range(width):
            region = padded_img[y:y + kernel_height, x:x + kernel_width]
            new_img_array[y, x] = np.clip(np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1)), 0, 255)

    return new_img_array

def adjust_sharpness_multithreading(img, intensity):
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + intensity / 20, -1],  # Intensity adjustment
                       [0, -1, 0]])
    img_array = np.array(img)
    kernel_height, kernel_width = kernel.shape
    height, width = img_array.shape[:2]
    new_img_array = np.zeros((height, width, 3), dtype=np.uint8)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_img = np.pad(img_array, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='edge')

    def process_pixel(y):
        for x in range(width):
            region = padded_img[y:y + kernel_height, x:x + kernel_width]
            new_img_array[y, x] = np.clip(np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1)), 0, 255)

    with ThreadPoolExecutor() as executor:
        executor.map(process_pixel, range(height))

    return Image.fromarray(new_img_array)

# Histogram calculation function
def calculate_histogram(img):
    pixels = np.array(img)
    if len(pixels.shape) == 3:  # RGB Image
        histogram_r = [0] * 256
        histogram_g = [0] * 256
        histogram_b = [0] * 256
        for row in pixels:
            for pixel in row:
                histogram_r[pixel[0]] += 1
                histogram_g[pixel[1]] += 1
                histogram_b[pixel[2]] += 1
        return histogram_r, histogram_g, histogram_b
    else:  # Grayscale image
        histogram = [0] * 256
        for row in pixels:
            for pixel in row:
                histogram[pixel] += 1
        return histogram
    
# Function to draw the histogram on the canvas
def draw_histogram(canvas, histogram_r, histogram_g, histogram_b, width=256, height=100):
    canvas.delete("all")
    
    # Normalize histogram values to fit within the canvas height
    max_value_r = max(histogram_r)
    max_value_g = max(histogram_g)
    max_value_b = max(histogram_b)
    
    for i in range(256):
        # Normalize the heights
        height_r = (histogram_r[i] / max_value_r) * height if max_value_r > 0 else 0
        height_g = (histogram_g[i] / max_value_g) * height if max_value_g > 0 else 0
        height_b = (histogram_b[i] / max_value_b) * height if max_value_b > 0 else 0

        # Draw the red channel histogram
        canvas.create_rectangle(i, height, i + 1, height - height_r, fill="red", outline="", stipple="gray50")
        
        # Draw the green channel histogram
        canvas.create_rectangle(i, height, i + 1, height - height_g, fill="green", outline="", stipple="gray50")
        
        # Draw the blue channel histogram
        canvas.create_rectangle(i, height, i + 1, height - height_b, fill="blue", outline="", stipple="gray50")

    # Add grid lines for better readability
    for j in range(0, height, 10):  # Horizontal lines every 10 pixels
        canvas.create_line(0, height - j, width, height - j, fill="lightgray", dash=(2, 2))
    
    # Add x-axis labels
    for j in range(0, 256, 32):  # Label every 32 pixels
        canvas.create_text(j, height + 10, text=str(j), fill="black")

    # Add y-axis label
    canvas.create_text(-20, height // 2, text="Frequency", fill="black", angle=90)

class EditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Brightness and Contrast Adjuster")

        # Load the image
        self.original_image = Image.open("./photo.jpg")
        self.processed_image_seq = self.original_image.copy()
        self.processed_image_simd = self.original_image.copy()
        self.processed_image_mt = self.original_image.copy()

        # Convert images to PhotoImage
        self.original_photo = ImageTk.PhotoImage(self.original_image)
        self.processed_photo_seq = ImageTk.PhotoImage(self.processed_image_seq)
        self.processed_photo_simd = ImageTk.PhotoImage(self.processed_image_simd)
        self.processed_photo_mt = ImageTk.PhotoImage(self.processed_image_mt)

        # Create labels to display images
        self.original_label = tk.Label(root, image=self.original_photo)
        self.original_label.grid(row=1, column=0, padx=10, pady=10)

        self.processed_label_seq = tk.Label(root, image=self.processed_photo_seq)
        self.processed_label_seq.grid(row=1, column=1, padx=10, pady=10)

        self.processed_label_simd = tk.Label(root, image=self.processed_photo_simd)
        self.processed_label_simd.grid(row=1, column=2, padx=10, pady=10)

        self.processed_label_mt = tk.Label(root, image=self.processed_photo_mt)
        self.processed_label_mt.grid(row=1, column=3, padx=10, pady=10)

        # Create buttons to toggle methods
        self.seq_button = tk.Button(root, text="Toggle Sequential", command=self.toggle_sequential)
        self.seq_button.grid(row=2, column=1, pady=10)

        self.simd_button = tk.Button(root, text="Toggle SIMD", command=self.toggle_simd)
        self.simd_button.grid(row=2, column=2, pady=10)

        self.mt_button = tk.Button(root, text="Toggle Multithreaded", command=self.toggle_multithreaded)
        self.mt_button.grid(row=2, column=3, pady=10)

        # Create sliders for brightness and contrast adjustment
        self.brightness_slider = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.adjust_brightness, label="Brightness Intensity")
        self.brightness_slider.grid(row=3, column=0, columnspan=3, pady=10)
        self.brightness_slider.set(0)

        self.contrast_slider = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.adjust_contrast, label="Contrast Intensity")
        self.contrast_slider.grid(row=3, column=1, columnspan=3, pady=10)
        self.contrast_slider.set(0)

        # Create slider for vignette adjustment
        self.vignette_slider = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.adjust_vignette, label="Vignette Intensity")
        self.vignette_slider.grid(row=4, column=0, columnspan=3, pady=10)
        self.vignette_slider.set(0)

        # Create slider for sharpness adjustment
        self.intensity_scale = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.adjust_sharpness, label="Sharpness Intensity")
        self.intensity_scale.grid(row=4, column=1, columnspan=3, pady=10)
        self.intensity_scale.set(0)  


        # Create a table to display processing time and contrast
        self.table = ttk.Treeview(root, columns=("Operation", "Time (s)", "Avg Time (s)"), show="headings")
        self.table.heading("Operation", text="Operation")
        self.table.heading("Time (s)", text="Time (s)")
        self.table.heading("Avg Time (s)", text="Avg Time (s)")
        self.table.grid(row=5, column=0, columnspan=4, pady=10)

        self.times_seq = []
        self.times_simd = []
        self.times_mt = []

        self.seq_enabled = True
        self.simd_enabled = True
        self.mt_enabled = True

        # Create canvas for histograms
        self.histogram_canvas_original = tk.Canvas(root, width=256, height=100)
        self.histogram_canvas_original.grid(row=0, column=0, padx=5, pady=5)

        self.histogram_canvas_seq = tk.Canvas(root, width=256, height=100)
        self.histogram_canvas_seq.grid(row=0, column=1, padx=5, pady=5)

        self.histogram_canvas_simd = tk.Canvas(root, width=256, height=100)
        self.histogram_canvas_simd.grid(row=0, column=2, padx=5, pady=5)

        self.histogram_canvas_mt = tk.Canvas(root, width=256, height=100)
        self.histogram_canvas_mt.grid(row=0, column=3, padx=5, pady=5)


        # Initial histogram plot for the original image
        self.update_histogram(self.original_image, self.histogram_canvas_original)


    def toggle_sequential(self):
        self.seq_enabled = not self.seq_enabled

    def toggle_simd(self):
        self.simd_enabled = not self.simd_enabled

    def toggle_multithreaded(self):
        self.mt_enabled = not self.mt_enabled

    def adjust_brightness(self, value):
        value = int(value)
        self.table.delete(*self.table.get_children())

        if self.seq_enabled:
            # Sequential brightness adjustment
            start_time_seq = time.perf_counter()
            self.processed_image_seq = adjust_brightness_sequential(self.original_image, value)
            end_time_seq = time.perf_counter()
            elapsed_time_seq = end_time_seq - start_time_seq
            self.times_seq.append(elapsed_time_seq)
            avg_time_seq = sum(self.times_seq) / len(self.times_seq)
            contrast_seq = calculate_contrast(self.processed_image_seq)
            self.processed_photo_seq = ImageTk.PhotoImage(self.processed_image_seq)
            self.processed_label_seq.config(image=self.processed_photo_seq)
            self.processed_label_seq.image = self.processed_photo_seq
            self.table.insert("", "end", values=("Sequential Brightness", f"{elapsed_time_seq:.5f}", f"{avg_time_seq:.5f}", f"{contrast_seq:.5f}"))
            self.update_histogram(self.processed_image_seq, self.histogram_canvas_seq)

        if self.simd_enabled:
            # SIMD brightness adjustment
            start_time_simd = time.perf_counter()
            self.processed_image_simd = adjust_brightness_simd(self.original_image, value)
            end_time_simd = time.perf_counter()
            elapsed_time_simd = end_time_simd - start_time_simd
            self.times_simd.append(elapsed_time_simd)
            avg_time_simd = sum(self.times_simd) / len(self.times_simd)
            contrast_simd = calculate_contrast(self.processed_image_simd)
            self.processed_photo_simd = ImageTk.PhotoImage(self.processed_image_simd)
            self.processed_label_simd.config(image=self.processed_photo_simd)
            self.processed_label_simd.image = self.processed_photo_simd
            self.table.insert("", "end", values=("SIMD Brightness", f"{elapsed_time_simd:.5f}", f"{avg_time_simd:.5f}", f"{contrast_simd:.5f}"))
            self.update_histogram(self.processed_image_simd, self.histogram_canvas_simd)

        if self.mt_enabled:
            # Multithreaded brightness adjustment
            start_time_mt = time.perf_counter()
            self.processed_image_mt = adjust_brightness_multithreading(self.original_image, value)
            end_time_mt = time.perf_counter()
            elapsed_time_mt = end_time_mt - start_time_mt
            self.times_mt.append(elapsed_time_mt)
            avg_time_mt = sum(self.times_mt) / len(self.times_mt)
            contrast_mt = calculate_contrast(self.processed_image_mt)
            self.processed_photo_mt = ImageTk.PhotoImage(self.processed_image_mt)
            self.processed_label_mt.config(image=self.processed_photo_mt)
            self.processed_label_mt.image = self.processed_photo_mt
            self.table.insert("", "end", values=("Multithreaded Brightness", f"{elapsed_time_mt:.5f}", f"{avg_time_mt:.5f}", f"{contrast_mt:.5f}"))
            self.update_histogram(self.processed_image_mt, self.histogram_canvas_mt)

        # Update histogram
        self.update_histogram()

    def adjust_contrast(self, value):
        value = int(value)
        self.table.delete(*self.table.get_children())

        if self.seq_enabled:
            # Sequential contrast adjustment
            start_time_seq = time.perf_counter()
            self.processed_image_seq = adjust_contrast_sequential(self.original_image, value)
            end_time_seq = time.perf_counter()
            elapsed_time_seq = end_time_seq - start_time_seq
            self.times_seq.append(elapsed_time_seq)
            avg_time_seq = sum(self.times_seq) / len(self.times_seq)
            contrast_seq = calculate_contrast(self.processed_image_seq)
            self.processed_photo_seq = ImageTk.PhotoImage(self.processed_image_seq)
            self.processed_label_seq.config(image=self.processed_photo_seq)
            self.processed_label_seq.image = self.processed_photo_seq
            self.table.insert("", "end", values=("Sequential Contrast", f"{elapsed_time_seq:.5f}", f"{avg_time_seq:.5f}", f"{contrast_seq:.5f}"))
            self.update_histogram(self.processed_image_seq, self.histogram_canvas_seq)

        if self.simd_enabled:
            # SIMD contrast adjustment
            start_time_simd = time.perf_counter()
            self.processed_image_simd = adjust_contrast_simd(self.original_image, value)
            end_time_simd = time.perf_counter()
            elapsed_time_simd = end_time_simd - start_time_simd
            self.times_simd.append(elapsed_time_simd)
            avg_time_simd = sum(self.times_simd) / len(self.times_simd)
            contrast_simd = calculate_contrast(self.processed_image_simd)
            self.processed_photo_simd = ImageTk.PhotoImage(self.processed_image_simd)
            self.processed_label_simd.config(image=self.processed_photo_simd)
            self.processed_label_simd.image = self.processed_photo_simd
            self.table.insert("", "end", values=("SIMD Contrast", f"{elapsed_time_simd:.5f}", f"{avg_time_simd:.5f}", f"{contrast_simd:.5f}"))
            self.update_histogram(self.processed_image_simd, self.histogram_canvas_simd)

        if self.mt_enabled:
            # Multithreaded contrast adjustment
            start_time_mt = time.perf_counter()
            self.processed_image_mt = adjust_contrast_multithreading(self.original_image, value)
            end_time_mt = time.perf_counter()
            elapsed_time_mt = end_time_mt - start_time_mt
            self.times_mt.append(elapsed_time_mt)
            avg_time_mt = sum(self.times_mt) / len(self.times_mt)
            contrast_mt = calculate_contrast(self.processed_image_mt)
            self.processed_photo_mt = ImageTk.PhotoImage(self.processed_image_mt)
            self.processed_label_mt.config(image=self.processed_photo_mt)
            self.processed_label_mt.image = self.processed_photo_mt
            self.table.insert("", "end", values=("Multithreaded Contrast", f"{elapsed_time_mt:.5f}", f"{avg_time_mt:.5f}", f"{contrast_mt:.5f}"))
            self.update_histogram(self.processed_image_mt, self.histogram_canvas_mt)

    def adjust_vignette(self, value):
        value = int(value)
        self.table.delete(*self.table.get_children())

        if self.seq_enabled:
            # Sequential vignette adjustment
            start_time_seq = time.perf_counter()
            self.processed_image_seq = adjust_vignette_sequential(self.original_image, value)
            end_time_seq = time.perf_counter()
            elapsed_time_seq = end_time_seq - start_time_seq
            self.times_seq.append(elapsed_time_seq)
            avg_time_seq = sum(self.times_seq) / len(self.times_seq)
            contrast_seq = calculate_contrast(self.processed_image_seq)
            self.processed_photo_seq = ImageTk.PhotoImage(self.processed_image_seq)
            self.processed_label_seq.config(image=self.processed_photo_seq)
            self.processed_label_seq.image = self.processed_photo_seq
            self.table.insert("", "end", values=("Sequential Vignette", f"{elapsed_time_seq:.5f}", f"{avg_time_seq:.5f}", f"{contrast_seq:.5f}"))
            self.update_histogram(self.processed_image_seq, self.histogram_canvas_seq)

        if self.simd_enabled:
            # SIMD vignette adjustment
            start_time_simd = time.perf_counter()
            self.processed_image_simd = adjust_vignette_simd(self.original_image, value)
            end_time_simd = time.perf_counter()
            elapsed_time_simd = end_time_simd - start_time_simd
            self.times_simd.append(elapsed_time_simd)
            avg_time_simd = sum(self.times_simd) / len(self.times_simd)
            contrast_simd = calculate_contrast(self.processed_image_simd)
            self.processed_photo_simd = ImageTk.PhotoImage(self.processed_image_simd)
            self.processed_label_simd.config(image=self.processed_photo_simd)
            self.processed_label_simd.image = self.processed_photo_simd
            self.table.insert("", "end", values=("SIMD Vignette", f"{elapsed_time_simd:.5f}", f"{avg_time_simd:.5f}", f"{contrast_simd:.5f}"))
            self.update_histogram(self.processed_image_simd, self.histogram_canvas_simd)

        if self.mt_enabled:
            # Multithreaded vignette adjustment
            start_time_mt = time.perf_counter()
            self.processed_image_mt = adjust_vignette_multithreading(self.original_image, value)
            end_time_mt = time.perf_counter()
            elapsed_time_mt = end_time_mt - start_time_mt
            self.times_mt.append(elapsed_time_mt)
            avg_time_mt = sum(self.times_mt) / len(self.times_mt)
            contrast_mt = calculate_contrast(self.processed_image_mt)
            self.processed_photo_mt = ImageTk.PhotoImage(self.processed_image_mt)
            self.processed_label_mt.config(image=self.processed_photo_mt)
            self.processed_label_mt.image = self.processed_photo_mt
            self.table.insert("", "end", values=("Multithreaded Vignette", f"{elapsed_time_mt:.5f}", f"{avg_time_mt:.5f}", f"{contrast_mt:.5f}"))
            self.update_histogram(self.processed_image_mt, self.histogram_canvas_mt)

    def adjust_sharpness(self, value):
        value = int(value)
        self.table.delete(*self.table.get_children())

        if self.seq_enabled:
            # Sequential sharpness adjustment
            start_time_seq = time.perf_counter()
            self.processed_image_seq = adjust_sharpnes_sequential(self.original_image, value)
            end_time_seq = time.perf_counter()
            elapsed_time_seq = end_time_seq - start_time_seq
            self.times_seq.append(elapsed_time_seq)
            avg_time_seq = sum(self.times_seq) / len(self.times_seq)
            contrast_seq = calculate_contrast(self.processed_image_seq)
            self.processed_photo_seq = ImageTk.PhotoImage(self.processed_image_seq)
            self.processed_label_seq.config(image=self.processed_photo_seq)
            self.processed_label_seq.image = self.processed_photo_seq
            self.table.insert("", "end", values=("Sequential Sharpness", f"{elapsed_time_seq:.5f}", f"{avg_time_seq:.5f}", f"{contrast_seq:.5f}"))
            self.update_histogram(self.processed_image_seq, self.histogram_canvas_seq)

        if self.simd_enabled:
            # SIMD sharpness adjustment
            start_time_simd = time.perf_counter()
            self.processed_image_simd = adjust_sharpness_simd(self.original_image, value)
            end_time_simd = time.perf_counter()
            elapsed_time_simd = end_time_simd - start_time_simd
            self.times_simd.append(elapsed_time_simd)
            avg_time_simd = sum(self.times_simd) / len(self.times_simd)
            contrast_simd = calculate_contrast(self.processed_image_simd)
            self.processed_photo_simd = ImageTk.PhotoImage(self.processed_image_simd)
            self.processed_label_simd.config(image=self.processed_photo_simd)
            self.processed_label_simd.image = self.processed_photo_simd
            self.table.insert("", "end", values=("SIMD sharpness", f"{elapsed_time_simd:.5f}", f"{avg_time_simd:.5f}", f"{contrast_simd:.5f}"))
            self.update_histogram(self.processed_image_simd, self.histogram_canvas_simd)

        if self.mt_enabled:
            # Multithreaded sharpness adjustment
            start_time_mt = time.perf_counter()
            self.processed_image_mt = adjust_sharpness_multithreading(self.original_image, value)
            end_time_mt = time.perf_counter()
            elapsed_time_mt = end_time_mt - start_time_mt
            self.times_mt.append(elapsed_time_mt)
            avg_time_mt = sum(self.times_mt) / len(self.times_mt)
            contrast_mt = calculate_contrast(self.processed_image_mt)
            self.processed_photo_mt = ImageTk.PhotoImage(self.processed_image_mt)
            self.processed_label_mt.config(image=self.processed_photo_mt)
            self.processed_label_mt.image = self.processed_photo_mt
            self.table.insert("", "end", values=("Multithreaded sharpness", f"{elapsed_time_mt:.5f}", f"{avg_time_mt:.5f}", f"{contrast_mt:.5f}"))
            self.update_histogram(self.processed_image_mt, self.histogram_canvas_mt)


    def update_histogram(self, image, canvas):
        histogram = image.histogram()
        canvas.delete("all")  # Clear previous histogram
        for i in range(256):
            canvas.create_line(i, 100, i, 100 - histogram[i] / 100, fill="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = EditorApp(root)
    root.mainloop()
