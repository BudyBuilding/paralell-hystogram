# paralell-hystogram
This program allows users to adjust the brightness and contrast of an image using three different processing techniques: sequential, SIMD (Single Instruction, Multiple Data), and multithreading. It also measures and displays the time taken for each method to complete both brightness and contrast adjustments.

Key features include:

Image Manipulation: Adjust brightness and contrast interactively using sliders.
Performance Comparison: It tracks the time for each processing method (sequential, SIMD, multithreading) to adjust brightness and contrast, and stores the results globally.
Visualization: The program displays the original and processed images for each method side by side and updates dynamically when sliders are changed.
Global Time Storage: The performance results are stored globally and displayed separately from the image update process, allowing independent analysis of timing results.
The GUI is built using tkinter, and image processing is handled via PIL (Python Imaging Library) and NumPy.
