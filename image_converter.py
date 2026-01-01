# =====================================================================================
# FILENAME: image_converter.py
#
# DESCRIPTION:
# A Python script to demonstrate the massive performance difference between
# traditional 'for-loop' processing and NumPy's 'Vectorization' for numerical
# operations. The script converts a color image to grayscale using both methods
# and benchmarks their execution time.
#
# AUTHOR: SeyyedSajjadFazeli
# GITHUB: https://github.com/SeyyedSajjadFazeli/Python-Vectorization-Showcase
# =====================================================================================

import cv2
import numpy as np
import time
import os

# --- METHOD 1: The Slow, Non-Idiomatic Approach (For-Loops) ---

def convert_to_grayscale_loop(image: np.ndarray) -> np.ndarray:
    """
    Converts a color image to grayscale using nested for-loops to iterate
    over every single pixel.

    This is the "naive" or "intuitive" approach but is extremely inefficient in
    Python due to the high overhead of the interpreter for each loop iteration.

    Args:
        image (np.ndarray): The input color image as a NumPy array with shape
                            (height, width, 3). The channels are expected to be
                            in BGR order, which is the default for OpenCV.

    Returns:
        np.ndarray: The resulting grayscale image as a 2D NumPy array.
    """
    # Get the dimensions of the image.
    height, width, _ = image.shape
    
    # Create an empty, black image with the same height and width.
    # This array will be filled with the calculated grayscale values.
    grayscale_image = np.zeros((height, width), dtype=np.uint8)
    
    # Iterate over each row (y-coordinate).
    for i in range(height):
        # Iterate over each column in the current row (x-coordinate).
        for j in range(width):
            # Extract the Blue, Green, and Red channel values for the current pixel.
            # OpenCV loads images in BGR order by default, not RGB.
            blue, green, red = image[i, j]
            
            # Apply the standard luminosity formula to calculate the grayscale value.
            # Formula: Y = 0.299*R + 0.587*G + 0.114*B
            # These coefficients account for the human eye's varying sensitivity to colors.
            gray_value = int(0.114 * blue + 0.587 * green + 0.299 * red)
            
            # Assign the calculated grayscale value to the corresponding pixel in the output image.
            grayscale_image[i, j] = gray_value
            
    return grayscale_image

# --- METHOD 2: The Fast, Idiomatic Python Approach (Vectorization) ---

def convert_to_grayscale_vectorized(image: np.ndarray) -> np.ndarray:
    """
    Converts a color image to grayscale using NumPy's vectorized operations.

    This approach avoids explicit Python loops and instead leverages NumPy's
    highly optimized, pre-compiled C code to perform the mathematical
    operations on the entire array at once. This is the standard and
    recommended way to perform such tasks.

    Args:
        image (np.ndarray): The input color image (BGR format).

    Returns:
        np.ndarray: The resulting grayscale image.
    """
    # Define the coefficients for the BGR channels for the luminosity formula.
    # Note the order matches OpenCV's BGR format: [Blue_Coeff, Green_Coeff, Red_Coeff]
    coefficients = np.array([0.114, 0.587, 0.299])
    
    # This is the core of vectorization.
    # We perform a dot product between the image's pixel values and the coefficients.
    # NumPy applies this operation across the entire image simultaneously, which is
    # thousands of times faster than iterating pixel by pixel.
    # image[..., :3] ensures we only use the B, G, R channels, ignoring a potential alpha channel.
    grayscale_image = np.dot(image[..., :3], coefficients)
    
    # The result of the dot product is a float array. We must convert it back to
    # an 8-bit unsigned integer, which is the standard format for image pixels (0-255).
    return grayscale_image.astype(np.uint8)

# --- MAIN EXECUTION BLOCK ---

def main():
    """
    The main entry point of the script. It loads an image, processes it
    using both the slow and fast methods, benchmarks them, and saves the results.
    """
    # Define the path to the input image.
    # Ensure you have an 'images' folder with a 'sample.jpg' inside it.
    image_path = os.path.join("images", "sample.jpg")

    # --- Input Validation ---
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path '{image_path}'.")
        print("Please make sure you have an 'images' folder with a 'sample.jpg' file inside.")
        return

    # Read the image from disk using OpenCV.
    color_image = cv2.imread(image_path)
    if color_image is None:
        print(f"Error: Failed to read the image from '{image_path}'. It might be corrupted.")
        return
        
    print(f"Processing a {color_image.shape} image...\n")

    # --- Benchmark the Slow Method (For-Loops) ---
    print("[Slow Method] Using For-Loops:")
    start_time_loop = time.perf_counter()
    grayscale_loop = convert_to_grayscale_loop(color_image)
    end_time_loop = time.perf_counter()
    time_loop = end_time_loop - start_time_loop
    print(f"Time taken: {time_loop:.4f} seconds.")
    cv2.imwrite("grayscale_loop.jpg", grayscale_loop)

    # --- Benchmark the Fast Method (Vectorization) ---
    print("\n[Fast Method] Using Vectorization:")
    start_time_vec = time.perf_counter()
    grayscale_vec = convert_to_grayscale_vectorized(color_image)
    end_time_vec = time.perf_counter()
    time_vec = end_time_vec - start_time_vec
    print(f"Time taken: {time_vec:.4f} seconds.")
    cv2.imwrite("grayscale_vectorized.jpg", grayscale_vec)

    # --- Final Comparison ---
    # Calculate the performance improvement factor.
    if time_vec > 0:
        speedup = time_loop / time_vec
        print(f"\n✨ Vectorization was ~{speedup:.1f} times faster!")

# This standard Python construct ensures that the `main()` function is called
# only when the script is executed directly (e.g., `python image_converter.py`),
# and not when it's imported as a module into another script.
if __name__ == "__main__":
    main()

