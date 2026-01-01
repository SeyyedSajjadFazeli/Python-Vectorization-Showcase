import cv2
import numpy as np
import time
import os

# --- تابع کند: پردازش پیکسل به پیکسل با حلقه ---
def convert_to_grayscale_loop(image: np.ndarray) -> np.ndarray:
    """
    تصویر رنگی را با استفاده از حلقه‌های for تودرتو به مقیاس خاکستری تبدیل می‌کند.
    این روش بسیار کند و غیربهینه است.
    """
    height, width, _ = image.shape
    grayscale_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # استخراج کانال‌های رنگی BGR (استاندارد OpenCV)
            blue, green, red = image[i, j]
            
            # محاسبه مقدار خاکستری با فرمول Luminosity
            # (Y = 0.299*R + 0.587*G + 0.114*B)
            gray_value = int(0.114 * blue + 0.587 * green + 0.299 * red)
            grayscale_image[i, j] = gray_value
            
    return grayscale_image

# --- تابع سریع: پردازش با NumPy Vectorization ---
def convert_to_grayscale_vectorized(image: np.ndarray) -> np.ndarray:
    """
    تصویر رنگی را با استفاده از عملیات ماتریسی NumPy (Vectorization) تبدیل می‌کند.
    این روش فوق‌العاده سریع و بهینه است.
    """
    # ضرایب برای کانال‌های BGR
    coefficients = np.array([0.114, 0.587, 0.299])
    
    # با یک ضرب ماتریسی، تمام پیکسل‌ها به صورت یکجا محاسبه می‌شوند
    grayscale_image = np.dot(image[..., :3], coefficients).astype(np.uint8)
    
    return grayscale_image

def main():
    """
    نقطه شروع برنامه: تصویر را بارگذاری، پردازش و نتایج را مقایسه می‌کند.
    """
    # مسیر تصویر ورودی
    image_path = os.path.join("images", "sample.jpg") # نام تصویر خود را اینجا وارد کنید

    if not os.path.exists(image_path):
        print(f"خطا: تصویر در مسیر '{image_path}' یافت نشد.")
        print("لطفاً یک تصویر در پوشه 'images' قرار دهید.")
        return

    # خواندن تصویر با OpenCV
    color_image = cv2.imread(image_path)
    if color_image is None:
        print(f"خطا: امکان خواندن تصویر از مسیر '{image_path}' وجود ندارد.")
        return
        
    print(f"Processing a {color_image.shape} image...\n")

    # --- تست روش کند ---
    print("[روش کند] با حلقه For:")
    start_time_loop = time.perf_counter()
    grayscale_loop = convert_to_grayscale_loop(color_image)
    end_time_loop = time.perf_counter()
    time_loop = end_time_loop - start_time_loop
    print(f"Time taken: {time_loop:.4f} seconds.")
    cv2.imwrite("grayscale_loop.jpg", grayscale_loop)

    # --- تست روش سریع ---
    print("\n[روش سریع] با Vectorization:")
    start_time_vec = time.perf_counter()
    grayscale_vec = convert_to_grayscale_vectorized(color_image)
    end_time_vec = time.perf_counter()
    time_vec = end_time_vec - start_time_vec
    print(f"Time taken: {time_vec:.4f} seconds.")
    cv2.imwrite("grayscale_vectorized.jpg", grayscale_vec)

    # --- مقایسه نتایج ---
    if time_vec > 0:
        speedup = time_loop / time_vec
        print(f"\n✨ Vectorization ~{speedup:.1f} برابر سریع‌تر بود!")

if __name__ == "__main__":
    main()
