import pytesseract
from PIL import Image
import cv2
import os
import re
import csv
import numpy as np
import tempfile

IMAGE_SIZE = 1800
BINARY_THREHOLD = 180

# Tesseract のパス（Windows用に明示）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new

def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y

    # Use new resampling attribute for Pillow 10+, fallback for older versions
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.ANTIALIAS  # Pillow < 10

    im_resized = im.resize(size, resample)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))

    return temp_filename

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

def crop_by_ratio(image, y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio):
    height, width = image.shape[:2]
    # print(f'Image dimensions: {height}x{width}')
    # 各比率に基づいて座標を計算

    y_start = int(height * y_start_ratio)
    y_end   = int(height * y_end_ratio)
    x_start = int(width * x_start_ratio)
    x_end   = int(width * x_end_ratio)
    # print(f'Cropping coordinates: y({y_start}-{y_end}), x({x_start}-{x_end})')

    cropped = image[y_start:y_end, x_start:x_end]
    return cropped

# def preprocess_image(image_path):
#     # Step 1: Load image in grayscale
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Step 2: Resize (scale up for better OCR performance)
#     scale_percent = 200  # e.g., 200% size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

#     # Step 3: Apply thresholding (convert to black & white)
#     _, thresh = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)

#     return thresh

def extract_asin_from_text(text):
    asin_match = re.search(r'\b(BO[A-Z0-9]{8})\b', text)
    return asin_match.group(1) if asin_match else ''

def extract_asin(image_path):
    # Preprocess the image
    img = process_image_for_ocr(image_path)
    
    # crop = crop_by_ratio(img, 0.55, 0.8, 0.1, 0.9)  # Adjust ratios as needed

    # Run Tesseract
    text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    print('Extracted text for ASIN:', text)
    asin = extract_asin_from_text(text)
    return asin

def process_images(input_dir, output_csv='test.csv'):
    files = sorted(os.listdir(input_dir))
    data = []

    # 2枚1セットをまとめて処理
    grouped = {}
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            key = file.split('_')[0]
            grouped.setdefault(key, []).append(file)

    for key, imgs in grouped.items():
        sell_price = cost_price = asin = product_name = memo = ''
        for i, img in enumerate(imgs):
            path = os.path.join(input_dir, img)
            if i == 1:
                an = extract_asin(path)
                if an: asin = an
        print(f'Processing {key}: Asin: {asin}')
        data.append([key, asin])
    
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['番号', '出品価格', '仕入値', 'ASINコード', 'メモ'])
        writer.writerows(data)

    print(f'CSVファイルを出力しました: {output_csv}')

# 実行例
if __name__ == '__main__':
    input_folder = 'test_images'  # 画像フォルダ名
    process_images(input_folder)
