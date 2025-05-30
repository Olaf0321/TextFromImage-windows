import pytesseract
from PIL import Image
import cv2
import os
import re
import csv
import numpy as np
import datetime
import tempfile

IMAGE_SIZE = 1800
BINARY_THREHOLD = 180

# Tesseract のパス
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new

# def set_image_dpi(file_path):
#     im = Image.open(file_path)
#     length_x, width_y = im.size
#     factor = max(1, int(IMAGE_SIZE / length_x))
#     size = factor * length_x, factor * width_y

#     # Use new resampling attribute for Pillow 10+, fallback for older versions
#     try:
#         resample = Image.Resampling.LANCZOS
#     except AttributeError:
#         resample = Image.ANTIALIAS  # Pillow < 10

#     im_resized = im.resize(size, resample)

#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
#     temp_filename = temp_file.name
#     im_resized.save(temp_filename, dpi=(300, 300))

#     return temp_filename


def set_image_dpi(file_path):
    im = Image.open(file_path)
    
    # Convert to RGB if image has an alpha channel
    if im.mode == 'RGBA':
        im = im.convert('RGB')

    length_x, width_y = im.size
    factor = max(1, int(1800 / length_x))
    size = factor * length_x, factor * width_y
    
    im_resized = im.resize(size, Image.Resampling.LANCZOS)  # Updated resizing method
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


def extract_sell_price_from_text(text):
    sp_match = re.search(r'出品価格\s*([0-9,]+)', text)
    return sp_match.group(1).replace(',', '') if sp_match else ''

def extract_cost_price_from_text(text):
    # cp_match = re.search(r'値\s*([0-9,]+)', text)
    # return cp_match.group(1).replace(',', '') if cp_match else ''
    numbers = re.findall(r'\d+', text)
    result = ''.join(numbers)
    return result if result else ''

def extract_asin_from_text_B0(text):
    asin_match = re.search(r'\b(B0[A-Z0-9]{8})\b', text)
    return asin_match.group(1) if asin_match else ''

def extract_asin_from_text_BO(text):
    asin_match = re.search(r'\b(BO[A-Z0-9]{8})\b', text)
    return asin_match.group(1) if asin_match else ''

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

def extract_sp_cp(img):
    sell_price = cost_price = ''
    for i in np.arange(0.3, 0.8, 0.05):
        cropped = crop_by_ratio(img, i, i + 0.1, 0.01, 0.5)
        text = pytesseract.image_to_string(cropped, lang='jpn')
        # print(f'ROI text (y={i:.2f}):', text)
        if '出品価格' in text:
            sell_price = extract_sell_price_from_text(text)
        if '値' in text:
            cost_price = extract_cost_price_from_text(text)
            break
    return sell_price, cost_price

def extract_asin(image_path):
    asin = ''
    text = pytesseract.image_to_string(image_path, lang='jpn', config='--psm 6')

    # print('Extracted text for ASIN:', text)
    if 'B0' in text:
        asin = extract_asin_from_text_B0(text)
    # if not asin:
    #     asin = extract_asin_from_text_BO(text)
    #     asin = asin[0] + '0' + asin[2:]
    return asin

def extract_memo(image_path):
    memo = ''
    text = pytesseract.image_to_string(image_path, lang='jpn')
    pattern = r"出品コンディション(.*?)SKU\s*再生成"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        
        # 不要ワードを除去
        content = content.replace("クリア", "").replace("既定取得", "")
        
        # 前後の空白や改行を削除して整形
        content = re.sub(r'[\s\u3000]+', '', content)
        memo = content
    else:
        print("該当する範囲が見つかりませんでした。")
    return memo

def process_images(input_dir):
    files = sorted(os.listdir(input_dir))
    data = []
    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    output_csv = f'result{now}.csv'

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
            pre_img = process_image_for_ocr(path)
            if i == 0:
                # cv2.imwrite('path.png', pre_img)
                # 1枚目の画像から情報を抽出
                sp, cp = extract_sp_cp(pre_img)
                if sp: sell_price = sp
                if cp: cost_price = cp
            elif i == 1:
                # 2枚目の画像から情報を抽出
                a = extract_asin(pre_img)
                if a: asin = a
                mo = extract_memo(pre_img)
                if mo: memo = mo
        print(f'Processing {key}: Sell Price: {sell_price}, Cost Price: {cost_price}, ASIN: {asin}, Memo: {memo}')
        data.append([key, sell_price, cost_price, asin, memo])
    
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['番号', '出品価格', '仕入値', 'ASINコード', 'メモ'])
        writer.writerows(data)

    print(f'CSVファイルを出力しました: {output_csv}')

# 実行例
if __name__ == '__main__':
    input_folder = 'images'  # 画像フォルダ名
    process_images(input_folder)
