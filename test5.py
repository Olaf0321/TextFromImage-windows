import cv2
import numpy as np
from PIL import Image
import tempfile
import pytesseract
import os
import datetime
import csv
import re

def normalize_image(img):
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    # im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized = im.resize(size, Image.Resampling.LANCZOS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]

def preprocess_image(file_path):
    temp_filename = set_image_dpi(file_path)
    image = cv2.imread(temp_filename)
    image = normalize_image(image)
    image = remove_noise(image)
    image = get_grayscale(image)
    image = thresholding(image)
    return image

def crop_by_ratio(image, y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio):
    height, width = image.shape[:2]

    y_start = int(height * y_start_ratio)
    y_end   = int(height * y_end_ratio)
    x_start = int(width * x_start_ratio)
    x_end   = int(width * x_end_ratio)

    cropped = image[y_start:y_end, x_start:x_end]
    return cropped

def extract_sell_price_from_img(img, key):
    sell_price = ''
    crop = crop_by_ratio(img, 0.2, 0.5, 0, 0.5)
    text = pytesseract.image_to_string(crop, lang='jpn', config='--psm 6')
    
    lines = text.splitlines()
    matching_lines = [line for line in lines if "品価" in line]
    
    for line in matching_lines:
        words = line.split()
        numeric_prefix_words = [word for word in words if len(word) >= 2 and word[0].isdigit() and word[1].isdigit()]
        if (len(numeric_prefix_words) == 0):
            continue
        sell_price = re.sub(r'[^a-z0-9\s]', '', numeric_prefix_words[0])
        len1 = len(sell_price)
        if (sell_price[len1-1] == 'd'):
            sell_price = sell_price[:len1-1] + '0'
    
    return sell_price

def extract_cost_price_from_img(img, key):
    cost_price = ''
    crop = crop_by_ratio(img, 0.3, 0.8, 0, 0.5)
    text = pytesseract.image_to_string(crop, lang='jpn', config='--psm 6')
    
    lines = text.splitlines()
    matching_lines = [line for line in lines if "値" in line]
    
    for line in matching_lines:
        words = line.split()
        numeric_prefix_words = [word for word in words if len(word) >= 2 and word[0].isdigit() and word[1].isdigit()]
        if (len(numeric_prefix_words) == 0):
            continue
        cost_price = re.sub(r'[^a-z0-9\s]', '', numeric_prefix_words[0])
        len1 = len(cost_price)
        if (cost_price[len1-1] == 'd'):
            cost_price = cost_price[:len1-1] + '0'
    
    return cost_price

def extract_asin_from_text_B0(text):
    asin_matches = re.findall(r'\bB0[A-Z0-9]*\b', text)
    for i in range(len(asin_matches)):
        if len(asin_matches[i]) < 5:
            continue
        return asin_matches[i]
    return ''

def extract_asin_from_text_BO(text):
    asin_matches = re.findall(r'\bBO[A-Z0-9]*\b', text)
    for i in range(len(asin_matches)):
        if len(asin_matches[i]) < 5:
            continue
        return asin_matches[i]
    return ''

def extract_asin(img):
    asin = ''
    crop = crop_by_ratio(img, 0.3, 0.8, 0, 0.3)
    text = pytesseract.image_to_string(crop, lang='jpn', config='--psm 6')

    if 'B0' in text:
        asin = extract_asin_from_text_B0(text)
    if 'BO' in text and not asin:
        asin = extract_asin_from_text_BO(text)
        if (asin != ''):
            asin = asin[0] + '0' + asin[2:]
            
    if (len(asin) == 11):
        asin = 'B0' + asin[3:]
    return asin

def extract_section_single_line(text, start_keyword, end_keyword, remove_phrase, exclude_prefixes):
    lines = text.splitlines()
    start_index = end_index = -1

    for i, line in enumerate(lines):
        if start_keyword in line and start_index == -1:
            start_index = i
        if end_keyword in line and start_index != -1:
            end_index = i
            break
    
    if start_index != -1 and end_index != -1:
        section = lines[start_index + 1:end_index]
        filtered_lines = []
        for line in section:
            stripped = line.strip()
            if not any(stripped.startswith(prefix) for prefix in exclude_prefixes):
                filtered_lines.append(stripped)
        one_line = ' '.join(filtered_lines)

        # Step 3: Remove specific phrase
        cleaned = one_line.replace(remove_phrase, '')
        return cleaned
    else:
        return 'Section not found.'

def extract_memo(img, key):
    crop = crop_by_ratio(img, 0, 1, 0, 0.85)
    text = pytesseract.image_to_string(crop, lang='jpn')
    result = extract_section_single_line(text, '平均データ未取得', 'SKU', '出品コンディション', exclude_prefixes=['新品', '中古', '新F'])
    return result

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
        sell_price = cost_price = asin = memo = ''
        for i, img in enumerate(imgs):
            path = os.path.join(input_dir, img)
            pre_img = preprocess_image(path)
            if i == 0:
                # 1枚目の画像から情報を抽出
                sp = extract_sell_price_from_img(pre_img, key)
                if sp: sell_price = sp
                cp = extract_cost_price_from_img(pre_img, key)
                if cp: cost_price = cp
            if i == 1:
                # 2枚目の画像から情報を抽出
                a = extract_asin(pre_img)
                if a: asin = a
                mo = extract_memo(pre_img, key)
                if mo: memo = mo
        print(f'Processing {key}: Sell Price: {sell_price}, Cost Price: {cost_price}, ASIN: {asin}, Memo: {memo}')
        data.append([key, sell_price, cost_price, asin, memo])
    
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['番号', '出品価格', '仕入値', 'ASINコード', 'メモ'])
        writer.writerows(data)

    print(f'CSVファイルを出力しました: {output_csv}')

if __name__ == '__main__':
    input_folder = 'images'  # 画像フォルダ名
    process_images(input_folder)