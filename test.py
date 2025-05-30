import pytesseract
from PIL import Image
import cv2
import os
import re
import csv
import numpy as np
import datetime

# Tesseract のパス
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def crop_by_ratio(image, y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio):
    height, width = image.shape[:2]

    y_start = int(height * y_start_ratio)
    y_end   = int(height * y_end_ratio)
    x_start = int(width * x_start_ratio)
    x_end   = int(width * x_end_ratio)

    cropped = image[y_start:y_end, x_start:x_end]
    return cropped

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_sell_price_from_text(text):
    sp_match = re.search(r'出品価格\s*([0-9,]+)', text)
    return sp_match.group(1).replace(',', '') if sp_match else ''

def extract_cost_price_from_text(text):
    cp_match = re.search(r'仕入値\s*([0-9,]+)', text)
    return cp_match.group(1).replace(',', '') if cp_match else ''

def extract_asin_from_text_B0(text):
    asin_match = re.search(r'\b(B0[A-Z0-9]{8})\b', text)
    return asin_match.group(1) if asin_match else ''

def extract_asin_from_text_BO(text):
    asin_match = re.search(r'\b(BO[A-Z0-9]{8})\b', text)
    return asin_match.group(1) if asin_match else ''

def extract_sp_cp(image_path):
    img = cv2.imread(image_path)
    sell_price = cost_price = ''
    for i in np.arange(0.3, 0.8, 0.05):
        cropped = crop_by_ratio(img, i, i + 0.1, 0.01, 0.5)
        text = pytesseract.image_to_string(cropped, lang='jpn')
        # print(f'ROI text (y={i:.2f}):', text)
        if '出品価格' in text:
            sell_price = extract_sell_price_from_text(text)
        if '仕入値' in text:
            cost_price = extract_cost_price_from_text(text)
            break
    return sell_price, cost_price

def extract_asin(image_path):
    img = cv2.imread(image_path)
    asin = ''
    cropped = crop_by_ratio(img, 0.4, 0.85, 0.01, 0.5)
    # text = pytesseract.image_to_string(cropped, lang='jpn')
    text = pytesseract.image_to_string(cropped, lang='jpn+eng')

    # print('Extracted text for ASIN:', text)
    if 'B0' in text:
        asin = extract_asin_from_text_B0(text)
    if not asin:
        asin = extract_asin_from_text_BO(text)
        # asin = asin[0] + '0' + asin[2:]
    return asin

def extract_memo(image_path):
    img = cv2.imread(image_path)
    memo = ''
    text = pytesseract.image_to_string(img, lang='jpn')
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
            if i == 0:
                # 1枚目の画像から情報を抽出
                sp, cp = extract_sp_cp(path)
                if sp: sell_price = sp
                if cp: cost_price = cp
            elif i == 1:
                # 2枚目の画像から情報を抽出
                a = extract_asin(path)
                if a: asin = a
                mo = extract_memo(path)
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
