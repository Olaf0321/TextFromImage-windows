import pytesseract
from PIL import Image
import cv2
import os
import re
import csv

# Tesseract のパス（Windows用に明示）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_info(image_path):
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img, lang='jpn')
    # print('text', text)  # デバッグ用にOCR結果を表示

    # 出品価格
    price_match = re.search(r'出品価格\s*([0-9,]+)', text)
    sell_price = price_match.group(1).replace(',', '') if price_match else ''

    # 仕入値
    cost_match = re.search(r'仕入値?\s*([0-9,]+)', text)
    cost_price = cost_match.group(1).replace(',', '') if cost_match else ''

    # 商品名（出品価格の直前の行を想定）
    lines = text.split('\n')
    product_name = ''
    for i, line in enumerate(lines):
        if '出品価格' in line and i > 0:
            product_name = lines[i - 1].strip()
            break

    # ASIN（例：B0xxxxxxxx）
    asin_match = re.search(r'\b(B0[A-Z0-9]{8})\b', text)
    asin = asin_match.group(1) if asin_match else ''

    return sell_price, cost_price, asin, product_name

def process_images(input_dir, output_csv='result.csv'):
    files = sorted(os.listdir(input_dir))
    data = []

    # 2枚1セットをまとめて処理
    grouped = {}
    for file in files:
        if file.lower().endswith(('.jpg', '.png')):
            key = file.split('_')[0]
            grouped.setdefault(key, []).append(file)

    for key, imgs in grouped.items():
        sell_price = cost_price = asin = product_name = ''
        for img in imgs:
            path = os.path.join(input_dir, img)
            sp, cp, a, pn = extract_info(path)
            if sp: sell_price = sp
            if cp: cost_price = cp
            if a: asin = a
            if pn: product_name = pn
        data.append([sell_price, cost_price, asin, product_name])

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['出品価格', '仕入値', 'ASINコード', '商品名'])
        writer.writerows(data)

    print(f'✅ CSVファイルを出力しました: {output_csv}')

# 実行例
if __name__ == '__main__':
    input_folder = 'images'  # 画像フォルダ名
    process_images(input_folder)
