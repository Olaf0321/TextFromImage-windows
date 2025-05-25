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

# def extract_sp_pn(image_path):
#     img = preprocess_image(image_path)
#     text = pytesseract.image_to_string(img, lang='jpn')
#     # print('text', text)  # デバッグ用にOCR結果を表示

#     # 出品価格
#     price_match = re.search(r'出品価格\s*([0-9,]+)', text)
#     sell_price = price_match.group(1).replace(',', '') if price_match else ''

#     # 商品名（出品価格の直前の行を想定）
#     lines = text.split('\n')
#     product_name = ''
#     for i, line in enumerate(lines):
#         if '出品価格' in line and i > 0:
#             product_name = lines[i - 1].strip()
#             break

#     # # ASIN（例：B0xxxxxxxx）
#     # asin_match = re.search(r'\b(B0[A-Z0-9]{8})\b', text)
#     # asin = asin_match.group(1) if asin_match else ''

#     return sell_price, product_name

def extract_sell_prict(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    # 出品価格の領域を切り出し
    roi = img[int(h * 0.3):int(h * 0.5), int(w * 0):int(w * 1)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh, lang='jpn')
    print('ROI text (cost area):', text)

    # 正規表現で金額抽出（「出品価格」キーワードがなくても数値だけを拾う）
    match = re.search(r'出品価格\s*([0-9,]+)', text)
    return match.group(1).replace(',', '') if match else ''

def extract_cost_price_roi(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    roi = img[int(h * 0.5):int(h * 0.55), int(w * 0):int(w * 1)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh, lang='jpn')
    # print('ROI text (cost area):', text)

    # 正規表現で金額抽出（「仕入値」キーワードがなくても数値だけを拾う）
    match = re.search(r'仕入値\s*([0-9,]+)', text)
    return match.group(1).replace(',', '') if match else ''

def extract_product_name(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    # 商品名の領域を切り出し（ここでは出品価格の下の領域を想定）
    roi = img[int(h * 0.1):int(h * 0.15), int(w * 0.18):int(w * 1)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh, lang='jpn')
    # print('ROI text (cost area):', text)
    
    # 改行を除去して一行にまとめる
    single_line_text = ''.join(text.splitlines())
    print('一行に整形されたテキスト:', single_line_text)
    
    return single_line_text if single_line_text else ''

def extract_asin(image_path):
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img, lang='jpn')

    # ASIN（例：B0xxxxxxxx）
    asin_match = re.search(r'\b(B0[A-Z0-9]{8})\b', text)
    return asin_match.group(1) if asin_match else ''

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
        for i, img in enumerate(imgs):
            path = os.path.join(input_dir, img)
            if i == 0:
                # 1枚目の画像から情報を抽出
                sp = extract_sell_prict(path)
                if sp: sell_price = sp
                cp = extract_cost_price_roi(path)
                if cp: cost_price = cp
                pn = extract_product_name(path)
                if pn: product_name = pn
            elif i == 1:
                # 2枚目の画像から情報を抽出
                a = extract_asin(path)
                if a: asin = a
                
        data.append([key, sell_price, cost_price, asin, product_name])

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['画像番号', '出品価格', '仕入値', 'ASINコード', '商品名'])
        writer.writerows(data)

    print(f'✅ CSVファイルを出力しました: {output_csv}')

# 実行例
if __name__ == '__main__':
    input_folder = 'images'  # 画像フォルダ名
    process_images(input_folder)
