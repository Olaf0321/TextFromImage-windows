import easyocr
import cv2
import re

# Load image
image_path = "images/123.png"
image = cv2.imread(image_path)

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

crop = crop_by_ratio(image, 0.6, 0.63, 0, 1)

# Initialize EasyOCR reader
reader = easyocr.Reader(['ja', 'en'], gpu=False)  # GPU optional

# Run OCR
results = reader.readtext(image)

print("OCR Results:")
for (bbox, text, prob) in results:
    print(f"Text: {text}, Probability: {prob:.2f}")

# Optional: draw detected boxes (for debugging)
for (bbox, text, prob) in results:
    (tl, tr, br, bl) = bbox
    tl = tuple([int(x) for x in tl])
    br = tuple([int(x) for x in br])
    cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save result image with bounding boxes (optional)
cv2.imwrite("output_with_boxes.png", image)

# Extract ASIN-like codes (10 uppercase alphanumeric characters)
asin_pattern = r'\b[A-Z0-9]{10}\b'
found_asins = []

print("Detected ASIN candidates:")
for (_, text, _) in results:
    match = re.findall(asin_pattern, text)
    if match:
        found_asins.extend(match)
        print("→", match)

# Final output
if found_asins:
    print("\n✅ Likely ASIN(s):", found_asins)
else:
    print("\n❌ No valid ASIN detected.")
