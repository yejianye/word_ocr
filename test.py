import re
from pprint import pprint
from word_ocr import extract_highlighted_words_from_image
import imgutil

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import easyocr
import boto3

ocr_reader = easyocr.Reader(['en'])

def find_highlighted_regions(image_path_or_image, region_min_size=200, output_path=None):
    if isinstance(image_path_or_image, str):
        image = cv2.imread(image_path_or_image)
    else:
        image = image_path_or_image

    # Convert the image to HSV color space for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for orange highlights in HSV
    orange_lower = np.array([10, 80, 100], dtype=np.uint8)
    orange_upper = np.array([60, 255, 255], dtype=np.uint8)

    # Create masks for orange and yellow regions
    orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)

    # Find contours of the highlighted regions
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the highlighted regions
    regions = []
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw a rectangle around the highlighted area
        if w * h > region_min_size:
            regions.append((x, y, w, h))
            if output_path:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if output_path:
        cv2.imshow('Highlighted Regions', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(output_path, image)

    return regions

def intersect_area(x1, y1, w1, h1, x2, y2, w2, h2):
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_overlap * y_overlap

def sanitize_text(text):
    # Use regex to extract only alphabets and hyphens from the text
    return re.search(r'[a-zA-Z- ]+', text).group().strip()

def get_text_with_easyocr(image_path_or_image):
    if isinstance(image_path_or_image, str):
        image = cv2.imread(image_path_or_image)
    else:
        image = image_path_or_image
    data = ocr_reader.readtext(image, paragraph=False)
    result = []
    for i, (region, word, confidence) in enumerate(data):
        x = int(region[0][0])
        y = int(region[0][1])
        w = int(region[1][0] - x)
        h = int(region[2][1] - y)
        result.append({
            'text': word,
            'idx': i,
            'confidence': confidence,
            'bounding_box': (x, y, w, h)
        })
    return result

def get_text_with_tesseract(image_path_or_image):
    if isinstance(image_path_or_image, str):
        image = cv2.imread(image_path_or_image)
    else:
        image = image_path_or_image
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    result = []
    for i in range(len(data['text'])):
        result.append({
            'text': data['text'][i],
            'idx': i,
            'confidence': data['conf'][i] / 100,
            'bounding_box': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        })
    return result

def get_text_with_textract(image_path_or_image):
    if isinstance(image_path_or_image, str):
        image = cv2.imread(image_path_or_image)
    else:
        image = image_path_or_image

    height, width = image.shape[:2]
    _, image_data = cv2.imencode(".jpg", image_path_or_image)
    image_bytes = image_data.tobytes()

    textract = boto3.client('textract')
    response = textract.detect_document_text(
        Document={'Bytes': image_bytes}
    )
    # Process each detected block
    result = []
    idx = 0
    for block in response['Blocks']:
        if block['BlockType'] == 'WORD':  # Get bounding boxes for words only
            text = block['Text']
            bounding_box = block['Geometry']['BoundingBox']
            bounding_box = (
                int(bounding_box['Left'] * width),
                int(bounding_box['Top'] * height),
                int(bounding_box['Width'] * width),
                int(bounding_box['Height'] * height)
            )
            confidence = block['Confidence'] / 100
            result.append({
                'text': text,
                'idx': idx,
                'confidence': confidence,
                'bounding_box': bounding_box
            })
            idx += 1
    return result

def extract_highlighted_words_v2(image_path, area_threshold=200):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    hl_regions = find_highlighted_regions(image_path)
    # Use pytesseract to get the bounding box information
    data = get_text_with_textract(image)
    # Loop over each detected word and its corresponding bounding box
    hl_words = []
    for word in data:
        if word['confidence'] > 0.5:  # Filter out weakly confident text (optional)
            (x, y, w, h) = word['bounding_box']
            overlap_area = 0
            for region in hl_regions:
                rx, ry, rw, rh = region
                overlap_area += intersect_area(x, y, w, h, rx, ry, rw, rh)
            if overlap_area > area_threshold:
                word['text'] = sanitize_text(word['text'])
                hl_words.append(word)
                print(f"Detected words: {word['text']} Confidence: {word['confidence']}")
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the image with bounding boxes
    cv2.imshow('Highlighted Words', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # merge adjacent words
    merged_words = []
    for i in range(len(hl_words)):
        if i == 0:
            merged_words.append(hl_words[i]['text'])
        else:
            if hl_words[i]['idx'] - hl_words[i-1]['idx'] == 1:
                merged_words[-1] += ' ' + hl_words[i]['text']
            else:
                merged_words.append(hl_words[i]['text'])
    return merged_words

def test_find_highlighted_regions():
    i = 6
    imgutil.preprocess_image(f"tests/test{i}.jpg", f"tests/test{i}_processed.jpg")
    find_highlighted_regions(f"tests/test{i}_processed.jpg", output_path=f"tests/test{i}_highlighted.jpg")


def test_extract_highlighted_words_from_image(images=None):
    if images is None:
        images = [f"tests/test{i}.jpg" for i in range(1, 11)]
    results = []
    for image in images:
        processed_image_path = image.replace(".jpg", "_processed.jpg")
        imgutil.preprocess_image(image, processed_image_path)
        detected_words = extract_highlighted_words_v2(processed_image_path)
        baseline = [w.strip() for w in open(f"{image.split('.')[0]}.txt", "r").readlines() if w.strip()]
        correct_words = [w for w in detected_words if w in baseline]
        results.append({"image": image,
                        "detected_words": detected_words,
                        "correct_words": correct_words,
                        "baseline": baseline})
        print(f"=== Image {image} ===")
        print(f"Correct: {', '.join(correct_words)}")
        print(f"Wrong: {', '.join(w for w in detected_words if w not in baseline)}")
        print(f"Missed: {', '.join(w for w in baseline if w not in detected_words)}")
    print(f"=== Overall Stats ===")
    precision = sum(len(result["correct_words"]) for result in results) / sum(len(result["detected_words"]) for result in results)
    recall = sum(len(result["correct_words"]) for result in results) / sum(len(result["baseline"]) for result in results)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")

if __name__ == "__main__":
    test_extract_highlighted_words_from_image()
    # test_extract_highlighted_words_from_image(["tests/test5.jpg"])
    # test_find_highlighted_regions()
    # get_text_with_textract("tests/test6.jpg")