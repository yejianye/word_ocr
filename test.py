from pprint import pprint
from word_ocr import extract_highlighted_words_from_image

def test_extract_highlighted_words_from_image(images=None):
    if images is None:
        images = [f"tests/test{i}.jpg" for i in range(1, 11)]
    results = []
    for image in images:
        detected_words = extract_highlighted_words_from_image(image)
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