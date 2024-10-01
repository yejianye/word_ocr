import cv2
import pytesseract

import io
import numpy as np

def read_image(image_path_or_file):
    """
    Read an image from either a file path or a BytesIO object.
    
    Args:
    image_path_or_file (str or io.BytesIO): The image file path or BytesIO object.
    
    Returns:
    numpy.ndarray: The image as a NumPy array.
    """
    if isinstance(image_path_or_file, str):
        # If it's a string, assume it's a file path
        return cv2.imread(image_path_or_file)
    elif isinstance(image_path_or_file, io.BytesIO):
        # If it's a BytesIO object, convert to numpy array and decode
        file_bytes = np.asarray(bytearray(image_path_or_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError("Input must be either a file path string or a BytesIO object")
    
def write_image(image, output_path_or_file):
    if isinstance(output_path_or_file, str):
        # If it's a string, assume it's a file path
        cv2.imwrite(output_path_or_file, image)
    elif isinstance(output_path_or_file, io.BytesIO):
        # If it's a BytesIO object, convert to numpy array and encode
        _, encoded_image = cv2.imencode('.jpeg', image)
        output_path_or_file.write(encoded_image.tobytes())
    else:
        raise ValueError("Output must be either a file path string or a BytesIO object")

def resize_image(image, max_size=2000):
    """
    Resize the image so that the long side is at most max_size pixels.
    
    Args:
    image (numpy.ndarray): The input image.
    max_size (int): The maximum size for the long side of the image.
    
    Returns:
    numpy.ndarray: The resized image.
    """
    height, width = image.shape[:2]
    if max(height, width) <= max_size:
        return image
    
    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def fix_image_orientation(image):    
    # Run Tesseract to detect text orientation
    osd = pytesseract.image_to_osd(image)
    
    # Extract the rotation angle from Tesseract's OSD (Orientation and Script Detection)
    rotation_angle = int(osd.split("\n")[2].split(":")[1].strip())

    # If the angle is 90, 180, or 270 degrees, rotate the image accordingly
    if rotation_angle == 90:
        corrected_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        corrected_image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == 270:
        corrected_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # The image is already correctly oriented
        corrected_image = image
    return corrected_image

def preprocess_image(image_path_or_file, output_path_or_file):
    image = read_image(image_path_or_file)
    image = resize_image(image)
    image = fix_image_orientation(image)
    write_image(image, output_path_or_file)

def test_preprocess_image():
    image_path = "/Users/ryan/Downloads/IMG_5293.jpg"
    output_path = "/Users/ryan/Downloads/processed_image.jpg"
    preprocess_image(image_path, output_path)

if __name__ == "__main__":
    test_preprocess_image()