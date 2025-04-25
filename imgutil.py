import cv2

import io
import numpy as np
from PIL import Image, ImageOps


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
    image = Image.open(image)# Correct orientation based on EXIF metadata
    image = ImageOps.exif_transpose(image)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')  # or 'PNG' depending on your image format
    img_bytes.seek(0)  # Important: reset the cursor to the start
    return img_bytes

def preprocess_image(image_path_or_file, output_path_or_file):
    image = read_image(image_path_or_file)
    image = resize_image(image)
    write_image(image, output_path_or_file)

def test_preprocess_image():
    image_path = "/Users/ryan/Downloads/IMG_5293.jpg"
    output_path = "/Users/ryan/Downloads/processed_image.jpg"
    preprocess_image(image_path, output_path)

def test_image_orientation():
    image_path= "/Users/ryan/Downloads/IMG_5736.jpg"
    image = read_image(image_path)
    image = resize_image(image)
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")

if __name__ == "__main__":
    # test_preprocess_image()
    test_image_orientation()