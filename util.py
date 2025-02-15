import hashlib
import io
import os
from joblib import Memory

mem = Memory(location=os.getenv('JOBLIB_CACHE_DIR'), verbose=0)
cache = mem.cache

def generate_file_hash(file_or_bytesio):
    md5_hash = hashlib.md5()
    
    if isinstance(file_or_bytesio, str):
        # If it's a string, assume it's a file path
        with open(file_or_bytesio, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
    elif isinstance(file_or_bytesio, io.BytesIO):
        # If it's a BytesIO object, read its content
        file_or_bytesio.seek(0)
        for chunk in iter(lambda: file_or_bytesio.read(4096), b""):
            md5_hash.update(chunk)
        file_or_bytesio.seek(0)  # Reset the BytesIO position
    else:
        raise ValueError("Input must be either a file path string or a BytesIO object")
    
    return md5_hash.hexdigest()[:16]

def strip_format_quote(text):
    """
    Strip format quotes from a string, such as "```markdown ... ```" or "```plaintext ... ```"
    
    Args:
    text (str): The input string that may contain format quotes
    
    Returns:
    str: The input string with format quotes removed
    """
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Check if the text starts and ends with format quotes
    if text.startswith("```") and text.endswith("```"):
        # Find the position of the first newline
        first_newline = text.find("\n")
        if first_newline != -1:
            # Remove the opening format quote and everything before the first newline
            text = text[first_newline + 1:]
        else:
            # If there's no newline, just remove the opening format quote
            text = text[3:]
        
        # Remove the closing format quote
        text = text[:-3]
    
    # Remove any remaining leading or trailing whitespace
    return text.strip()



