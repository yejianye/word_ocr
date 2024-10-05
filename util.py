import hashlib
import os
import io
import base64
from joblib import Memory
from openai import OpenAI

DEFAULT_MODEL = 'gpt-4o-2024-08-06'
mem = Memory(location=os.getenv('JOBLIB_CACHE_DIR'), verbose=0)
cache = mem.cache

@cache
def llm_completion(prompt, model=DEFAULT_MODEL):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

@cache
def llm_image_completion(image_file_or_path, prompt, model=DEFAULT_MODEL):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    if isinstance(image_file_or_path, str): 
        with open(image_file_or_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        encoded_string = base64.b64encode(image_file_or_path.read()).decode('utf-8')

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", 
                   "content": "You are a helpful assistant that extract text from English textbook images."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}, 
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}]}]
    )
    return resp.choices[0].message.content

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



