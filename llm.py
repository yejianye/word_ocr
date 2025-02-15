import os
import base64
import json
import tiktoken
from openai import OpenAI
from util import cache

_model_configs = {
    "doubao-1.5-pro": {
        "model": "ep-20250215120137-snt46",
        "api_key": os.environ.get("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "token_model": "gpt-4o",
    },
    "doubao-1.5-lite": {
        "model": "ep-20250215120330-2x75z",
        "api_key": os.environ.get("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "token_model": "gpt-4o",
    },
    "qwen-plus": {
        "model": "qwen-plus-latest",
        "api_key": os.environ.get("QWEN_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "token_model": "gpt-4o",
    },
    "qwen-turbo": {
        "model": "qwen-turbo-latest",
        "api_key": os.environ.get("QWEN_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "token_model": "gpt-4o",
    },
    "deepseek-chat": {
        "model": "deepseek-chat",
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/v1",
        "token_model": "gpt-4o",
    },
}

# DEFAULT_MODEL = 'gpt-4o-2024-08-06'
DEFAULT_MODEL = os.getenv('WORD_OCR_MODEL', 'doubao-1.5-pro')

@cache
def llm_completion(prompt: str, model: str = DEFAULT_MODEL):
    if model in _model_configs:
        config = _model_configs[model]
        model = config["model"]
        client = OpenAI(    
            api_key = config["api_key"],
            base_url = config["base_url"],
        )
    else:
        client = OpenAI()
    completion = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

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




def llm_response_to_json(response: str):
    if response.startswith("```json"):
        response = response.strip("```json\n").strip("\n```")
    return json.loads(response)

def llm_response_to_markdown(response: str):
    if response.startswith("```markdown"):
        response = response.strip("```markdown\n").strip("\n```")
    return response

def calculate_token(text, model):
    token_model = _model_configs.get(model, {}).get("token_model") or model
    encoding = tiktoken.encoding_for_model(token_model)
    tokens = encoding.encode(text)
    return len(tokens)

if __name__ == "__main__":
    # print(llm_completion("Hello, who are you!", "qwen-plus"))
    print(calculate_token("Hello, who are you!", "qwen-turbo"))
