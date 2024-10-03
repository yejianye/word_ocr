import re
import os
import base64
import hashlib
import io
from pprint import pprint
from docx import Document
from docx.shared import RGBColor, Pt
from openai import OpenAI
from joblib import Memory

import imgutil

DEFAULT_MODEL = 'gpt-4o-2024-08-06'

mem = Memory(location=os.getenv('JOBLIB_CACHE_DIR'), verbose=0)
@mem.cache
def llm_completion(prompt):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

@mem.cache
def llm_image_completion(image_file_or_path, prompt):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    if isinstance(image_file_or_path, str): 
        with open(image_file_or_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        encoded_string = base64.b64encode(image_file_or_path.read()).decode('utf-8')

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
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

def extract_highlighted_words_from_image(image_path_or_file):
    prompt = """
# 任务描述
1. 从图片中提取所有的被荧光笔标记的文本块
2. 根据以下条件对在第1步找到的文本块进行筛选
   a. 去掉有下划线或是粗体，但未被荧光笔高亮的文本块
   b. 去掉文字本身不是黑色，但背景未被高亮的文本块
   c. 去掉包含单词数量超过3个的文本块
3. 对于第2步中筛选出来的文本块，按要求输出

# 输出要求
- 每个文本块输出为一行
- 文本块中字母的大小写与图中保持一致
- 文本块的输出顺序与图中保持一致，按从上到下，从左到右的顺序输出。对于双栏布局的图片，先输出左栏，再输出右栏。
- 每一行除了文本块以外，不包含其他任何内容，如序号、列表标记等
- 除上述要求外，不要输出任何其他内容，如解释、注释等

# 输出示例
apple
orange
shuttle bus

"""
    image_hash = generate_file_hash(image_path_or_file)
    processed_image = f"/tmp/img_{image_hash}.jpg"
    imgutil.preprocess_image(image_path_or_file, processed_image)
    result = llm_image_completion(processed_image, prompt)
    words = [w.strip() for w in result.strip().split('\n') if w.strip()]
    return words

def extract_text_from_image(image_path_or_file):
#     prompt = """
# # 任务描述
# 输出图片中所有印刷体的英文文本

# # 定义
# 高亮文本：
# - 背景色为橙色或是黄色的单词短语
# - 但不包含有下划线的文本

# # 输出要求
# - 输出文本的换行与图中保持一致。
# - 对于高亮文本进行加粗（左右两边各添加两个星号)。
# - 如果高亮文本中包含逗号或是句号，则将标点符号两边的单词短语分别进行加粗。
# - 对于其他文本，按原文输出，不添加任何标记。
# - 不要输出除此以外的任何其他内容，如解释、注释等
# - 在最终输出前，再次与原图进行比对，确保高亮文本的加粗符合要求，否则需要进行调整。例如：下划线的文本不应该加粗。
# """
    prompt = """
# Task Description
Output all printed English text in the picture.

# Definition
Highlighted text:
- Words or phrases with an orange or yellow background.
- Underlined or bold text in the picture is not considered as Highlighted text.

# Output Requirements
## Content
- The line breaks in the output text should be the same as in the picture.
- Make sure not missing any text from the picture, especially paragraphs containing highlighted text.
- If the text layout in the picture is 2-column, output text in the left column first, and then output text in the right column.

## Formatting
- For highlighted text, make it bold (adding two asterisks on each side).
- If the highlighted text contains a comma or period, separately bold the words or phrases around the punctuation.
- For other text, output it as it is without any formatting.

## Others
- Do not output any other content such as explanations, comments, etc.
- Before the final output, compare it with the original image again to ensure that the boldness of the highlighted text meets the requirements; 
  otherwise, adjustments need to be made. For example, text with underline should not be bold.
"""
    image_hash = generate_file_hash(image_path_or_file)
    processed_image = f"/tmp/img_{image_hash}.jpg"
    imgutil.preprocess_image(image_path_or_file, processed_image)
    result = llm_image_completion(processed_image, prompt)

    if len(result) < 100: # retry if the result is not valid
        result = llm_image_completion(processed_image, f"The request is for student learning and education purpose.\n {prompt}")
        if len(result) < 100: # retry failed
            image_name = image_path_or_file if isinstance(image_path_or_file, str) else image_path_or_file.name
            result = f"Unable to process image {image_path_or_file}"
    return result.strip("```markdown").strip("```")

def add_markdown_to_docx(doc, markdown_text):
    # Regex to find underlined text (in markdown, __text__)
    lines = markdown_text.split('\n')
    for line in lines:
        asterisked_pattern = re.compile(r'\*\*(.*?)\*\*')
        # Split the markdown text by underlined sections
        parts = asterisked_pattern.split(line)
        
        para = doc.add_paragraph()
        for i, part in enumerate(parts):
            run = para.add_run(part)
            if i % 2 == 1:
                run.font.highlight_color = 7  # 7 represents yellow highlight in python-docx

def images_to_doc(images, output_file, process_callback=None):
    doc = Document()
    for i, img in enumerate(images):
        if process_callback:
            process_callback(i+1, len(images))
        md_text = extract_text_from_image(img)
        add_markdown_to_docx(doc, md_text)
        print(f"--------------------------\n{md_text}")
        doc.add_page_break()
    doc.save(output_file)

def doc_to_markdown(doc):
    paragraphs = []
    for para in doc.paragraphs:
        text = []
        for run in para.runs:
            if run.font.highlight_color:
                text.append(f"**{run.text}**")
            else:
                text.append(run.text)
        paragraphs.append(' '.join(text).strip())
    return '\n\n'.join(paragraphs)

def create_vocabulary(article):
    prompt = f"""
    在 ARTICLE 中提取加粗的单词或是词组，并在 ARTICLE 的上下文语境中翻译。每个单词输出一行，格式为
    单词 | 音标 | 词性 | 中文翻译

    - 如果加粗的单词是动词，则将其转化为动词原形后再翻译输出。例如，加粗的单词为动词lingered，则转成linger。
    - 如果加粗的单词是名词，则将其转化为名词单数形式后再翻译输出。例如，加粗的单词为名词tables，则转成table。
    - 如果是一个词组，则音标与词性可以为空，但 | 不能省略。
    - 音标左右两边需要用 / 来包裹。
    - 如果一个单词在之前已出现过，则跳过该单词不再输出。
    - 单词默认采用小写；专有名词首字母大写。
    - 单词前不需要加序号。
    - 除了上述要求的输出格式以外，不要输出任何其他内容。
    - 在最终输出前，再次检查每个输出的单词或短语，确保符合上述要求，否则进行调整。例如检查是存在动词过去式、名词复数、单词音标或是词性缺失等不符合要求的情况。

    # EXAMPLE
    正确的输出：
    lurch | /lɜːrtʃ/ | v. | 突然倾斜

    错误的输出：
    stranded | /ˈstrændɪd/ | v. | 搁浅
    错误原因：动词没有输出为原形，而是输出了动词的过去式。

    # ARTICLE 
    ```markdown
    {article}
    ```
    """
    result = llm_completion(prompt).strip("```")
    result = [l.strip() for l in result.split('\n') if l.strip()]
    result = [[j.strip() for j in i.split('|')] for i in result]
    return result

def add_vocabulary_table(doc, vocabulary, title):
    table = doc.tables[0]

    for word in vocabulary:
        row = table.add_row()
        row.height = Pt(20)
        for j, cell in enumerate(row.cells):
            cell.text = word[j] if j < len(word) else ""

def gen_vocabulary_doc(input_file, output_file, title="Vocabulary"):
    doc = Document(input_file)
    md_text = doc_to_markdown(doc)
    vocabulary = create_vocabulary(md_text)

    output_doc = Document("template.docx")
    title_paragraph = output_doc.paragraphs[0]
    title_paragraph.text = title  # Set the text of the paragraph directly
    title_paragraph.runs[0].font.size = Pt(16)  # Set font size to 16 points
    add_vocabulary_table(output_doc, vocabulary, title)
    output_doc.save(output_file)

### TESTS ###
def test_convert_markdown_to_docx():
    test_markdown = """
Brenda Z. Guiberson wanted to be a jungle explorer when she was a child. Much of her childhood was spent swimming, watching birds and __salmon__, and searching for arrowheads near her home along the Columbia River in the state of Washington. After volunteering at her child’s school, Guiberson became interested in writing nature books for children. She says that she writes for the child in herself, the one who loves adventure, surprises, and learning new things—a jungle explorer in words.

SETTING A PURPOSE As you read, pay attention to how earthquakes affect people, animals, the land, and the ocean, and think about how people explain and deal with the impact of these damaging events.

Head for the Hills! It’s Earth Against Earth

-----------------------------------------

For centuries, a big __chunk__ of earth under the Indian Ocean known as the India plate has been __scraping__ against another chunk of earth, the Burma plate. At eight o’clock in the morning on December 26, 2004, this scraping reached a breaking point near the island of Sumatra in Indonesia. A 750-mile section of earth snapped and __popped__ up as a new __40-foot-high__ cliff. This created one of the biggest earthquakes ever, 9.2 to 9.3 on the Richter scale.1 At a hospital, oxygen tanks __tumbled__ and beds __lurched__. At a __mosque__, the dome crashed to the floor. On the street, athletes running a race fell

1 Richter scale (rĭk’tar): a scale ranging from 1 to 10 that expresses the amount of energy released by an earthquake; named after Charles Richter, an American seismologist.
"""
    doc = Document()
    add_markdown_to_docx(doc, test_markdown)
    doc.save("test_markdown.docx")

def test_extract_text_from_image():
    test_image_path = "test.jpg"
    print(extract_text_from_image(test_image_path))

def test_images_to_doc():
    images_to_doc(["/Users/ryan/Downloads/IMG_5292.jpg"], "test_markdown.docx")

def test_gen_vocabulary_doc():
    gen_vocabulary_doc("test_markdown.docx", "test_vocabulary.docx", "Test Vocabulary")

def test_extract_highlighted_words_from_image():
    print(extract_highlighted_words_from_image("tests/test1.jpg"))

if __name__ == "__main__":
    # test_extract_text_from_image()
    # test_images_to_doc()
    # test_convert_markdown_to_docx()
    # test_gen_vocabulary_doc()
    test_extract_highlighted_words_from_image()
