import re
import os
import base64
from pprint import pprint
from docx import Document
from docx.shared import RGBColor, Pt
from openai import OpenAI
from joblib import Memory

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
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt}, 
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}]}]
    )
    return resp.choices[0].message.content

def extract_text_from_image(image_path_or_file):
    prompt = """
任务描述：
抽取图片中所有印刷体的英文文本

输出要求：
- 对于背景色为橙色或是黄色的单词短语，进行加粗（左右两边各添加两个星号)。
- 如果加粗的单词短语中包含逗号或是句号，则将标点符号两边的单词短语分别进行加粗。
- 对于在图中有下划线，但是背景色不是橙色或黄色的单词短语，不要加粗。
- 对于其他文本，按原文输出，不添加任何标记。
- 不要输出除此以外的任何其他内容，如解释、注释等
- 在最终输出前，再次与原图进行比对，确保加粗的单词短语与背景色要求一致，否则需要进行调整。
"""
    result = llm_image_completion(image_path_or_file, prompt)
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

    - 如果是一个动词，单词一栏中输出动词原形。
    - 如果是一个名词，单词一栏中输出名词单数形式。
    - 如果是一个词组，则音标与词性可以为空，但 | 不能省略。
    - 音标左右两边需要用 / 来包裹。
    - 如果一个单词在之前已出现过，则跳过该单词不再输出。
    - 单词默认采用小写；专有名词首字母大写。
    - 单词前不需要加序号。
    - 除了上述要求的输出格式以外，不要输出任何其他内容。

    # ARTICLE 
    ```markdown
    {article}
    ```
    """
    result = llm_completion(prompt).strip("```")
    result = result.strip().split('\n')
    result = [[j.strip() for j in i.split('|')] for i in result]
    return result

def add_vocabulary_table(doc, vocabulary):
    table = doc.tables[0]
    for word in vocabulary:
        row = table.add_row()
        row.height = Pt(20)
        for j, cell in enumerate(row.cells):
            cell.text = word[j] if j < len(word) else ""

def gen_vocabulary_doc(input_file, output_file):
    doc = Document(input_file)
    md_text = doc_to_markdown(doc)
    vocabulary = create_vocabulary(md_text)
    pprint(vocabulary)

    output_doc = Document("template.docx")
    add_vocabulary_table(output_doc, vocabulary)
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
    images_to_doc(["test2.jpg","test3.jpg"], "test_markdown.docx")

def test_gen_vocabulary_doc():
    gen_vocabulary_doc("test_markdown.docx", "test_vocabulary.docx")


if __name__ == "__main__":
    # test_extract_text_from_image()
    # test_images_to_doc()
    # test_convert_markdown_to_docx()
    test_gen_vocabulary_doc()
