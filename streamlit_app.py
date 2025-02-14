import io
import threading
import time
from queue import Queue
from datetime import datetime

import pandas as pd
import streamlit as st

import word_ocr

class AppState:
    def __init__(self):
        self.stage = "upload_file"
        self.flow = None

class ImageAppState:
    def __init__(self):
        self.stage = "upload_file"
        self.uploaded_images = []
        self.image_idx = 0
        self.highlighted_words = {}
        self.job_queue_ocr = Queue()
        self.ocr_results = {}
        self.job_queue_vocabulary = Queue()
        self.vocabulary_results = {}

        self.ocr_thread = threading.Thread(target=self.process_ocr_queue, daemon=True)
        self.vocabulary_thread = threading.Thread(target=self.process_vocabulary_queue, daemon=True)

    def upload_images(self, uploaded_images):
        self.uploaded_images = uploaded_images
        [self.job_queue_ocr.put((idx, image)) for idx, image in enumerate(uploaded_images)]
        self.job_queue_ocr.put((None, None))

        self.ocr_thread.start()
        self.stage = 'verify_ocr_results'

    def process_ocr_queue(self):
        while True:
            idx, image = self.job_queue_ocr.get()
            if image is None:
                break
            print(f"====Processing OCR queue for image {idx}====")
            result = word_ocr.extract_highlighted_words_from_image(image)
            self.ocr_results[idx] = result
            self.job_queue_ocr.task_done()

    def process_vocabulary_queue(self):
        while True:
            idx, (words, article) = self.job_queue_vocabulary.get()
            if idx is None:
                break
            print(f"====Processing Vocabulary queue for image {idx}====")
            result = word_ocr.create_vocabulary(words, article)
            self.vocabulary_results[idx] = result
            self.job_queue_vocabulary.task_done()

    def get_current_ocr_result(self):
        while self.ocr_results.get(self.image_idx) is None:
            time.sleep(0.1)
        return self.ocr_results[self.image_idx]
    
    def submit_vocabulary_job(self, idx):
        if not self.vocabulary_thread.is_alive():
            self.vocabulary_thread.start()
        full_text = self.ocr_results[idx]['full_text']
        highlighted_words = self.highlighted_words[idx]
        self.job_queue_vocabulary.put((idx, (highlighted_words, full_text)))

    def submit_vocabulary_empty_job(self):
        self.job_queue_vocabulary.put((None, (None, None)))

    def prev_image(self):
        self.submit_vocabulary_job(self.image_idx)
        self.image_idx = max(0, self.image_idx - 1)

    def next_image(self):
        self.submit_vocabulary_job(self.image_idx)
        self.image_idx = min(len(self.uploaded_images) - 1, self.image_idx + 1)

    def update_highlighted_words(self, words):
        words = [word.strip() for word in words.split('\n') if word.strip()]
        self.highlighted_words[self.image_idx] = words

    def create_vocabulary_doc(self, title):
        self.submit_vocabulary_job(self.image_idx)
        self.submit_vocabulary_empty_job()
        self.stage = 'create_vocabulary_doc'
        self.doc_title = title

    def get_vocabulary_doc(self):
        self.vocabulary_thread.join()
        vocabulary = []
        for i in range(len(self.uploaded_images)):
            vocabulary += self.vocabulary_results[i]

        doc = io.BytesIO()
        word_ocr.vocabulary_to_doc(vocabulary, doc, self.doc_title)
        return vocabulary, doc

def image_to_vocabulary(app_state):
    stage = app_state.stage

    if stage == 'upload_file':
        st.markdown("### 上传包含高亮单词的图片（可上传多张）")
        uploaded_images = st.file_uploader("", accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_images:
            uploaded_images.sort(key=lambda x: x.name)
            st.write("已上传的图片")
            cols = st.columns(3)            
            for i, uploaded_image in enumerate(uploaded_images):
                with cols[i % 3]:
                    st.image(uploaded_image, caption=f"Image {i + 1}", use_column_width=True)
        else:
            st.write("图片样例")
            st.image("image_sample.jpg", width=600)
        if st.button("从图片中抽取高亮单词", disabled=(len(uploaded_images) == 0)):
            app_state.upload_images(uploaded_images)
            st.rerun()

    if stage == 'verify_ocr_results':
        with st.spinner('正在分析图片...'):
            ocr_result = app_state.get_current_ocr_result()
        image_idx = app_state.image_idx
        image = ocr_result['debug_image']
        highlighted_words = app_state.highlighted_words.get(image_idx) or ocr_result['highlighted_words']
        print(f"highlighted_words: {highlighted_words}")
        st.markdown("### 输入单词本标题")
        title = st.text_input("", placeholder="请输入单词本的标题", label_visibility="collapsed")

        st.markdown("### 编缉或添加单词短语（每行一个）")
        cols = st.columns([0.6, 0.4])
        with cols[0]:
            st.image(image, caption=f"图片 {image_idx+1}", use_column_width=True)
        with cols[1]:
            height = len(highlighted_words) * 25 + 75
            st.text_area("", value='\n'.join(highlighted_words), height=height, key="hl_textarea", label_visibility="collapsed")
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button("上一张图片", key="prev_button", disabled=(image_idx == 0)):
                    app_state.update_highlighted_words(st.session_state['hl_textarea'])
                    app_state.prev_image()
                    st.rerun()
            with button_col2:
                if st.button("下一张图片", key="next_button", disabled=(image_idx == len(app_state.uploaded_images) - 1)):
                    app_state.update_highlighted_words(st.session_state['hl_textarea'])
                    app_state.next_image()
                    st.rerun()
            if image_idx == len(app_state.uploaded_images) - 1:
                if st.button("创建单词本", use_container_width=True, type="primary"):
                    app_state.update_highlighted_words(st.session_state['hl_textarea'])
                    app_state.create_vocabulary_doc(title or "Vocabulary")
                    st.rerun()
    
    if stage == 'create_vocabulary_doc':
        download_vocabulary_flow(app_state)

class DocAppState:
    def __init__(self):
        self.stage = "upload_file"

    def upload_doc(self, uploaded_doc, title):
        self.doc_title = title
        self.uploaded_doc = uploaded_doc
        self.stage = 'create_vocabulary_doc'

    def get_vocabulary_doc(self):
        output_doc = io.BytesIO()
        vocabulary = word_ocr.doc_to_vocabulary(self.uploaded_doc, output_doc, title=self.doc_title)
        return vocabulary, output_doc

def doc_to_vocabulary(app_state):
    stage = app_state.stage
    if stage == 'upload_file':
        st.markdown("### 上传包含高亮单词的 Word 文件")
        uploaded_doc = st.file_uploader("", label_visibility="collapsed")
        if not uploaded_doc:
            st.write("文件样例")
            st.image("doc_sample.jpg", width=600)
        else:
            st.markdown("### 输入单词本标题")
            default_title = "单词本 - " + uploaded_doc.name.rsplit('.', 1)[0]
            title = st.text_input("Vocabulary Title", value=default_title, placeholder="请输入单词本的标题", label_visibility="collapsed")
            if st.button("根据文档中的高亮文本生成单词本", disabled=(not title)):
                with st.spinner('正在生成单词本...'):
                    app_state.upload_doc(uploaded_doc, title)
                    st.rerun()
    if stage == 'create_vocabulary_doc':
        download_vocabulary_flow(app_state)

def download_vocabulary_flow(app_state):
    with st.spinner('正在创建单词本...'):
        vocabulary, doc = app_state.get_vocabulary_doc()
    st.markdown("### 下载单词本")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="下载 Word 格式的单词本",
            data=doc.getvalue(),
            file_name=f"{app_state.doc_title}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            type="primary"
        )
    with col2:
        if st.button("创建另一个单词本", use_container_width=True, type="secondary"):
            del st.session_state['app_state']
            del st.session_state['img_app_state']
            del st.session_state['doc_app_state']
            st.session_state.flow = None
            st.rerun()

    table_data = [["单词", "音标", "词性", "解释"]]
    for line in vocabulary:
        if len(line) < 4:
            line += [''] * (4 - len(line))
        table_data.append(line)
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    df.index += 1  # Make index start with 1 instead of 0
    st.subheader(app_state.doc_title)
    st.table(df)

def main():
    st.set_page_config(page_title="创建单词本", layout="wide")
    flow = st.session_state.get('flow') or None
    if flow == "image_to_vocabulary":
        app_state = st.session_state.setdefault('img_app_state', ImageAppState())
    elif flow == "doc_to_vocabulary":
        app_state = st.session_state.setdefault('doc_app_state', DocAppState())
    else:
        app_state = st.session_state.setdefault('app_state', AppState())

    stage = app_state.stage
    print(f"====Rerun at {datetime.now()}=====")
    print(f"Stage: {stage}")

    if stage == 'upload_file':
        st.markdown("### 选择上传文件类型")
        flow_type = st.radio(
            "选择上传类型",
            ("包含高亮单词的图片", "包含高亮单词的 Word 文件"),
            horizontal=True,
            label_visibility="collapsed"
        )
        if flow_type == "包含高亮单词的图片":
            st.session_state.flow = "image_to_vocabulary"
        elif flow_type == "包含高亮单词的 Word 文件":
            st.session_state.flow = "doc_to_vocabulary"
        if flow != st.session_state.flow:
            st.rerun()

    if flow == "image_to_vocabulary":
        image_to_vocabulary(app_state)
    elif flow == "doc_to_vocabulary":
        doc_to_vocabulary(app_state)

if __name__ == "__main__":
    main()
