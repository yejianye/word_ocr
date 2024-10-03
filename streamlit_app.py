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
        self.stage = "upload_images"
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
        output_doc = io.BytesIO()
        word_ocr.vocabulary_to_doc(vocabulary, output_doc, self.doc_title)
        return vocabulary, output_doc


def main():
    st.title("Create Vocabulary from Images")
    app_state = st.session_state.setdefault('app_state', AppState())
    stage = app_state.stage
    print(f"====Rerun at {datetime.now()}=====")
    print(f"Stage: {stage}")

    if stage == 'upload_images':
        uploaded_images = st.file_uploader("Upload multiple images", accept_multiple_files=True)
        if uploaded_images:
            uploaded_images.sort(key=lambda x: x.name)
            st.write("Images uploaded:")
            cols = st.columns(3)
            
            for i, uploaded_image in enumerate(uploaded_images):
                with cols[i % 3]:
                    st.image(uploaded_image, caption=f"Image {i + 1}", use_column_width=True)
        if st.button("Extract Highlighted Words from Images", disabled=(len(uploaded_images) == 0)):
            app_state.upload_images(uploaded_images)
            st.rerun()

    if stage == 'verify_ocr_results':
        image_idx = app_state.image_idx
        ocr_result = app_state.get_current_ocr_result()
        image = ocr_result['image']
        highlighted_words = app_state.highlighted_words.get(image_idx) or ocr_result['highlighted_words']
        print(f"highlighted_words: {highlighted_words}")
        title =st.text_input("Enter Vocabulary Title", value="Vocabulary")
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            st.image(image, caption=f"Image {image_idx+1}", use_column_width=True)
        with cols[1]:
            height = len(highlighted_words) * 25 + 75
            st.text_area("Highlighted Words", value='\n'.join(highlighted_words), height=height, key="hl_textarea")
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button("Prev Page", key="prev_button", disabled=(image_idx == 0)):
                    app_state.update_highlighted_words(st.session_state['hl_textarea'])
                    app_state.prev_image()
                    st.rerun()
            with button_col2:
                if st.button("Next Page", key="next_button", disabled=(image_idx == len(app_state.uploaded_images) - 1)):
                    app_state.update_highlighted_words(st.session_state['hl_textarea'])
                    app_state.next_image()
                    st.rerun()
            if image_idx == len(app_state.uploaded_images) - 1:
                if st.button("Create Vocabulary Document", type="primary"):
                    app_state.update_highlighted_words(st.session_state['hl_textarea'])
                    app_state.create_vocabulary_doc(title)
                    st.rerun()
    
    if stage == 'create_vocabulary_doc':
        with st.spinner('Creating vocabulary document...'):
            vocabulary, doc = app_state.get_vocabulary_doc()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Vocabulary Document",
                data=doc.getvalue(),
                file_name=f"{app_state.doc_title}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                type="primary"
            )
        with col2:
            if st.button("Create Another Vocabulary Document", use_container_width=True, type="secondary"):
                del st.session_state['app_state']
                st.rerun()

        table_data = [["单词", "音标", "词性", "解释"]]
        for line in vocabulary:
            if len(line) < 4:
                line += [''] * (4 - len(line))
            table_data.append(line)
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        df.index += 1  # Make index start with 1 instead of 0
        st.table(df)

if __name__ == "__main__":
    main()
