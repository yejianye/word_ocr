import streamlit as st
import io
from datetime import datetime
import word_ocr

def today_str():
    return datetime.now().strftime("%Y-%m-%d")

def images_to_text():
    st.title("Convert scanned images to text")

    uploaded_images = st.file_uploader("Upload multiple images", accept_multiple_files=True)

    if uploaded_images:
        uploaded_images.sort(key=lambda x: x.name)
        st.write("Images uploaded:")
        cols = st.columns(3)
        
        for i, uploaded_image in enumerate(uploaded_images):
            with cols[i % 3]:
                st.image(uploaded_image, caption=f"Image {i + 1}", use_column_width=True)

    if st.button("Convert to Word Document"):
        output_doc = io.BytesIO()
        word_ocr.images_to_doc(uploaded_images, output_doc, 
                               process_callback=lambda i, total: st.write(f"Processing {i}/{total} images..."))
        st.download_button(
            label="Download Word Document",
            data=output_doc.getvalue(),
            file_name=f"scanned-doc-{today_str()}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

def text_to_vocabulary():
    st.title("Create vocabulary from text")
    title = st.text_input("Document Title", value="Vocabulary")
    uploaded_doc = st.file_uploader("Upload a Word document")
    if uploaded_doc and st.session_state.get('text_to_vocabulary_doc') != uploaded_doc:
        output_doc = io.BytesIO()
        st.write("Building vocabulary...")
        word_ocr.gen_vocabulary_doc(uploaded_doc, output_doc, title)
        st.session_state['text_to_vocabulary_doc'] = uploaded_doc
        st.download_button(
            label="Download Vocabulary Document",
            data=output_doc.getvalue(),
            file_name=f"vocabulary-{today_str()}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Images to Text", "Text to Vocabulary"])

    if page == "Images to Text":
        images_to_text()

    elif page == "Text to Vocabulary":
        text_to_vocabulary()


if __name__ == "__main__":
    main()
