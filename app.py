import streamlit as st
from transformers import pipeline
from googletrans import Translator

# Choose a model for Arabic NER
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
fallback_model = "asafaya/bert-base-arabic"

# Load Named Entity Recognition (NER) model
try:
    ner_model = pipeline("ner", model=model_name)
except Exception as e:
    st.error(f"خطأ في تحميل نموذج التعرف على الكيانات: {e}")
    ner_model = pipeline("ner", model=fallback_model)
    model_name = fallback_model

# Load pre-trained Arabic Sentence Completion model
sentence_completion_model = pipeline("fill-mask", model="asafaya/bert-base-arabic")

# Set layout to RTL and use Arabic text
st.set_page_config(page_title="التعرف على الكيانات المسماة وإكمال الجمل", layout="wide")
st.markdown(
    """
    <style>
    body {
        direction: rtl !important;
        text-align: right !important;
        background-color: #f0f4f8;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 2.5em;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.5em;
        color: #555;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .button {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        font-size: 1em;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #0056b3;
    }
    .container {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 20px auto;
        max-width: 800px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("التعرف على الكيانات المسماة (NER) وإكمال الجمل")
st.subheader("التعرف على الكيانات المسماة باستخدام " + model_name)

# Input text for both NER and Sentence Completion
text = st.text_area("أدخل نصًا لإجراء التعرف على الكيانات وإكمال الجملة:", height=100)

# Create columns for buttons and translation
col1, col2, col3 = st.columns(3)

# Button for Named Entity Recognition
with col1:
    if st.button("التعرف على الكيانات", key="ner_button"):
        if text:
            entities = ner_model(text)
            threshold = 0.8  # Minimum confidence score to show entities
            st.subheader("الكيانات المعترف بها:")
            for entity in entities:
                if entity['score'] >= threshold:
                    st.write(f"**{entity['word']}**: {entity['entity']} (النسبة: {entity['score']:.4f})")

# Button for Sentence Completion
with col2:
    if st.button("إكمال الجملة", key="completion_button"):
        if text:
            input_sentence = text.replace("[MASK]", "[MASK]")  # Ensure the mask token is correct
            
            # Ensure there is exactly one mask token
            if input_sentence.count("[MASK]") != 1:
                st.error("يجب أن تحتوي الجملة على رمز ماسك واحد فقط ([MASK]).")
            else:
                try:
                    completions = sentence_completion_model(input_sentence)
                    st.subheader("إكمال الجمل:")
                    for completion in completions:
                        st.write(f"الخيار: **{completion['sequence']}** (النسبة: {completion['score']:.4f})")
                except Exception as e:
                    st.error(f"خطأ أثناء إكمال الجملة: {e}")

# Dropdown for language selection and translation button in a single row
with col3:
    # Create columns for dropdown and button in the same row
    lang_col1, lang_col2 = st.columns([2, 1])  # Adjust the proportions as needed
    translator = Translator()
    
    language_options = {
        "English": "en",
        "French": "fr",
        "Chinese": "zh-cn",
        "Hebrew": "he"
    }

    with lang_col1:
        # Removed the title and placeholder
        selected_language = st.selectbox("اختر اللغة لتتم الترجمة لها :", list(language_options.keys()), key="language_select")

    with lang_col2:
        st.write("")
        st.write("")
        translate_button = st.button("ترجمة", key="translate_button")
    
    translation_output = st.empty()  # Placeholder for translation output
    
    if translate_button:
        if text:
            target_language = language_options[selected_language]
            try:
                translation = translator.translate(text, dest=target_language)
                translation_output.markdown(f"الترجمة إلى {selected_language}: **{translation.text}**")
            except Exception as e:
                translation_output.error(f"خطأ أثناء ترجمة النص: {e}")

# Add some spacing between sections
st.markdown("<br>", unsafe_allow_html=True)
