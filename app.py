import streamlit as st
from transformers import pipeline
from googletrans import Translator

# Choose a model for Arabic NER
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
fallback_model = "asafaya/bert-base-arabic"

try:
    ner_model = pipeline("ner", model=model_name)
except Exception as e:
    st.error(f"خطأ في تحميل نموذج التعرف على الكيانات: {e}")
    ner_model = pipeline("ner", model=fallback_model)
    model_name = fallback_model

# Load pre-trained Arabic Sentence Completion model
sentence_completion_model = pipeline("fill-mask", model="asafaya/bert-base-arabic")

# Set layout to RTL and use Arabic text
st.markdown(
    """
    <style>
    body {
        direction: rtl !important;
        text-align: right !important;
        background-color: #f8f9fa; /* Light background for contrast */
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 2em;
        color: #343a40;
    }
    .subheader {
        font-size: 1.5em;
        color: #6c757d;
    }
    .button {
        background-color: #007bff; /* Bootstrap primary color */
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .button:hover {
        background-color: #0056b3; /* Darker shade for hover */
    }
    .container {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("التعرف على الكيانات المسماة (NER) وإكمال الجمل")
st.subheader("التعرف على الكيانات المسماة " + model_name)

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

# Dropdown for language selection and translation button
with col3:
    st.subheader("ترجمة النص")
    translator = Translator()
    
    language_options = {
        "English": "en",
        "French": "fr",
        "Chinese": "zh-cn",
        "Hebrew": "he"
    }
    
    selected_language = st.selectbox("اختر اللغة:", list(language_options.keys()))
    
    if st.button("ترجمة", key="translate_button"):
        if text:
            target_language = language_options[selected_language]
            try:
                translation = translator.translate(text, dest=target_language)
                st.write(f"الترجمة إلى {selected_language}: **{translation.text}**")
            except Exception as e:
                st.error(f"خطأ أثناء ترجمة النص: {e}")

