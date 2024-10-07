import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Choose a model for Arabic NER
model_name = "Davlan/xlm-roberta-large-ner-hrl"
fallback_model = "asafaya/bert-base-arabic"

try:
    # Load tokenizer and model explicitly for NER
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_model = pipeline("ner", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.write(f"Falling back to {fallback_model}...")
    # Load fallback model if the primary one fails
    ner_model = pipeline("ner", model=fallback_model)
    model_name=fallback_model

# Set layout to RTL and use Arabic text
st.markdown(
    """
    <style>
    body {
        direction: rtl !important;
        text-align: right !important;
    }
    .css-1d391kg {
        direction: rtl !important;
        text-align: right !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("التعرف على الكيانات المسماة (NER) وإكمال الجمل")
st.subheader("التعرف على الكيانات المسماة" + model_name)

# Input text for NER
text = st.text_area("أدخل نصًا للتعرف على الكيانات:")

if st.button("التعرف على الكيانات"):
    if text:
        entities = ner_model(text)
        threshold = 0.8  # Minimum confidence score to show entities
        for entity in entities:
            if entity['score'] >= threshold:
                st.write(f"{entity['word']}: {entity['entity']} (النسبة: {entity['score']:.4f})")

# Load pre-trained Arabic Sentence Completion model
sentence_completion_model = pipeline("fill-mask", model="asafaya/bert-base-arabic")

st.subheader("إكمال الجمل")
input_sentence = st.text_area("أدخل جملة تحتوي على [MASK]:")

if st.button("إكمال الجملة"):
    if input_sentence:
        completions = sentence_completion_model(input_sentence)
        for completion in completions:
            st.write(f"الخيار: {completion['sequence']} (النسبة: {completion['score']:.4f})")
