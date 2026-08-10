import streamlit as st
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="ExplainASD", page_icon="🧩", layout="centered")

st.title("🧩 ExplainASD")
st.caption("A Quantum-Enhanced Transformer Framework for Explainable AI-Based Screening of Autism Spectrum Disorder from Text")

st.warning(
    "⚠️ This tool is a research-based screening aid, NOT a medical diagnostic tool. "
    "It does not replace a professional clinical assessment. If you have concerns about "
    "a child's development, please consult a qualified healthcare provider."
)

MODEL_REPO = "Sameenatasleem/explainasd-bert"

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).to(device)
    bert_model.eval()

    symptom_vectorizer = joblib.load("symptom_vectorizer.pkl")
    symptom_clf = joblib.load("symptom_clf.pkl")
    symptom_label_encoder = joblib.load("symptom_label_encoder.pkl")

    return bert_tokenizer, bert_model, symptom_vectorizer, symptom_clf, symptom_label_encoder, device


bert_tokenizer, bert_model, symptom_vectorizer, symptom_clf, symptom_label_encoder, device = load_models()

class_names = ["Non-ASD", "ASD"]
explainer = LimeTextExplainer(class_names=class_names)


def bert_predict_proba(texts):
    inputs = bert_tokenizer(
        list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    return probs


def predict_symptom(text):
    vec = symptom_vectorizer.transform([text])
    pred = symptom_clf.predict(vec)[0]
    return symptom_label_encoder.inverse_transform([pred])[0]


st.subheader("Enter a behavioral description")
user_text = st.text_area(
    "Example: \"He does not respond when I call his name and avoids eye contact.\"",
    height=120,
)

analyze_clicked = st.button("🔍 Analyze", type="primary")

if analyze_clicked:
    if not user_text.strip():
        st.error("Please enter a sentence to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            probs = bert_predict_proba([user_text])[0]
            non_asd_prob, asd_prob = probs[0], probs[1]
            prediction = "ASD" if asd_prob > non_asd_prob else "Non-ASD"

            symptom = predict_symptom(user_text)

            exp = explainer.explain_instance(user_text, bert_predict_proba, num_features=8)

        st.divider()
        st.subheader("Results")

        col1, col2 = st.columns(2)
        with col1:
            if prediction == "ASD":
                st.error(f"**Prediction: {prediction}**")
            else:
                st.success(f"**Prediction: {prediction}**")
            st.metric("Confidence", f"{max(asd_prob, non_asd_prob) * 100:.1f}%")

        with col2:
            st.info(f"**Likely Symptom Category:**\n\n{symptom}")

        st.write("**Prediction probabilities:**")
        st.progress(float(non_asd_prob), text=f"Non-ASD: {non_asd_prob*100:.1f}%")
        st.progress(float(asd_prob), text=f"ASD: {asd_prob*100:.1f}%")

        st.divider()
        st.subheader("Why did the model predict this?")
        st.caption("Words highlighted in orange pushed the prediction toward ASD; blue pushed toward Non-ASD.")

        word_weights = dict(exp.as_list())
        words = user_text.split()
        highlighted = []
        for w in words:
            clean_w = w.strip(".,!?").lower()
            matched_weight = None
            for lime_word, weight in word_weights.items():
                if lime_word.lower() == clean_w:
                    matched_weight = weight
                    break
            if matched_weight is None:
                highlighted.append(w)
            elif matched_weight > 0:
                highlighted.append(f":orange[**{w}**]")
            else:
                highlighted.append(f":blue[**{w}**]")
        st.markdown(" ".join(highlighted))

        st.write("**Top contributing words:**")
        for word, weight in exp.as_list():
            direction = "→ ASD" if weight > 0 else "→ Non-ASD"
            st.write(f"- `{word}` (weight: {weight:.4f}) {direction}")

        st.divider()
        st.caption(
            "Model: Fine-tuned BERT (89.1% accuracy, 0.901 F1-score) | "
            "Explainability: LIME | Built as part of the ExplainASD project"
        )
