"""Gradio interface for the Nigerian Language Identifier capstone."""
from pathlib import Path
import joblib
import gradio as gr

MODEL_PATH = Path(__file__).with_name("nigerian_language_identifier_svm.pkl")
model = joblib.load(MODEL_PATH)  # Only load a model file you created/trust.


def identify_language(text: str) -> str:
    if not text or not text.strip():
        return "Please enter some text."
    return f"Predicted Language: {model.predict([text.strip()])[0]}"


app = gr.Interface(
    fn=identify_language,
    inputs=gr.Textbox(lines=3, label="Enter Your Text", placeholder="Type a phrase in Yoruba, Hausa, Igbo or English..."),
    outputs=gr.Textbox(label="Language Prediction"),
    title="Nigerian Language Identifier",
    description="A machine learning application that identifies Yoruba, Hausa, Igbo and English text using a character-level TF-IDF and Linear SVM classifier. Predictions may be less reliable for very short or mixed-language text.",
    examples=[["How are you today?"], ["Ẹ káàárọ̀"], ["Sannu da zuwa"], ["Kedu ka ị mere"]],
)

if __name__ == "__main__":
    app.launch()
