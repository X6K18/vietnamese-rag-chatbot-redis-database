import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from backend.config import MODEL_PATH, TOKENIZER_PATH, LABEL_PATH, DEVICE

class TextClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.label_encoder = joblib.load(LABEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=len(self.label_encoder.classes_)
        )
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return self.label_encoder.inverse_transform([pred])[0]