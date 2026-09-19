import spacy
import torch
from transformers import pipeline

class LinguisticEngine:

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            raise RuntimeError(f"spaCy model load error: {e}")

        self.grammar_fixer = pipeline(
            "text2text-generation",
            model="t5-small",
            tokenizer="t5-small",
            device=0 if torch.cuda.is_available() else -1,
            max_length=128,
            num_return_sequences=1,
        )

    def extract_entities(self, text: str):
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def polish_response(self, text: str) -> str:
        if not text:
            return ""
        try:
            result = self.grammar_fixer(text)
            polished = result[0].get("generated_text", "").strip()
            polished = " ".join(polished.split())
            return polished
        except Exception:
            return text
