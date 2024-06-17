import torch
from pathlib import Path
from flask import request, Response
from ner import bp as recognise_text_bp
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

root = Path(__file__).parent.parent


@recognise_text_bp.post('/ner')
def index():
    arr = []

    if request.method == 'POST':
        try:
            text_list = request.json['text_list']
        except Exception as e:
            return Response(e, 400)

        tokenizer = AutoTokenizer.from_pretrained(
            "Ilya-Nazimov/rubert-tiny2-odonata-f3-ner",
        )
        model = AutoModelForTokenClassification.from_pretrained(
            "Ilya-Nazimov/rubert-tiny2-odonata-f3-ner",
        )

        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                logits = model(**inputs).logits
                predictions = torch.argmax(logits, dim=2)

                predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
                arr.append(predicted_token_class)

    return Response(arr,  200)
