import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

THIS_FOLDER = Path(__file__).parent.resolve()

dataset_reader = pd.read_csv(f'{THIS_FOLDER}/gt_test.csv')
text_list = dataset_reader['processed_text'].tolist()

tokenizer = AutoTokenizer.from_pretrained(
   "Ilya-Nazimov/rubert-tiny2-odonata-f3-ner",
)
model = AutoModelForTokenClassification.from_pretrained(
    "Ilya-Nazimov/rubert-tiny2-odonata-f3-ner",
)

arr = []

for text in text_list:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    arr.append(predicted_token_class)

df = pd.DataFrame({'processed_text': text_list, 'label': arr})
df.to_csv('submission.csv', index=False)
