from recognise_text import bp as recognise_text_bp
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset
from datasets import ClassLabel, Value, Sequence
import datasets
import evaluate
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

metric = evaluate.load("seqeval")
THIS_FOLDER = Path(__file__).parent.resolve()

@recognise_text_bp.post('/recognise_text')
def index():
    # Подключаем токенизатор из предобученной на русском языке модели
    model_checkpoint = "cointegrated/rubert-tiny2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Открываем csv датасет
    dataset_reader = pd.read_csv(f'{THIS_FOLDER}/train.csv')
    dataset = dataset_reader.head(300)
    dataset_validation = dataset_reader.tail(40)

    # Создаём списки из столбцов датасета
    text_list = dataset['processed_text'].tolist()
    marks_json = dataset['target_labels_positions'].tolist()
    text_list_validation = dataset_validation['processed_text'].tolist()
    marks_json_validation = dataset_validation['target_labels_positions'].tolist()

    # Создаём список позиций токенов
    def marks_json_to_ner_tags_list(marks_json):
        ner_tags = []
        for mark in marks_json:
            mark = mark.replace("\'", "\"")
            marks_arr = json.loads(mark)
            row_ner_tags = [[0], [0], [0]]
            if 'B-discount' in marks_arr:
                row_ner_tags[0] = marks_arr['B-discount']
            if 'B-value' in marks_arr:
                row_ner_tags[1] = marks_arr['B-value']
            if 'I-value' in marks_arr:
                row_ner_tags[2] = marks_arr['I-value']
            ner_tags.append(row_ner_tags)
        return ner_tags

    ner_tags = marks_json_to_ner_tags_list(marks_json)
    ner_tags_validation = marks_json_to_ner_tags_list(marks_json_validation)

    # Создаём список токенов в соответствии с разметкой датасета
    count = 0
    ner_tokens = []
    text_split = []
    for text in text_list:
        text_len = len(text.split())
        row_ner_tokens = [0] * text_len
        B_discount = ner_tags[count][0][0]
        B_value = ner_tags[count][1][0]
        I_value = ner_tags[count][2][0]
        if B_discount:
            for b_disc in ner_tags[count][0]:
                row_ner_tokens[b_disc] = 1
        if B_value:
            for b_val in ner_tags[count][1]:
                row_ner_tokens[B_value] = 2
        if I_value:
            for i_val in ner_tags[count][2]:
                row_ner_tokens[I_value] = 3
        ner_tokens.append(row_ner_tokens)
        text_split.append(text.split())
        count += 1

    count_validation = 0
    ner_tokens_validation = []
    text_split_validation = []

    for text_validation in text_list_validation:
        text_len = len(text_validation.split())
        row_ner_tokens = [0] * text_len
        B_discount = ner_tags_validation[count_validation][0][0]
        B_value = ner_tags_validation[count_validation][1][0]
        I_value = ner_tags_validation[count_validation][2][0]
        if B_discount:
            for b_disc in ner_tags_validation[count_validation][0]:
                row_ner_tokens[b_disc] = 1
        if B_value:
            for b_val in ner_tags_validation[count_validation][1]:
                row_ner_tokens[B_value] = 2
        if I_value:
            for i_val in ner_tags_validation[count_validation][2]:
                row_ner_tokens[I_value] = 3
        ner_tokens_validation.append(row_ner_tokens)
        text_split_validation.append(text_validation.split())
        count_validation += 1

    # Сопоставляем нер токены с токенизированным текстом
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Начало нового слова!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Специальный токен
                new_labels.append(-100)
            else:
                # То же слово, что и предыдущий токен
                label = labels[word_id]
                # Если метка B-XXX, заменяем ее на I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    # Готовый к токенизации датасет
    new_dataset = {
        'tokens': text_split, 'ner_tags': ner_tokens
    }
    new_dataset_validation = {
        'tokens': text_split_validation, 'ner_tags': ner_tokens_validation
    }
    train_dataset = Dataset.from_dict(new_dataset)
    train_dataset_validation = Dataset.from_dict(new_dataset_validation)
    raw_dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": train_dataset_validation, "test": train_dataset_validation})

    new_features = raw_dataset["train"].features.copy()
    new_features["ner_tags"] = Sequence(
        feature=ClassLabel(num_classes=4, names=['O', 'B-discount', 'B-value', 'I-value'], names_file=None, id=None),
        length=-1, id=None)
    raw_dataset = raw_dataset.cast(new_features)

    ner_feature = raw_dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names

    # Токенизация датасета
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tokenized_datasets = raw_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    batch = data_collator([tokenized_datasets["train"][i] for i in range(3)])

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Удаляем игнорируемый индекс (специальные токены) и преобразуем в метки
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        "lct-rubert-tiny2-ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.push_to_hub(commit_message="Training complete")
    return 'Hello'
