import email
from email import policy
from email.parser import BytesParser
import base64
import os
from PIL import Image
import pytesseract
import io
import requests
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

LABELS = ['O', 'B-DEAL_NAME', 'I-DEAL_NAME', 'B-AMOUNT', 'I-AMOUNT', 'B-EXPIRATION_DATE', 'I-EXPIRATION_DATE']

class EmailProcessor:
    def __init__(self, eml_file_path=None):
        self.eml_file_path = eml_file_path
        self.email_content = {}

    def extract_email_content(self, eml_file_path):
        with open(eml_file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        self.email_content = {
            'subject': msg['subject'] or '',
            'from': msg['from'] or '',
            'to': msg['to'] or '',
            'body': '',
            'attachments': []
        }

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    self.email_content['body'] += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            self.email_content['body'] = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        return self.email_content

# Preprocessing and Fine-tuning BERT for NER
class BERTEmailNER:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(LABELS))

    def load_custom_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.rename(columns={
            'Email Text': 'text',
            'Deal Number': 'deal_name',
            'Transfer Amount': 'amount',
            'Expiration Date': 'expiration_date'
        }, inplace=True)
        return df

    def create_ner_labels(self, text, deal_name, amount, expiration_date):
        words = text.split()
        labels = ['O'] * len(words)

        def label_entity(entity, label_prefix):
            if entity:
                entity_words = entity.split()
                for i, word in enumerate(entity_words):
                    label = f'B-{label_prefix}' if i == 0 else f'I-{label_prefix}'
                    for j, w in enumerate(words):
                        if w.strip().lower() == word.strip().lower():
                            labels[j] = label

        label_entity(deal_name, 'DEAL_NAME')
        label_entity(amount, 'AMOUNT')
        label_entity(expiration_date, 'EXPIRATION_DATE')

        return labels

    def preprocess_data(self, df):
        texts = df['text'].tolist()
        all_labels = [self.create_ner_labels(row['text'], row['deal_name'], row['amount'], row['expiration_date']) for _, row in df.iterrows()]

        input_ids, attention_masks, label_ids = [], [], []

        for i, text in enumerate(texts):
            tokenized_data = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            word_ids = self.tokenizer(text, padding='max_length', truncation=True, max_length=512).word_ids()
            input_ids.append(tokenized_data['input_ids'][0])
            attention_masks.append(tokenized_data['attention_mask'][0])

            label_id = []
            for word_id in word_ids:
                if word_id is None or word_id >= len(all_labels[i]):
                    label_id.append(-100)
                else:
                    label_id.append(LABELS.index(all_labels[i][word_id]))
            label_ids.append(label_id)

        return Dataset.from_dict({
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': label_ids
        })

    def fine_tune_model(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        trainer.train()
        trainer.save_model('./fine_tuned_bert_ner')

    def extract_fields(self, email_text):
        inputs = self.tokenizer(email_text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        deal_name = amount = expiration_date = None

        for token, label_id in zip(tokens, predictions[0].numpy()):
            label = LABELS[label_id]
            if label.startswith('B-DEAL_NAME') and not deal_name:
                deal_name = token.replace('##', '')
            elif label.startswith('B-AMOUNT') and not amount:
                amount = token.replace('##', '')
            elif label.startswith('B-EXPIRATION_DATE') and not expiration_date:
                expiration_date = token.replace('##', '')

        return {
            'deal_name': deal_name,
            'amount': amount,
            'expiration_date': expiration_date
        }
class EmailSimilarityChecker:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2',hf_token=None):
        self.hf_token = hf_token
        self.model = SentenceTransformer(model_name, use_auth_token=hf_token)
        self.index = None
        self.email_texts = []

    def create_index(self, email_texts):
        self.email_texts = email_texts
        embeddings = self.model.encode(email_texts, convert_to_tensor=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.cpu().numpy())

    def find_duplicates(self, query_text, threshold=0.7):
        query_embedding = self.model.encode([query_text], convert_to_tensor=True).cpu().numpy()
        distances, indices = self.index.search(query_embedding, k=5)
        duplicates = []
        for i, dist in zip(indices[0], distances[0]):
            if dist >= threshold:
                duplicates.append(self.email_texts[i])
        return duplicates

# Example Usage
    processor = EmailProcessor()
    email_data = processor.extract_email_content('sample.eml')

    classifier = BERTEmailNER()
    df = classifier.load_custom_data('CUSTOM_DTASET.csv')
    train_df = df.sample(frac=0.8, random_state=42)
    eval_df = df.drop(train_df.index)

    train_dataset = classifier.preprocess_data(train_df)
    eval_dataset = classifier.preprocess_data(eval_df)
    hf_token = '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'
    # classifier.fine_tune_model(train_dataset, eval_dataset)
    extracted_fields = classifier.extract_fields(email_data['body'])
    print("Extracted Fields:", extracted_fields)
    sample_emails = ["Email is about lalllaa of Deal A with amount $5000", "Deal A  and confirmed with expiration date 12/12/2025", "Random finance discussion"]
    checker = EmailSimilarityChecker(hf_token=hf_token)
    checker.create_index(sample_emails)
    duplicates = checker.find_duplicates(email_data['body'])
    print("Potential Duplicates:", duplicates)
