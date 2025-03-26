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


REQUEST_TYPES = [
    "Adjustment", "AU Transfer", "Closing Notice", "Commitment Change",
    "Fee Payment", "Money Movement-Inbound", "Money Movement - Outbound"
]

SUB_REQUEST_TYPES = [
    "Reallocation Fees", "Amendment Fees", "Reallocation Principal", "Cashless Roll",
    "Decrease", "Increase", "Ongoing Fee", "Letter of Credit Fee", "Principal",
    "Interest", "Principal+Interest+Fee", "Principal + Interest", "Timebound", "Foreign Currency"
]

class EmailProcessor:
    def __init__(self, eml_file_path=None):
        self.eml_file_path = eml_file_path
        self.email_content = {}

    def extract_email_content(self, eml_file):
        if isinstance(eml_file, (str, bytes, os.PathLike)):
        # File path or bytes input
         with open(eml_file, 'rb') as file:
            msg = email.message_from_binary_file(file)
        elif hasattr(eml_file, 'read'):
        # Streamlit UploadedFile or file-like object
            msg = email.message_from_binary_file(eml_file)

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
                elif part.get_filename():
                    attachment_data = part.get_payload(decode=True)
                    attachment_name = part.get_filename()
                    self.email_content['attachments'].append({'name': attachment_name, 'data': attachment_data})
        else:
            self.email_content['body'] = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        return self.email_content

    def print_email_content(self):
        print("Subject:", self.email_content.get('subject', 'N/A'))
        print("From:", self.email_content.get('from', 'N/A'))
        print("To:", self.email_content.get('to', 'N/A'))
        print("Body:", self.email_content.get('body', 'N/A'))
        print("Attachments:", len(self.email_content.get('attachments', [])))

    # def process_attachments(self):
    #     for attachment in self.email_content['attachments']:
    #         filename = attachment['name']
    #         print(f"Processing attachment: {filename}")
    #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    #             image = Image.open(io.BytesIO(attachment['data']))
    #             text = pytesseract.image_to_string(image)
    #             print("Extracted Text from Image:", text)

    def call_openrouter_deepseek(self, email_text: str) -> dict:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer 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",
            "Content-Type": "application/json",
        }

        messages = [
        {
            "role": "user",
            "content": f"""
            **System Prompt:**
            **Task:**
            You are an AI model specialized in processing emails and their attached files (.docx, .txt, .pdf) for Commercial Bank Lending Service teams. Your objective is to analyze the email content and attachments to accurately interpret the sender's intent, classify the request, and extract key information.

            **Instructions:**
            1. **Request Classification:**
                - Analyze the email body and attachments to classify the request into one of the predefined **Request Types** {', '.join(REQUEST_TYPES)} and **Sub-Request Types**:{', '.join(SUB_REQUEST_TYPES)}. 
                - Identify the primary intent when multiple requests are present, providing a clear classification of the main request.
                - Give the confidence Score for the output given

            2. **Contextual Data Extraction:**
                - Extract relevant fields from the email and attachments, including:
                    - **Deal Name**
                    - **Loan Amount**
                    - **Expiration Date**
                    - **Customer Name**
                    - **Reference Number**

            3. **Handling Multi-Request Emails:**
                - Identify and list all requests within emails containing multiple topics.
                - Determine the primary request representing the sender’s main intent.

            4. **Duplicate Detection:**
                - Detect and flag potential duplicate emails, including:
                    - Multiple replies or forwards within the same thread.
                    - Identical content received through different channels.

            5. **Priority Rules for Extraction:**
                - Prioritize the email body for classifying requests.
                - Extract numerical and tabular data from attachments when necessary.

            **Expected Output:**
            - **Request Type:** [Identified type]
            - **Sub-Request Type:** [Identified sub-type]
            - **Confidence:** [Confidence Score]
            - **Extracted Data:**
                - **Deal Name:** [Extracted value]
                - **Loan Amount:** [Extracted value]
                - **Expiration Date:** [Extracted value]
                - **Customer Name:** [Extracted value]
                - **Reference Number:** [Extracted value]
            - **Reasoning:** [Provide a clear justification for classification and data extraction decisions]
            """
        }
    ]

        payload = {"model": "deepseek/deepseek-r1:free", "messages": messages}

        try:
            response = requests.post(url=url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return {
                "primary_request": "Error",
                "sub_request": None,
                "confidence": 0.0,
                "details": str(e)
            }

        result = response.json()
        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "Unknown")
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]

        primary_request = lines[0] if len(lines) > 0 else "Unknown"
        sub_request = lines[1] if len(lines) > 1 and lines[1] != "" else None

        if sub_request and primary_request in REQUEST_TYPES and sub_request not in SUB_REQUEST_TYPES:
            sub_request = None

        confidence = lines[2] if len(lines) > 0 else "60"
        Deal_Name=lines[4] if len(lines) > 0 else "Unknown"
        Loan_Amount=lines[5] if len(lines) > 0 else "Unknown"
        Expiration_Date=lines[6] if len(lines) > 0 else "Unknown"                

        return {
            "primary_request": primary_request,
            "sub_request": sub_request,
            "confidence": confidence,
            "Deal_Name":Deal_Name,
            "Loan_Amount":Loan_Amount,
            "Expiration_Date":Expiration_Date,
            "details": "DeepSeek R1 analysis confirms the classification."
        }
    
    def check_duplicate_mail_or_new_request(email_text):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json",
        }

        messages = [
            {
                "role": "user",
                "content": f"Analyse the mail \n\n{email_text} to determine if it is a confirmation of a previous mail or a new service request.Consider the following indicators of confirmation:- Subject containing 'RE:' or 'FWD:'- Presence of previous mail threads or quoted content- Repetitive language like 'Following up' or 'Per our last conversation'- Lack of new queries or actionable requests. Output should be only one word 'Y' if its a confirmation mail or 'N' if new service request"
            }
        ]

        payload = {"model": "deepseek/deepseek-r1:free", "messages": messages}

        try:
            response = requests.post(url=url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            message_content = result.get("choices", [{}])[0].get("message", {}).get("content", "Unknown").strip()
            if message_content == 'Y':
                print("This is a confirmation mail.")
                print(processor.make_prompt_request_for_duplicate_mail())
            else:
                print("New request")
            # return result.get("choices", [{}])[0].get("message", {}).get("content", "Unknown").strip()
        except requests.exceptions.RequestException as e:
            return str(e)
        
# Function to extract the deal name using DeepSeek
    def make_prompt_request_for_duplicate_mail(clean_text):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json",
        }

        messages = [
            {
                "role": "user",
                "content": f"Take the data \n\n{clean_text} and identify only the Deal name."
            }
        ]

        payload = {"model": "deepseek/deepseek-r1:free", "messages": messages}

        try:
            response = requests.post(url=url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            deal_name = result.get("choices", [{}])[0].get("message", {}).get("content", "Unknown").strip()
            response_data = {
                "status": "success",
                "message": "This is a duplicate request type",
                "result": f"This is a duplicate request type.\n{deal_name}"
            }
            return response_data
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
     
   
#Tried bert model for Duplicates and NER but results not good:
# class BERTEmailClassifier:
#     def __init__(self, model_name='bert-base-uncased'):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(REQUEST_TYPES))
#         self.ner_model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')

#     def preprocess_data(self, emails, labels):
#         tokenized_data = self.tokenizer(emails, padding=True, truncation=True, max_length=512)
#         return Dataset.from_dict({
#             'input_ids': tokenized_data['input_ids'],
#             'attention_mask': tokenized_data['attention_mask'],
#             'labels': labels
#         })

#     def fine_tune_model(self, train_dataset, eval_dataset):
#         training_args = TrainingArguments(
#             output_dir='./results',
#             num_train_epochs=3,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             evaluation_strategy='epoch',
#             save_strategy='epoch'
#         )

#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset
#         )
#         trainer.train()
#         trainer.save_model('./fine_tuned_bert')

#     def classify(self, email_text):
#         inputs = self.tokenizer(email_text, return_tensors='pt', truncation=True, max_length=512)
#         with torch.no_grad():
#             logits = self.model(**inputs).logits
#             predicted_class = torch.argmax(logits, dim=1).item()
#         return REQUEST_TYPES[predicted_class]

#     def extract_fields(self, email_text):
#         inputs = self.tokenizer(email_text, return_tensors='pt', truncation=True, max_length=512)
#         with torch.no_grad():
#             logits = self.ner_model(**inputs).logits
#             predictions = torch.argmax(logits, dim=2)
#         tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

#         deal_name = amount = expiration_date = None
#         for token, label_id in zip(tokens, predictions[0].numpy()):
#             label = self.ner_model.config.id2label[label_id]
#             if label == 'B-ORG' and not deal_name:
#                 deal_name = token.replace('##', '')
#             elif label == 'B-MONEY' and not amount:
#                 amount = token.replace('##', '')
#             elif label == 'B-DATE' and not expiration_date:
#                 expiration_date = token.replace('##', '')

#         return {
#             'deal_name': deal_name,
#             'amount': amount,
#             'expiration_date': expiration_date
#         }


# Main
if __name__ == "__main__":
    processor = EmailProcessor()
    email = processor.extract_email_content('adjustment_reallocation.eml')
    processor.print_email_content()
    # processor.check_duplicate_mail_or_new_request()
    result = processor.call_openrouter_deepseek(email['body'])
    print("Classification Result:", result)
  

    # classifier = BERTEmailClassifier()
    # print("Classified as:", classifier.classify(email_data['body']))
    # extracted_fields = classifier.extract_fields(email_data['body'])
    # print("Extracted Fields:", extracted_fields)
