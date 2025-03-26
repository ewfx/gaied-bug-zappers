# *Email Categorization and Entity Mapping using DeepSeek and SBERT*

### *Overview*
This project focuses on processing emails and extracting meaningful insights using NLP models. The workflow is broken down into three parts:

1. *Email Categorization* – Classify emails into request type, sub-request type, and assign a confidence score.
2. *Duplicate Detection* – Identify duplicate emails using NLP models.
3. *Named Entity Mapping* – Extract key information such as expiration date, deal name, and amount.

---

### *Project Structure*

📂 project_root
│── main.py               # Main script for DeepSeek-based categorization, duplicate detection, and entity extraction
│── other_models.py       # Contains alternative approaches using SBERT and BERT
│── README.md             # Project documentation
└── data/                 # Sample email datasets (if applicable)


---

### *1️⃣ Email Categorization*
- *Goal:* Classify emails into *request type, sub-request type, and **confidence score*.
- *Approach:* Used *DeepSeek* model for better accuracy.
- *Implementation:* 
  - DeepSeek processes email content.
  - Generates request type, sub-request type, and confidence score.
- *Code Location:* main.py

---

### *2️⃣ Duplicate Email Detection*
- *Goal:* Identify duplicate emails.
- *Approach:* 
  - *Tried Hugging Face Pretrained BERT Models* – Results were not satisfactory.
  - *Final Approach: DeepSeek* – Improved accuracy in detecting duplicates.
- *Code Location:*
  - DeepSeek implementation in main.py
  - Initial BERT approach in other_models.py

---

### *3️⃣ Named Entity Mapping*
- *Goal:* Extract key fields such as *expiration_date, deal_name, amount*.
- *Approach:*
  - *Tried SBERT (Sentence-BERT)* – Results were inaccurate.
  - *Final Approach: DeepSeek* – Better performance in extracting relevant entities.
- *Code Location:*
  - DeepSeek implementation in main.py
  - SBERT approach in other_models.py

---

### *Setup and Installation*
1. *Clone the Repository*
   sh
   git clone <repo_url>
   cd project_root
   
2. *Install Dependencies*
   sh
   pip install streamlit Pillow pytesseract requests torch transformers datasets faiss-cpu

   NOTE: pip install faiss-cpu (For CPU)
	 	pip install faiss-gpu (For GPU)

   streamlit - For the frontend (UI)

   email - For email parsing (comes with Python)

   Pillow (PIL) - For image processing

   pytesseract - For OCR (Optical Character Recognition)

   requests - For API requests

   json - For handling JSON responses (comes with Python)

   re - For regular expressions (comes with Python)

   io - For handling in-memory file objects (comes with Python)
   
3. *Run the Script*
   sh
   python main.py
   streamlit run ui.py (command to run UI)

	
   

---

### *Future Enhancements*
- Improve Named Entity Recognition (NER) by fine-tuning models.
- Experiment with additional NLP techniques for better accuracy.
- Automate retraining with new email data.

---

## *PowerPoint Documentation Points*

### *Slide 1: Project Title*
- *"Automated Email Categorization & Entity Mapping using NLP"*

### *Slide 2: Problem Statement*
- Emails contain unstructured data.
- Manual classification is inefficient.
- Need for automation in classification, duplicate detection, and key information extraction.

### *Slide 3: Solution Breakdown*
- Divided into *3 parts*:
  1. Email Categorization
  2. Duplicate Detection
  3. Named Entity Mapping

### *Slide 4: Email Categorization*
- Used *DeepSeek* for classification.
- Extracted *request type, sub-request type, confidence score*.
- *Why DeepSeek?* Outperformed other models in accuracy.

### *Slide 5: Duplicate Detection*
- *Initial Approach:* Hugging Face *BERT* models (low accuracy).
- *Final Approach:* Used *DeepSeek* (better performance).
- Code in main.py, initial BERT approach in other_models.py.

### *Slide 6: Named Entity Mapping*
- Extracting *expiration_date, deal_name, amount*.
- *Initial Approach:* SBERT (inaccurate).
- *Final Approach:* *DeepSeek* (more reliable).
- Code in main.py, SBERT approach in other_models.py.

### *Slide 7: Technical Implementation*
- *Project structure overview* (referencing README.md).
- *Technologies Used:* Python, NLP Models (DeepSeek, SBERT, BERT).

### *Slide 8: Challenges Faced*
- Initial NLP models (BERT, SBERT) did not provide accurate results.
- DeepSeek provided better results but still has scope for improvement.

### *Slide 9: Future Scope*
- Fine-tuning DeepSeek for better entity extraction.
- Experimenting with other NLP techniques.
- Automating model retraining.

### *Slide 10: Conclusion*
- Successfully automated *email categorization, duplicate detection, and key entity extraction*.
- *DeepSeek was the most effective approach* across all three tasks.


NOTE - Token expiration:

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 -> this is the key used in the codebase presently.
Generated from https://openrouter.ai/settings/keys
The key expires after a particular interval of time.
Chatbot key - sk-7d0045022d574e34a9bdc67f70ba77ca
