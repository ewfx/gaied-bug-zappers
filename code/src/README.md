Bugzappers - Email Classification and Service Request Creation

This project is designed to classify emails into predefined categories using AI, extract relevant information, and generate service requests. It uses Streamlit for the UI, DeepSeek AI for classification, and Tesseract OCR for text extraction from attachments.

Features

Email Classification: Classifies emails into specific request and sub-request types.

Field Extraction: Extracts details like deal name, loan amount, expiration date, customer name, and reference number.

Duplicate Detection: Detects whether an email is a follow-up or a new request.

Service Request Creation: Generates a service request form with extracted details.

Prerequisites

Ensure you have the following installed:

Python 3.10+

Tesseract OCR

pip (Python Package Manager)

Install Tesseract OCR

Windows: Download from Tesseract OCR GitHub Releases

Ubuntu: sudo apt install tesseract-ocr

Mac: brew install tesseract

Installation

Clone the repository:

git clone https://github.com/your-repo/bugzappers-email-classification.git
cd bugzappers-email-classification

Install dependencies:

pip install -r requirements.txt

Verify Tesseract installation:

tesseract --version

Set the path to Tesseract in main.py if required:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Usage

1. Start the Application

streamlit run ui.py

2. Email Classification

Upload an .eml file using the UI.

The AI will analyze the email, extract fields, and classify it into a request and sub-request type.

3. Service Request Form

Navigate to the Service Request Form page.

Review and edit the extracted fields.

Submit the form to generate a service request.

Project Structure

.
├── main.py               # Email processing and classification logic
├── ui.py                 # Streamlit UI for file upload and service request
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── Wells_Fargo_Logo.png  # Logo for UI

API Configuration

This project uses DeepSeek AI for classification. Ensure you have a valid API key and update main.py with your key:

headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json",
}

Error Handling

FileNotFoundError: Ensure the correct file path is provided.

401 Unauthorized: Verify your API key for DeepSeek AI.

Tesseract Error: Confirm Tesseract is installed and correctly configured.

Contributing

Feel free to fork the repository and submit pull requests. Ensure changes are well-documented.

License

This project is licensed under the MIT License.

