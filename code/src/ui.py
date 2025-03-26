import streamlit as st
from main import EmailProcessor  # Import EmailProcessor from main.py
import os

# Streamlit UI Setup
st.set_page_config(page_title="Bugzappers - Email Classification", layout="wide")

# Display Wells Fargo logo
logo_path = "Wells_Fargo_Logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=150)
else:
    st.warning("Logo not found! Please add 'Wells_Fargo_Logo.png' to the project folder.")

st.title("Email Classification - Bugzappers")

# File uploader
uploaded_file = st.file_uploader("Upload an .eml file", type=["eml"])

def clean_label(value):
    if value and isinstance(value, str):
        return value.split(':**')[-1].strip()
    return value

if uploaded_file is not None:
    with st.spinner("Processing email..."):
        processor = EmailProcessor()  # Create an instance of EmailProcessor
        email_data = processor.extract_email_content(uploaded_file)
        uploaded_file.seek(0)  # Reset file pointer after reading

        if email_data:
            classification_result = processor.call_openrouter_deepseek(email_data['body'])

            # Display extracted email content
            st.subheader("Email Details")
            st.write(f"**Subject:** {email_data['subject']}")
            st.write(f"**From:** {email_data['from']}")
            st.write(f"**To:** {email_data['to']}")
            st.write(f"**Body Preview:** {email_data['body'][:500]}...")  # Show only first 500 chars

            # Display classification results
            st.subheader("Classification Results")
            st.write(f"**Request Type:** {clean_label(classification_result.get('primary_request', 'Unknown'))}")
            st.write(f"**Sub-Request Type:** {clean_label(classification_result.get('sub_request', 'Unknown'))}")
            st.write(f"**Confidence Score:** {classification_result.get('confidence', 'N/A')}")
            st.write(f"**Deal Name:** {clean_label(classification_result.get('Deal_Name', 'Unknown'))}")
            st.write(f"**Loan Amount:** {clean_label(classification_result.get('Loan_Amount', 'Unknown'))}")
            st.write(f"**Expiration Date:** {clean_label(classification_result.get('Expiration_Date', 'Unknown'))}")

            # Check if the email is a duplicate request
            # duplicate_check = processor.check_duplicate_mail_or_new_request(email_data['body'])
            # st.write(f"**Duplicate Check:** {duplicate_check}")

            # Create Service Request button
            if st.button("Create Service Request"):
                st.success("Service request created successfully!")
        else:
            st.error("Failed to extract email content. Please check the uploaded file.")
