import streamlit as st
import os
from main import EmailProcessor  # Import EmailProcessor from main.py

st.set_page_config(page_title="Bugzappers - Email Classification", layout="wide")

page = st.sidebar.radio("Navigation", ["Email Classification", "Service Request Form"])

if page == "Email Classification":
    
    logo_path = "Wells_Fargo_Logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.warning("Logo not found! Please add 'Wells_Fargo_Logo.png' to the project folder.")

    st.title("Email Classification - Bugzappers")
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

                st.session_state['classification_result'] = classification_result

                # Display extracted email content
                st.subheader("Email Details")
                st.write(f"**Subject:** {email_data['subject']}")
                st.write(f"**From:** {email_data['from']}")
                st.write(f"**To:** {email_data['to']}")
                st.write(f"**Body Preview:** {email_data['body'][:500]}...") 
                # Display classification results
                st.subheader("Classification Results")
                st.write(f"**Request Type:** {clean_label(classification_result.get('primary_request', 'Unknown'))}")
                st.write(f"**Sub-Request Type:** {clean_label(classification_result.get('sub_request', 'Unknown'))}")
                st.write(f"**Confidence Score:** {classification_result.get('confidence', 'N/A')}")
                st.write(f"**Deal Name:** {clean_label(classification_result.get('Deal_Name', 'Unknown'))}")
                st.write(f"**Loan Amount:** {clean_label(classification_result.get('Loan_Amount', 'Unknown'))}")
                st.write(f"**Expiration Date:** {clean_label(classification_result.get('Expiration_Date', 'Unknown'))}")

                if st.button("Create Service Request"):
                    st.session_state['service_request'] = True
                    st.experimental_rerun()

elif page == "Service Request Form":
    classification_result = st.session_state.get('classification_result', {})
    
    logo_path = "Wells_Fargo_Logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.warning("Logo not found! Please add 'Wells_Fargo_Logo.png' to the project folder.")
    st.title("Service Request Form")

    request_type = st.text_input("Request Type", classification_result.get('primary_request', ""))
    sub_request_type = st.text_input("Sub-Request Type", classification_result.get('sub_request', ""))
    deal_name = st.text_input("Deal Name", classification_result.get('Deal_Name', ""))
    loan_amount = st.text_input("Loan Amount", classification_result.get('Loan_Amount', ""))
    expiration_date = st.text_input("Expiration Date", classification_result.get('Expiration_Date', ""))

    if st.button("Submit Service Request"):
        st.success("Service request submitted successfully!")
        st.experimental_rerun()
