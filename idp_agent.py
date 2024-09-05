import streamlit as st
from openai import AzureOpenAI
import json
import os
import pdf2image
import tempfile
import base64
from config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
from datetime import datetime

# Constants
AZURE_OPENAI_TEMP = 0
AZURE_OPENAI_MAX_TOKENS = 2500
RESULTS_DIR = "results"

# Initialize Azure OpenAI client
client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version="2024-02-01")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ocr_data_from_image_form(image_path: str):
    """Extract key marked up and filled out fields and checkboxes from an image of a form."""
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={"type": "text"},
        messages=[
            {"role": "system", "content": "You are a helpful image analysis and data extraction assistant. You will be tasked with analyzing various images of forms, ID cards, invoices etc. and you will provide a list of key value pairs with the checkbox data, labels and field values, and additional data as applicable. Return a list of key value pairs where each field is in a new line with a logical numbering convention and spacing. Do not return any additional content other than the list of key value pairs - this is a strict requirement with a penalty for violation."},
            {"role": "user", "content": [
                {"type": "text", "text": 
                 """You will perform 2 tasks: \
                    1. Extract key marked up checkboxes, filled out fields, and additional content that have been filled out or marked up in the form. \
                    2. Verify and validate that the returned list is complete and accurate without missing key details, adding any missing information back to the list and correcting any inaccuracies. \

                 Task 1: \
                 Extract key marked up checkboxes, filled out fields, and additional content that have been filled out or marked up in the form. \
                 For checkboxes, return the checkbox label and the status (checked or unchecked). If you are not sure if a checkbox is checked or unchecked, return undetermined. \
                 For filled out fields, return the field label and the filled out content. \
                 For additional content, return the content and the context in which it appears. \
                 Ensure that the the returned list is complete and concise without missing key details. \
                
                 Task 2: \
                 Verify and validate that the returned list is complete and concise without missing key details or key fields/checkboxes in the form, adding any missing information back to the list. \
                 Be very diligent as there is a financial penalty for missing fields/checkboxes or inaccurate values. In some cases, the data may be laid out in horizontal columns as well as vertical rows, and needs to included in both cases. \
                
                 Once both tasks are complete, return a list of key value pairs with a logical numbering convention and spacing. Do not return any additional details other than the extracted key value pairs, as you will be penalized for doing so."""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        temperature=AZURE_OPENAI_TEMP,
        max_tokens=AZURE_OPENAI_MAX_TOKENS
    )

    return response.choices[0].message.content

def split_pdf_to_images(pdf_path):
    """Split PDF into images."""
    images = pdf2image.convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        temp_image_path = os.path.join(tempfile.gettempdir(), f"page_{i}.png")
        image.save(temp_image_path, 'PNG')
        image_paths.append(temp_image_path)
    return image_paths

def generate_temp_url(file_path):
    """Generate a temporary URL for the PDF file."""
    return f"file://{file_path}"

def main():
    st.title("Document Indexer")

    # List previous runs in the sidebar
    previous_runs = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    selected_run = st.sidebar.selectbox("Select a previous run", [""] + previous_runs)

    uploaded_files = st.file_uploader("Upload your documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        results_dict = {}
        for uploaded_file in uploaded_files:
            expander = st.expander(f"Document: {uploaded_file.name}")
            with expander:
                status = st.text("Status: In Progress")
                results = []  # Reset results for new run
                # Save the uploaded file to a temporary location
                pdf_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize session state for PDF viewer and results
                if f"pdf_viewer_{uploaded_file.name}" not in st.session_state:
                    st.session_state[f"pdf_viewer_{uploaded_file.name}"] = False
                if f"results_{uploaded_file.name}" not in st.session_state:
                    st.session_state[f"results_{uploaded_file.name}"] = None
                
                # Generate a temporary URL for the PDF file
                pdf_url = generate_temp_url(pdf_path)
                
                # Create a link to open the PDF in a new tab
                st.markdown(f'<a href="{pdf_url}" target="blah">Open PDF in New Tab</a>', unsafe_allow_html=True)

                # Check if results are already in session state
                if st.session_state[f"results_{uploaded_file.name}"] is None:
                    # Split the PDF into images
                    image_paths = split_pdf_to_images(pdf_path)
                    
                    for page_number, image_path in enumerate(image_paths, start=1):
                        # Call the OCR function for each image
                        result = ocr_data_from_image_form(image_path)
                        if result:
                            results.append(result)
                            st.subheader(f"Page {page_number}")
                            st.write(result)
                    
                    if results:
                        status.text("Status: Completed")
                        # Save results to session state
                        st.session_state[f"results_{uploaded_file.name}"] = results
                        results_dict[uploaded_file.name] = results
                    else:
                        status.text("Status: Failed")
                else:
                    # Display results from session state
                    results = st.session_state[f"results_{uploaded_file.name}"]
                    results_dict[uploaded_file.name] = results
                    for page_number, result in enumerate(results, start=1):
                        st.subheader(f"Page {page_number}")
                        st.write(result)
        
        # Save results to a JSON file with a timestamp
        if results_dict:
            run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(os.path.join(RESULTS_DIR, f"run_{run_date}.json"), "w") as f:
                json.dump(results_dict, f)

    elif selected_run:
        st.sidebar.write(f"Displaying results for: {selected_run}")
        with open(os.path.join(RESULTS_DIR, selected_run), "r") as f:
            old_results = json.load(f)
        for document_name, results in old_results.items():
            st.subheader(f"Document: {document_name}")
            for page_number, result in enumerate(results, start=1):
                st.subheader(f"Page {page_number}")
                st.write(result)

if __name__ == "__main__":
    main()