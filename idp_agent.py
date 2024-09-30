import streamlit as st
from openai import AzureOpenAI
import json
import os
import pdf2image
import tempfile
import base64
from datetime import datetime
from config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
from doc_intel import analyze_document
from doc_classifier import classify

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

def ocr_data_from_image_form(image_path: str, doc_intel_result: str):
    """Extract key marked up and filled out fields and checkboxes from an image of a form."""
    base64_image = encode_image(image_path)
    print(f"Document Intelligence Markdown output: {doc_intel_result}")
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful image analysis and data extraction assistant. You will be tasked with analyzing various images of forms, ID cards, invoices etc. and you will provide an array of key value pairs with the checkbox data, labels and field values, and additional data as applicable. Return a JSON array of key value pairs. Do not return any additional content other than the JSON array - this is a strict requirement with a penalty for violation."},
            {"role": "user", "content": [
                {"type": "text", "text": 
                 """You will perform 2 tasks: \
                    1. Extract key marked up checkboxes, filled out fields, and additional content that have been filled out or marked up in the form. \
                    2. Verify and validate that the returned list is complete and accurate without missing key details, adding any missing information back to the list and correcting any inaccuracies. You will also check against the provided markdown data to correct any mistakes. \

                 Task 1: \
                 Extract key marked up checkboxes, filled out fields, and additional content that have been filled out or marked up in the form. \
                 For checkboxes, return the checkbox label and the status (checked or unchecked). If you are not sure if a checkbox is checked or unchecked, return undetermined. \
                 For filled out fields, return the field label and the filled out content. \
                 For additional content, return the content and the context in which it appears. \
                 Ensure that the the returned list is complete and concise without missing key details. \
                
                 Task 2: \
                 Verify and validate that the returned list is complete and concise without missing key details or key fields/checkboxes in the form, adding any missing information back to the list. \
                 Be very diligent as there is a financial penalty for missing fields/checkboxes or inaccurate values. In some cases, the data may be laid out in horizontal columns as well as vertical rows, and needs to included in both cases. \
                 Pay specific attention to include fields that include personal identification information, dates, addresses, numeric values, and names. \
                 You will have to cross-reference the extracted data with the provided markdown data from another OCR tool to ensure accuracy. \
                 
                 ---------------------------------------- \
                 MARKDOWN DATA: \
                 f{doc_intel_result} \
                 MARKDOWN DATA END \
                
                 Once both tasks are complete, return a JSON array containing the final key value pairs. Do not return any additional details other than the extracted key value pairs as a JSON array, as you will be penalized for doing so. If an image is not as expected, return an empty array."""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        temperature=AZURE_OPENAI_TEMP,
        max_tokens=AZURE_OPENAI_MAX_TOKENS
    )

    return response.choices[0].message.content

def detect_discrepancies(results_dict):
    """Detect discrepancies in extracted data."""
    all_results = []
    for doc_name, results in results_dict.items():
        all_results.extend(results)
    
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant that validates the consistency in extracted data from multiple documents. You will compare the extracted data and highlight any discrepancies. You will return a JSON object listing the most important key value pairs (with one value for each document as applicable, returned as an array of values under the key) with an additional field marked 'type' with value as either 'discrepancy' or 'consistent' to indicate if the data is consistent or not. Do not return any additional content other than the list of key value pairs - this is a strict requirement with a penalty for violation."},
            {"role": "user", "content": f"Compare the following extracted data and highlight discrepancies as a JSON object highlighting key value pairs and discrepancies:\n\n{json.dumps(all_results)}. Only compare fields between two documents, not within the same document. The results JSON is structured to have each document as one JSON object with the key value pairs within an array, and therefore you should compare fields across multiple document objects, and not within the same array. If any of the documents is empty or invalid, use the keys from the other document(s) and mark them as discrepancies. Double check the list and only include the most important key value pairs, preferably no more than 10-15 pairs."}
        ],
        temperature=AZURE_OPENAI_TEMP,
        max_tokens=AZURE_OPENAI_MAX_TOKENS
    )

    discrepancies_json = response.choices[0].message.content
    discrepancies_dict = json.loads(discrepancies_json)  # Parse JSON string into dictionary
    return discrepancies_dict

def highlight_discrepancies(discrepancies_dict):
    """Highlight discrepancies with Markdown, handling None values and making them line by line."""
    highlighted_discrepancies = []
    for key, item in discrepancies_dict.items():
        color = "🔴" if item["type"] == "discrepancy" else "🟢"
        # Replace None values with "N/A"
        values = [str(value) if value is not None else "N/A" for value in item["values"]]
        values_str = " vs ".join(values)
        # Add each discrepancy in a new line
        highlighted_discrepancies.append(f"{color} **{key}**: {values_str}\n")
    return "\n".join(highlighted_discrepancies)



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
    st.markdown("<h1 style='text-align: center;'>GPT + DOC Intel OCR Engine</h1>", unsafe_allow_html=True)

    # Move the previous runs dropdown to the sidebar
    previous_runs = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    selected_run = st.sidebar.selectbox("Select a previous run", [""] + previous_runs)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader("Upload your PDF files here", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        results_dict = {}
        for uploaded_file in uploaded_files:
            expander = st.expander(f"Document: {uploaded_file.name}", expanded=True)
            with expander:
                status = st.text("Status: In Progress")
                results = []

                pdf_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pdf_url = generate_temp_url(pdf_path)
                st.markdown(f'<a href="{pdf_url}" target="blah">Open PDF in New Tab</a>', unsafe_allow_html=True)

                image_paths = split_pdf_to_images(pdf_path)
                progress_bar = st.progress(0)

                for page_number, image_path in enumerate(image_paths, start=1):
                    doc_intel_result = analyze_document(image_path)
                    doc_classification = classify(image_path)
                    print(doc_classification)
                    result = ocr_data_from_image_form(image_path, doc_intel_result)
                    if result:
                        results.append(result)
                        st.subheader(f"Page {page_number}")
                        st.image(image_path, caption=f"Page {page_number} Preview", use_column_width=True)
                        st.json(result)

                    progress_bar.progress(page_number / len(image_paths))

                if results:
                    status.text("Status: Completed")
                    results_dict[uploaded_file.name] = results
                else:
                    status.text("Status: Failed")

        discrepancies_dict = detect_discrepancies(results_dict)
        highlighted_discrepancies = highlight_discrepancies(discrepancies_dict)

        # Display discrepancies in the sidebar
        st.sidebar.title("Validation")
        if highlighted_discrepancies:
            st.sidebar.markdown(highlighted_discrepancies)

        # Save results and discrepancies to a JSON file with a timestamp
        if results_dict:
            run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(os.path.join(RESULTS_DIR, f"run_{run_date}.json"), "w") as f:
                json.dump({"results": results_dict, "discrepancies": discrepancies_dict}, f)

    # Handle displaying previous runs
    elif selected_run:
        st.sidebar.write(f"Displaying results for: {selected_run}")
        with open(os.path.join(RESULTS_DIR, selected_run), "r") as f:
            old_data = json.load(f)
        old_results = old_data.get("results", {})
        old_discrepancies = old_data.get("discrepancies", {})

        for document_name, results in old_results.items():
            st.subheader(f"Document: {document_name}")
            for page_number, result in enumerate(results, start=1):
                st.subheader(f"Page {page_number}")
                st.json(result)

        # Display discrepancies for previous runs
        if old_discrepancies:
            highlighted_discrepancies = highlight_discrepancies(old_discrepancies)
            st.sidebar.title("Discrepancies")
            st.sidebar.markdown(highlighted_discrepancies)

if __name__ == "__main__":
    main()
