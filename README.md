# GPT + Document Intelligence Based OCR Engine

## Overview

The Document Indexer is a Streamlit-based application that allows users to upload PDF files, convert them to images, and extract key data from these images using Azure OpenAI. The application also detects discrepancies in the extracted data and highlights them for validation. For additional accuracy, the application leverages Azure Document Intelligence prebuilt layouts.

## Features

- Upload multiple PDF files.
- Convert PDF pages to images.
- Extract key marked-up fields and checkboxes from images.
- Validate and highlight discrepancies in extracted data.
- Save results and discrepancies to JSON files.
- View previous runs and their results.
- Utilize Azure Document Intelligence prebuilt layouts for enhanced accuracy.

## Requirements

- Python 3.7+
- Streamlit
- OpenAI
- pdf2image
- PIL (Pillow)
- Azure OpenAI credentials
- Azure Document Intelligence endpoint and key
- Azure Blob Storage connection string

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/document-indexer.git
    cd document-indexer
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up your Azure credentials in a [`config.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FRepos%2Fidp_agent%2Fconfig.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Repos\idp_agent\config.py") file:
    ```python
    AZURE_OPENAI_KEY = ""
    AZURE_OPENAI_ENDPOINT = ""
    AZURE_OPENAI_DEPLOYMENT = ""
    AZURE_DOC_INTEL_ENDPOINT = ""
    AZURE_DOC_INTEL_KEY = ""
    AZURE_STORAGE_CONNECTION_STRING = ""
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run idp_agent.py
    ```

2. Upload your PDF files or images using the file uploader.

3. View the extracted data and discrepancies in the main interface and sidebar.

4. Previous runs can be selected from the sidebar to view past results and discrepancies.

## License

This project is licensed under the MIT License. See the LICENSE file for details.