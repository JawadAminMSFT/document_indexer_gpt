from openai import AzureOpenAI
import json
import os
import base64
from config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
from doc_intel import analyze_document
import base64


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def classify(image_path:str) -> str:
    # Constants
    AZURE_OPENAI_TEMP = 0
    AZURE_OPENAI_MAX_TOKENS = 2500

    # Initialize Azure OpenAI client
    client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version="2024-02-01")

    base64_image = encode_image(image_path)

    print(image_path)

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": """You are a helpful image analysis and data extraction and document classifier assistant who can check the provided image and classify it into one of the followings:
             NB-APPFLROM: Application form
             NB-NBID: ID card/ Birth certificate
             NB-NBDOC: Agent's evaluation form.
             NB-SUPPDOC: Sale illustration
             NB-CLMEDOC: Customer’s medical document
             NB-FNA: Financial Needs Analysis
             NB-RPQ: Risk profile questionnaire
             NB-EDISCLOSURE: Health information declaration form
             NB-CONFIMG: Consent form
             NB-TIADOC: Temporary insurance
             """},
            {"role": "user", "content": [
                {"type": "text", "text": 
                 """classify the following document \
                 Once both tasks are complete, return a JSON object like this {"documentType": "document class", "reason":"explain the reason you classify the document this way"}. If you can't classify it just provide :unknown: as the documentType ."""},
                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        temperature=AZURE_OPENAI_TEMP,
        max_tokens=AZURE_OPENAI_MAX_TOKENS
    )

    return response.choices[0].message.content