import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat, AnalyzeResult
from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_DOC_INTEL_ENDPOINT, AZURE_DOC_INTEL_KEY

def upload_file_to_blob(container_name, local_file_path):
    connect_str = AZURE_STORAGE_CONNECTION_STRING
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    local_file_name = os.path.basename(local_file_path)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)
    
    with open(local_file_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
    return blob_client.url

def analyze_document(local_file_path):
    formUrl = upload_file_to_blob("sampleapp", local_file_path)
    if not formUrl:
        return "Failed to upload file to blob storage."

    document_intelligence_client = DocumentIntelligenceClient(endpoint=AZURE_DOC_INTEL_ENDPOINT, credential=AzureKeyCredential(AZURE_DOC_INTEL_KEY))
    try:
        print(formUrl)
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(url_source=formUrl),
            output_content_format=ContentFormat.MARKDOWN,
        )
        result = poller.result()
    except Exception as e:
        return f"Failed to analyze document: {e}"

    markdown_lines = []
    if result.styles is None:
        # Handle the None case, e.g., log an error or return an empty list
        print("Result.styles is None")
        #return markdown_lines
    else:
        for idx, style in enumerate(result.styles):
            markdown_lines.append(
                f"Document contains {'handwritten' if style.is_handwritten else 'no handwritten'} content"
            )

    for page in result.pages:
        for line_idx, line in enumerate(page.lines):
            markdown_lines.append(
                f"...Line # {line_idx} has text content '{line.content}'"
            )
        if page.selection_marks is not None:
            for selection_mark in page.selection_marks:
                markdown_lines.append(
                    f"...Selection mark is '{selection_mark.state}' and has a confidence of {selection_mark.confidence}"
                )
    if result.tables is not None:
        for table_idx, table in enumerate(result.tables):
            markdown_lines.append(
                f"Table # {table_idx} has {table.row_count} rows and {table.column_count} columns"
            )

            for cell in table.cells:
                markdown_lines.append(
                    f"...Cell[{cell.row_index}][{cell.column_index}] has content '{cell.content}'"
                )

    markdown_lines.append("----------------------------------------")
    return "\n".join(markdown_lines)

# Example usage
markdown_result = analyze_document(r"C:\Users\jawadamin\Downloads\appl_form_image.png")
print(markdown_result)