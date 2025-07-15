import pdfplumber
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import os

def extract_text_from_digital_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
    return text.strip()

def extract_text_from_scanned_pdf(pdf_path, lang='eng+hin', poppler_path=None):
    if poppler_path is None:
        poppler_path = os.environ.get("POPLER_PATH", r"C:\poppler-24.08.0\Library\bin")
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    text = ''
    for img in images:
        text += pytesseract.image_to_string(img, lang=lang) + '\n'  
    return text.strip()          

def extract_text(pdf_path):
    text = extract_text_from_digital_pdf(pdf_path)
    if len(text.strip()) < 100:
        print("Detected scanned PDF. Using OCR...")
        text = extract_text_from_scanned_pdf(pdf_path)
    return text

    
# Note: The above code will save the first 4 pages of the PDF as JPEG images

# Logic explanation of the code:

# 1. The code aims to extract text from PDF files, handling two types:
#    - Digital PDFs: PDFs with selectable/searchable text.
#    - Scanned PDFs: PDFs that are images of text, requiring OCR.

# 2. extract_text_from_digital_pdf(pdf_path):
#    - Opens the PDF using pdfplumber.
#    - Extracts text from the first 4 pages.
#    - Concatenates and returns all extracted text.

# 3. extract_text_from_scanned_pdf(pdf_path, lang='eng+hin', poppler_path=None):
#    - Converts the first 4 pages of the PDF into images using pdf2image.
#    - Performs OCR on each image using pytesseract with support for English and Hindi.
#    - Concatenates and returns the OCR-extracted text.

# 4. extract_text(pdf_path):
#    - First attempts digital extraction.
#    - If extracted text is too short (likely scanned PDF), switches to OCR extraction.
#    - Returns the extracted text.

# Notes:
# - The first 4 pages are processed for performance reasons.
# - Tesseract OCR executable path must be correctly set.
# - Poppler binaries are needed for PDF to image conversion.: 
                     # Poppler is the tool doing the conversion.
                     # pdf2image wraps Poppler to provide Python-friendly access to that functionality.


