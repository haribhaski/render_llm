import requests
import os
import fitz  # PyMuPDF for PDF parsing
from docx import Document
import email
from email import policy
from email.parser import BytesParser
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str) -> str:
    """
    Download a file from a URL and save it locally with a unique filename.
    
    Args:
        url: URL of the file to download.
        
    Returns:
        Path to the downloaded file.
        
    Raises:
        HTTPException: If the download fails or URL is invalid.
    """
    try:
        # Validate URL
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")
        
        # Generate a unique temporary filename
        from urllib.parse import urlparse
        filename = f"temp_{hash(url)}_{os.path.basename(urlparse(url).path)}"
        temp_path = os.path.join("temp_downloads", filename)
        os.makedirs("temp_downloads", exist_ok=True)
        
        # Download the file
        logger.info(f"Downloading file from {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"File downloaded to {temp_path}")
        return temp_path
    
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise Exception(f"Download failed: {str(e)}")


def extract_text_from_eml_file(file_path: str) -> str:
    import email
    from email import policy
    from email.parser import BytesParser
    import logging

    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Extracting text from EML: {file_path}")
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        if msg.is_multipart():
            parts = [part for part in msg.walk() if part.get_content_type() == "text/plain"]
            text_parts = []
            for part in parts:
                payload = part.get_payload(decode=True)
                if payload:
                    text_parts.append(payload.decode(errors="ignore"))
            text = "\n".join(text_parts)
        else:
            payload = msg.get_payload(decode=True)
            text = payload.decode(errors="ignore") if payload else ""
        if not text.strip():
            logger.warning(f"No text extracted from EML: {file_path}")
        return text.strip()
    except Exception as e:
        logger.error(f"EML extraction failed for {file_path}: {str(e)}")
        raise Exception(f"EML extraction failed: {str(e)}")


def extract_text_from_pdf_url(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        Extracted text as a string.
        
    Raises:
        Exception: If PDF parsing fails.
    """
    try:
        logger.info(f"Extracting text from PDF: {file_path}")
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
        return text.strip()
    
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
        raise Exception(f"PDF extraction failed: {str(e)}")

def extract_text_from_docx_file(file_path: str) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file.
        
    Returns:
        Extracted text as a string.
        
    Raises:
        Exception: If DOCX parsing fails.
    """
    try:
        logger.info(f"Extracting text from DOCX: {file_path}")
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
        return text.strip()
    
    except Exception as e:
        logger.error(f"DOCX extraction failed for {file_path}: {str(e)}")
        raise Exception(f"DOCX extraction failed: {str(e)}")



