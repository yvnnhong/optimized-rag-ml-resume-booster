#pdf_extractor.py is step 1 
import fitz #to handle opening PDF files and extracting text from them (aka PyMUPDF)
import logging 
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self): 
        self.supported_formats = ['.pdf']

    #extract_text_disk is for disk; does NOT need gradio interface to test
    #local development testing aka command line testing 
    def extract_text_disk(self, pdf_path: str) -> Dict[str, Any]: 
        try:
            if not Path(pdf_path).exists(): 
                logger.error(f"File not found: {pdf_path}")
                return {
                    'full_text': '',
                    'page_texts': [],
                    'page_count': 0,
                    'char_count': 0,
                    'file_name': 'unknown',
                    'extraction_status': 'failed',
                    'error': 'File not found'
                }
            if not pdf_path.lower().endswith('.pdf'):
                logger.error(f"This is unsupported file format: {pdf_path}")
                return {
                    'full_text': '',
                    'page_texts': [],
                    'page_count': 0,
                    'char_count': 0,
                    'file_name': Path(pdf_path).name,
                    'extraction_status': 'failed',
                    'error': 'Unsupported file format'
                }
            doc = fitz.open(pdf_path)
            #initialize variables for text extraction
            full_text = ""
            page_texts = [] 
            #^for entry level, resumes are usually only 1 page -- relevant for knowledge_base
            for page_num in range(doc.page_count): 
                page = doc[page_num]
                page_text = page.get_text()
                page_texts.append(page_text)
                full_text += page_text + "\n"
            #Close the document 
            doc.close()
            result = {
                'full_text': full_text.strip(), 
                #.strip() removes leading and trailing whitespace from text 
                'page_texts': page_texts,
                'page_count': len(page_texts),
                'char_count': len(full_text),
                'file_name': Path(pdf_path).name,
                'extraction_status': 'success'
            }
            logger.info(f"Successfully extracted text from {pdf_path}")
            logger.info(f"Pages: {result['page_count']}, Characters: {result['char_count']}")
            return result 
        
        except Exception as e: 
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            if pdf_path is not None: 
                file_name = Path(pdf_path).name
            else:
                file_name = 'unknown'
            #return a dict with empty stats to prevent crashing 
            return {
                'full_text': '',
                'page_texts': [],
                'page_count': 0,
                'char_count': 0,
                'file_name': file_name,
                'extraction_status': 'failed',
                'error': str(e)
            }
        
    #extract_from_bytes is for gradio testing (web upload scenario, aka other users)
    def extract_from_bytes(self, pdf_bytes: bytes, filename: str = "upload.pdf") -> Dict[str, Any]: 
        try: 
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            page_texts = []
            for page_num in range(doc.page_count): 
                page = doc[page_num]
                page_text = page.get_text()
                page_texts.append(page_text)
                full_text += page_text + "\n"
            doc.close()
            result = {
                'full_text': full_text.strip(),
                'page_texts': page_texts,
                'page_count': len(page_texts),
                'char_count': len(full_text),
                'file_name': filename,
                'extraction_status': 'success'
            }
            logger.info(f"Successfully extracted text from uploaded file: {filename}")
            logger.info(f"Pages: {result['page_count']}, Characters: {result['char_count']}")
            return result 
        except Exception as e: 
            logger.error(f"Error extracting text from uploaded file {filename}: {str(e)}")
            return {
                'full_text': '',
                'page_texts': [],
                'page_count': 0,
                'char_count': 0,
                'file_name': filename,
                'extraction_status': 'failed',
                'error': str(e)
            }
    