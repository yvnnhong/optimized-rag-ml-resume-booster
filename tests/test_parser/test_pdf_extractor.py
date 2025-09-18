#This file, test_pdf_extractor.py, tests pdf_extractor.py.
#To run this file, ensure you are in the project root 
#(ensure you are using powershell for windows): 
#cd C:\Users\yvonn\rag-nlp-resume-booster
#python tests/test_parser/test_pdf_extractor.py

import logging
import sys
sys.path.append('.')
from src.parser.pdf_extractor import PDFExtractor

def test_single_resume_file_disk(extractor, file_path, test_name):
    print(f"\nTesting {test_name}")
    result = extractor.extract_text_disk(file_path)
    if result and result['extraction_status'] == 'success': 
        print(f"Extraction is successful.")
        print(f"File name: {result['file_name']}")
        print(f"Total number of Pages: {result['page_count']}")
        print(f"Total number of characters: {result['char_count']}")
        print(f"First 150 chars: {result['full_text'][:150]}...")
        return True 
    else: 
        if result and 'error' in result:
            error_msg = result['error']
        else: 
            error_msg = 'Unknown error'
        print(f"(Disk) Extraction failed: {error_msg}")
        return False
    
def test_single_resume_file_bytes(extractor, file_path, test_name):
    print(f"\nTesting {test_name} using the bytes method.")
    try:
        with open(file_path, 'rb') as file: #open pdf in binary read mode ('rb')
            pdf_bytes = file.read()
        result = extractor.extract_from_bytes(pdf_bytes, filename=file_path.split('/')[-1])
        if result and result['extraction_status'] == 'success': 
            print(f"Bytes extraction is successful.")
            print(f"File name: {result['file_name']}")
            print(f"Total number of Pages: {result['page_count']}")
            print(f"Total number of characters: {result['char_count']}")
            print(f"First 150 chars: {result['full_text'][:150]}...")
            return True
        else: 
            if result and 'error' in result: 
                error_msg = result['error']
            else: 
                error_msg = 'Unknown error'
            print(f"Bytes Extraction failed: {error_msg}")
            return False
    except FileNotFoundError: 
        print(f"File not found: {file_path}")
        return False
    
def test_all_resumes():
    extractor = PDFExtractor()
    test_files = [
        ("data/sample_resumes/standard_1pg_resume.pdf", "Standard 1-Page Resume"),
        ("data/sample_resumes/long_resume_6pgs.pdf", "Long 6-Page Resume"),
        ("data/sample_resumes/sparse_resume.pdf", "Sparse 1-Page Resume")
    ]
    disk_results = []
    bytes_results = []
    print("=" * 60)
    print("TESTING PDF EXTRACTION - DISK METHOD")
    print("=" * 60)
    #Test the disk method for all files 
    for file_path, test_name in test_files: 
        success = test_single_resume_file_disk(extractor, file_path, test_name)
        disk_results.append((test_name, success))

    print("\n")
    print("=" * 60)
    print("TESTING PDF EXTRACTION - BYTES METHOD")
    print("=" * 60)

    #test the bytes method for all files 
    for file_path, test_name in test_files: 
        success = test_single_resume_file_bytes(extractor, file_path, test_name)
        bytes_results.append((test_name, success))

    #summary! 
    print("\n")
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print("\nDisk Method Results:")
    for test_name, success in disk_results: 
        if success: 
            status = "Pass"
        else: 
            status = "Fail"
        print(f"{test_name}: {status}")

    print("\nBytes Method Results: ")
    for test_name, success in bytes_results: 
        if success: 
            status = "Pass"
        else: 
            status = "Fail"
        print(f"{test_name}: {status}")

    #compile overall stats!!
    disk_passed = 0
    for test_name, success in disk_results: 
        if success: 
            disk_passed += 1

    bytes_passed = 0
    for test_name, success in bytes_results: 
        if success: 
            bytes_passed += 1 

    #todo: find a way to clearly print out all final results (if necessary?)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_all_resumes()
    #name = name of module (e.x. src.parser.pdf_extractor)
    #levelname = severity level of warning:
    #DEBUG (lowest priority)
    #INFO
    #WARNING
    #ERROR
    #CRITICAL (highest priority))