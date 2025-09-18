#test_section_parser.py
import logging
from typing import Dict, Any
import sys
sys.path.append('.')
from src.parser.pdf_extractor import PDFExtractor
from src.parser.text_processor import TextProcessor
from src.parser.section_parser import SectionParser

def test_section_parsing_full_pipeline(): 
    """Test section parser using the full pipeline: 
    PDF Extraction -> Text Processing -> Section Parsing"""
    print("=" * 70)
    print("Testing Section Parser -- Full Pipeline")
    print("=" * 70)
    #initialize all processors.
    pdf_extractor = PDFExtractor()
    text_processor = TextProcessor()
    section_parser = SectionParser()

    test_files = [
        ("data/sample_resumes/standard_1pg_resume.pdf", "Standard 1-Page Resume"),
        ("data/sample_resumes/long_resume_6pgs.pdf", "Long 6-Page Resume"),
        ("data/sample_resumes/sparse_resume.pdf", "Sparse 1-Page Resume")
    ]

    for file_path, test_name in test_files: 
        print(f"\n--- Testing {test_name} ---")
        try: 
            #Step 1: extract text from pdf 
            pdf_result = pdf_extractor.extract_text_disk(file_path)
            if not pdf_result or pdf_result['extraction_status'] != 'success':
                print(f"Failed to extract text from {file_path}")
                continue 
            #Step 2: Process the extracted text
            processing_result = text_processor.process_text(pdf_result['full_text'])
            if processing_result['processing_status'] != 'success': 
                print(f"Failed to process text: {processing_result.get('error', 'Unknown error')}")
                continue
            #Step 3: parse sections from processed text 
            section_result: Dict[str, Any] = section_parser.parse_sections(processing_result['cleaned_text'])
            if section_result['parsing_status'] == 'success': 
                print(f"Section parsing successful.")
                print(f"Sections found: {section_result['total_sections']}")
                #show corresponding confidence scores.
                #note: section_metadata is a ResumeSection object 
                for section_name, section_metadata in section_result['sections'].items():
                    print(f"{section_name.title()}: {len(section_metadata.content)} "
                          f"total chars (confidence: {section_metadata.confidence:.2f})")
                #Show contact info
                contact = section_result['contact_info']
                found_contact = []
                for k, v in contact.items(): 
                    if v: 
                        found_contact.append(f"{k}: {v}")
                if found_contact: 
                    print(f"Contact info: {', '.join(found_contact)}")
                else:
                    print("No contact info found.")
                #Show preview of each section: 
                print(f"\nSection content previews: ")
                for section_name, section_metadata in section_result['sections'].items(): 
                    preview = section_metadata.content[:100].replace('\n', ' ').strip()
                    if preview: 
                        print(f"{section_name}: {preview}...")
                    else:
                        print(f"{section_name}: [No content]")
            else: 
                print(f"Section parsing failed: {section_result.get('error', 'Unknown error')}")
        except Exception as e: 
            print(f"Pipeline test failed with error: {str(e)}")

def test_section_parsing_with_sample_text(): 
    """Test section parsing with known sample resume text."""
    print("\n" + "=" * 70)
    print("Testing section parsing with sample text: ")
    print("=" * 70)
    sample_resume_text = """
    Marcus Rodriguez
    Software Engineer
    marcus.rodriguez@email.com | (555) 123-4567 | linkedin.com/in/marcus-rodriguez | github.com/marcus-dev
    
    PROFESSIONAL SUMMARY
    Experienced software engineer with 5+ years developing scalable web applications and machine learning systems.
    Passionate about clean code and innovative solutions.
    
    EXPERIENCE
    Senior Software Engineer - TechCorp Inc (2020-Present)
    - Built scalable web applications using Python, React, and AWS
    - Led team of 4 engineers on major product features
    - Improved system performance by 40% through optimization
    
    Software Engineer - StartupXYZ (2018-2020)
    - Developed REST APIs and microservices architecture
    - Implemented CI/CD pipelines reducing deployment time by 60%
    - Collaborated with cross-functional teams on product development
    
    EDUCATION
    Bachelor of Science in Computer Science
    Massachusetts Institute of Technology (2014-2018)
    GPA: 3.8/4.0
    Relevant Coursework: Data Structures, Algorithms, Machine Learning, Databases
    
    SKILLS
    Programming Languages: Python, JavaScript, Java, C++, SQL
    Frameworks & Tools: React, Django, Flask, Node.js, Docker, Kubernetes
    Databases: PostgreSQL, MongoDB, Redis
    Cloud Platforms: AWS, Google Cloud Platform, Azure
    
    PROJECTS
    Resume Analyzer Tool (2024)
    - Built ML-powered resume analysis system using NLP and RAG architecture
    - Deployed on AWS with Docker containerization and CI/CD pipeline
    - Technologies: Python, PyTorch, FastAPI, ChromaDB
    
    E-commerce Platform (2023)
    - Developed full-stack e-commerce application with payment integration
    - Implemented real-time inventory management and analytics dashboard
    - Technologies: React, Node.js, PostgreSQL, Stripe API
    
    CERTIFICATIONS
    AWS Certified Solutions Architect - Associate (2023)
    Google Cloud Professional Developer (2022)
    Certified Kubernetes Administrator (2021)
    """
    parser = SectionParser()
    result = parser.parse_sections(sample_resume_text)
    if result['parsing_status'] == 'success': 
        print(f"Sample text parsing successful.")
        print(f"Total number of sections found: {result['total_sections']}")
        #Show detailed section analysis
        print(f"\nDetailed section analysis: ")
        #note: section_metadata is a ResumeSection object. 
        for section_name, section_metadata in result['sections'].items(): 
            print(f"\n{section_name.upper()}:")
            print(f"Confidence: {section_metadata.confidence:.2f}")
            print(f"Length: {len(section_metadata.content)} characters long")
            print(f"Start index: {section_metadata.start_index}-{section_metadata.end_index}")
            #Show first few lines of content: 
            lines = section_metadata.content.split('\n')[:3]
            for line in lines: 
                if line.strip():
                    print(f"{line.strip()}")
        #show contact info extraction: 
        print(f"\nContact info extracted: ")
        for k, v in result['contact_info'].items(): 
            if v: 
                print(f"{k.title()}: {v}")
    else: 
        print(f"Sample text parsing failed: {result.get('error', 'Unknown error')}")

def test_edge_cases(): 
    """Test edge cases and error handling for section parsing."""
    print("\n" + "=" * 70)
    print("Testing edge cases and error handling for section parsing.")
    print("=" * 70)
    parser = SectionParser()
    #1) Test empty text 
    print("\n 1) Testing empty text ")
    result = parser.parse_sections("")
    print(f"Empty text result: {result['parsing_status']}")
    #2) Test text with no sections 
    print("\n 2) Testing text with no recognizable sections ")
    no_sections_text = ("This is just some random text with no resume" 
    "sections at all. Just words and sentences.")
    result = parser.parse_sections(no_sections_text)
    print(f"No sections result: {result['parsing_status']}, " 
          f"number of sections found: {result['total_sections']}")
    #Test None input 
    print("\n 3) Testing None input: ")
    result = parser.parse_sections(None)
    print(f"None input result: {result['parsing_status']}")
    #Testing text with only contact info 
    print("\n 4) Testing text with only contact info: ")
    contact_only_text = ("John Doe\njohn.doe@email.com\n(555)" 
    "123-4567\nlinkedin.com/in/johndoe")
    result = parser.parse_sections(contact_only_text)
    if result['parsing_status'] == 'success': 
        contact_fields_found = []
        for k, v in result['contact_info'].items(): 
            if v: 
                contact_fields_found.append(k)
        print(f"Contact-only text: {len(contact_fields_found)} contact fields found, "
              f"{result['total_sections']} total sections found")
    #Test text with duplicate section headers
    print("\n 5) Testing text with duplicate section headers " )
    duplicate_text = """
    EXPERIENCE
    First experience section
    
    Some other content here
    
    EXPERIENCE
    Second experience section (duplicate)
    
    SKILLS
    My skills here
    """
    result = parser.parse_sections(duplicate_text)
    if result['parsing_status'] == 'success': 
        print(f"Duplicate headers: {result['total_sections']} unique sections found")
        if 'experience' in result['sections']: 
            exp_section = result['sections']['experience']
            print(f"Experience section confidence: {exp_section.confidence:.2f}")

def test_specific_patterns(): 
    """Test specific header pattern matching"""
    print("\n" + "=" * 70)
    print("6) Testing specific section header patterns")
    print("=" * 70)
    parser = SectionParser()
    # Test various header formats
    test_cases = [
        ("EXPERIENCE", "Standard uppercase"),
        ("Experience", "Title case"),
        ("experience", "Lowercase"),
        ("Work Experience", "Multi-word"),
        ("Professional Experience", "Extended form"),
        ("TECHNICAL SKILLS", "Technical skills variant"),
        ("Skills and Abilities", "Skills variant"),
        ("Education", "Simple education"),
        ("Educational Background", "Extended education"),
        ("Projects", "Simple projects"),
        ("Key Projects", "Extended projects"),
        ("Certifications", "Simple certifications"),
        ("Professional Certifications", "Extended certifications")
    ]
    for header, header_description in test_cases: 
        test_text = f"""
        Some intro text here
        
        {header}
        Content under this header goes here
        More content on next line
        
        SKILLS
        Some skills listed here
        """
        result = parser.parse_sections(test_text)
        if result['parsing_status'] == 'success' and result['total_sections'] >= 1: 
            #Check if the expected section was found.
            found_sections = list(result['sections'].keys())
            print(f"'{header}' ({header_description}): Found sections: {found_sections}")
        else: 
            print(f"'{header}' ({header_description}): No sections found")

def test_contact_extraction_patterns(): 
    """7) Test contact information extraction patterns"""
    print("\n" + "=" * 70)
    print("7) Testing contact information extraction patterns.")
    print("=" * 70)
    parser = SectionParser()
    # Test various contact formats
    contact_test_cases = [
        ("john.doe@email.com", "Standard email"),
        ("(555) 123-4567", "Phone with parentheses"),
        ("555-123-4567", "Phone with dashes"),
        ("555.123.4567", "Phone with dots"),
        ("+1 555 123 4567", "International phone"),
        ("linkedin.com/in/johndoe", "LinkedIn profile"),
        ("github.com/johndoe", "GitHub profile"),
        ("https://johndoe.com", "Personal website"),
        ("www.johndoe.com", "Website with www")
    ]

    for contact_text, contact_text_description in contact_test_cases: 
        test_text = f"""
        John Doe
        Software Engineer
        {contact_text}
        
        EXPERIENCE
        Some work experience here
        """
        result = parser.parse_sections(test_text)
        if result['parsing_status'] == 'success':
            found_contact = []
            for k, v in result['contact_info'].items(): 
                if v: 
                    found_contact.append(k)
            contact_vals = []
            for v in result['contact_info'].values(): 
                if v: 
                    contact_vals.append(v)
            print(f"{contact_text_description}: Found {len(found_contact)} fields: {contact_vals}")
        else: 
            print(f"{contact_text_description}: Parsing failed.")

if __name__ == "__main__": 
    #Configure logging 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Run all tests
    test_section_parsing_full_pipeline()
    test_section_parsing_with_sample_text()
    test_edge_cases()
    test_specific_patterns()
    test_contact_extraction_patterns()