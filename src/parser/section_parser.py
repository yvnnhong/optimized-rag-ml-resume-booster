#section_parser.py is step 3 to parse individual sections (e.x education, work history, etc)
import re
import logging
from typing import Dict, List, Optional, Any 

logger = logging.getLogger(__name__)

class ResumeSection: 
    """Data class to represent a parsed resume section."""
    def __init__(self, name: str, content: str, start_index: int, end_index: int, 
                 confidence: float) -> None: 
        self.name = name
        self.content = content
        self.start_index = start_index
        self.end_index = end_index
        self.confidence = confidence

class SectionParser:
    """Parse resume text into distinct sections."""
    def __init__(self): #Define common section headers and their variations.
        self.section_patterns: Dict[str, List[str]] = {
            'contact': [
                r'contact\s+information',
                r'personal\s+information',
                r'contact\s+details?'
            ],
            'summary': [
                r'professional\s+summary',
                r'career\s+summary',
                r'summary\s+of\s+qualifications?',
                r'summary',
                r'profile',
                r'objective',
                r'career\s+objectives?'
            ],
            'experience': [
                r'professional\s+experience',
                r'work\s+experience',
                r'employment\s+history',
                r'experience',
                r'career\s+history',
                r'work\s+history'
            ],
            'education': [
                r'education',
                r'educational\s+background',
                r'academic\s+background',
                r'qualifications?'
            ],
            'skills': [
                r'technical\s+skills?',
                r'core\s+competencies',
                r'skills?\s+and\s+abilities',
                r'skills?',
                r'competencies',
                r'technologies',
                r'languages', #make a section for programming languages??? 
                r'programming\s+languages?'
            ],
            'projects': [
                r'projects?',
                r'key\s+projects?',
                r'selected\s+projects?',
                r'notable\s+projects?'
            ],
            'certifications': [
                r'certifications?',
                r'licenses?\s+and\s+certifications?',
                r'professional\s+certifications?',
            ],
            'achievements': [
                r'achievements?',
                r'accomplishments?',
                r'awards?\s+and\s+honou?rs?',
                r'awards?',
                r'recognition'
            ]
        }

        #Compile regex patterns for improved performance
        #pipe | means to combine both flags 
        self.compiled_patterns = {}
        for section, patterns in self.section_patterns.items(): 
            self.compiled_patterns[section] = []
            for pattern in patterns: 
                compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                self.compiled_patterns[section].append(compiled_pattern)

    def parse_sections(self, resume_text: str) -> Dict[str, Any]: 
        """
        Parses resume text into sections. 
        Args: resume_text (str): Full resume text 
        Returns: dict: Parsed section w/ metadata 
        """
        try: 
            #step 1: Clean text 
            cleaned_text = self._clean_text(resume_text)
            #step 2: find section boundaries 
            section_matches = self._find_section_boundaries(cleaned_text)
            #step 3: extract section content 
            parsed_sections = self._extract_section_content(cleaned_text, section_matches)
            #step 4: extract contact information (usually at the top)
            contact_info = self._extract_contact_info(cleaned_text)
            result = {
                'sections': parsed_sections,
                'contact_info': contact_info,
                'total_sections': len(parsed_sections),
                'text_length': len(cleaned_text),
                'parsing_status': 'success'
            }
            logger.info(f"Successfully parsed {len(parsed_sections)} sections")
            return result
        #exception 
        except Exception as e: 
            logger.error(f"Error parsing resume sections: {str(e)}")
            if resume_text: 
                text_length = len(resume_text)
            else: 
                text_length = 0
            return {
                'sections': {},
                'contact_info': {},
                'total_sections': 0,
                'text_length': text_length,
                'parsing_status': 'failed',
                'error': str(e)
            }
        
    def _clean_text(self, text: str) -> str: 
        """Step 1: Clean and normalize resume text."""
        if not text: 
            return ""
        #remove extra whitespace and normalize line breaks 
        #re.sub(pattern, replacement, string)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        #remove special chars that might interfere with parsing
        text = re.sub(r'[^\w\s\n\.,;:()\-@/]', '', text)
        return text.strip()
    
    def _find_section_boundaries(self, text: str) -> List[Dict[str, Any]]: 
        """Step 2: find where each section starts in the text."""
        header_pattern_matches = []
        lines = text.split('\n')
        for line_idx, line in enumerate(lines): 
            line = line.strip()
            if not line: 
                continue
            #Check if this line matches any section header pattern: 
            for section_name, patterns in self.compiled_patterns.items(): 
                for pattern in patterns: 
                    match = pattern.search(line) 
                    #^.search() only the first occurrence of pattern, and stores it in match
                    #^returns a Match object (re module has built in re.Match) or None.
                    if match: 
                        char_position = 0 
                        for i in range(line_idx): 
                            char_position += len(lines[i]) + 1 #+1 since we need to 
                            #re-insert the newline that was removed from 
                            #lines = text.split('\n'). We're summing the lengths of all 
                            #prev lines, plus 1 for each newline. 
                            #and we're moving left to right starting from the beginning of the 
                            #text, to see which char position the header starts at relative
                            #from the very beginning. aka, 
                            #summing the lengths of all previous lines (including newlines)
                        header_pattern_matches.append({
                            'section': section_name,
                            'line_number': line_idx,
                            'section_starting_char_position': char_position,
                            'matched_text': match.group(),
                            #^returns the actual text from match = pattern.search(line)
                            'confidence': self._calculate_confidence(line, match.group())
                        })
                        break #stop checking for more patterns for the current section once 
                        #we've found a match (e.x. why keep checking if we already matched one
                        #variation of 'experience'?)
        #sort the matches from smallest char pos to largest char pos 
        header_pattern_matches = sorted(header_pattern_matches, key=lambda x: x['section_starting_char_position'])
        header_pattern_matches = self._remove_duplicate_matches(header_pattern_matches)
        return header_pattern_matches
    
    def _extract_section_content(self, text: str, section_matches: List[Dict[str, Any]]
                                 ) -> Dict[str, ResumeSection]: 
        """Extract content for each identified section."""
        sections = {}
        for i, match in enumerate(section_matches): 
            section_name = match['section']
            current_section_start_pos = match['section_starting_char_position']
            #Determine end position (start of nextsection OR end of text)
            if i + 1 < len(section_matches): 
                current_section_end_pos = section_matches[i + 1]['section_starting_char_position']
            else: 
                current_section_end_pos = len(text)
            #extract content 
            current_section_content = text[current_section_start_pos:current_section_end_pos].strip() 
            #^reminder: slicing is end-index exclusive
            #next, clean up the content (remove the header line)
            current_section_content_lines = current_section_content.split('\n')
            if current_section_content_lines: 
                current_section_content_lines = current_section_content_lines[1:] 
                #^remove header line 
            current_section_content = '\n'.join(current_section_content_lines).strip()
            """
            Note: 
            - current_section_content starts as a string 
            - .split('\n') makes it a list of strings
            - .join() turns the list back into one string, inserting \n between each item
            Example: 
            ```
            lines = ["Line 1", "Line 2", "Line 3"] 
            '\n'.join(lines) 
            ```
            Becomes: 
            ```
            "Line 1\nLine 2\nLine 3"
            ```
            Analogy: .join('\n') uses '\n' like "glue"
            """
            #next: only add if we have substantial content 
            if len(current_section_content) > 10: #greater than 10 chars 
                sections[section_name] = ResumeSection(
                    name = section_name,
                    content = current_section_content,
                    start_index = current_section_start_pos,
                    end_index = current_section_end_pos,
                    confidence = match['confidence']
                )
        return sections
    
    def _extract_contact_info(self, text: str) -> Dict[str, Optional[str]]: 
        """Extract contact info from resume text."""
        #note: Dict[str, Optional[str]] means that the values are allowed to be None
        #note: escape character \ allows us to treat the next char literally
        contact_info = {
            'email': None, 
            'phone': None, 
            'linkedin': None, 
            'github': None, 
            'website': None
        }
        #Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text) #find the first occurrence of the pattern
        if email_match: 
            contact_info['email'] = email_match.group() #store the match we found in LHS

        #Phone pattern (various formats)
        phone_pattern = r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, text)
        if phone_match: 
            contact_info['phone'] = phone_match.group()

        #Linkedin pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match: 
            contact_info['linkedin'] = linkedin_match.group()

        #Github pattern
        github_pattern = r'github\.com/[\w-]+'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match: 
            contact_info['github'] = github_match.group()

        #Wesite pattern (basic)
        website_pattern = r'https?://[\w.-]+\.[a-z]{2,}(?:/[\w.-]*)*'
        website_match = re.search(website_pattern, text, re.IGNORECASE)
        if website_match: 
            contact_info['website'] = website_match.group()

        return contact_info
    
    #note: refactor _calculate_confidence later to be more specific??
    def _calculate_confidence(self, line: str, matched_text: str) -> float: 
        """Calculate confidence score for section header match.
        This function checks how likely it is that a given line is a true 
        section header. It is only used for detecting section headers like 
        'experience', 'education', etc, as defined earlier in this file (see 
        self.section_patterns). """
        confidence = 0.5 #base confidence
        #higher confidence if it's on its own line 
        if line.strip().lower() == matched_text.lower():
            confidence += 0.3
        #higher confidence if it's at the beginning of the line 
        if line.strip().lower().startswith(matched_text.lower()): 
            confidence += 0.2 
        #lower confidence if there is a lot of other text on the same line 
        if len(line.strip()) > len(matched_text) * 3: 
            confidence -= 0.2
        return min(1.0, max(0.1, confidence)) #confidence ranges from 0.1 to 1
    
    def _remove_duplicate_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]: 
        """Remove duplicate section matches, keeping the highest confidence ones."""
        if not matches: 
            return matches
        #group by section name
        sections_found: Dict[str, Dict[str, Any]] = {}
        """
        sections_found is a dictionary mapping section names (str) to match dictionaries 
        (Dict[str, Any]).
        """
        for match in matches: 
            section_name = match['section']
            if section_name not in sections_found or match['confidence'] > sections_found[section_name]['confidence']:
                sections_found[section_name] = match
                #we're only keeping the match with the highest confidence (if dupes exist)
        #return sorted by ascending (increasing) char position: 
        return sorted(sections_found.values(), key=lambda x: x['section_starting_char_position'])
    
    #call get_section_summary from root/tests/test_parser/test_section_parser.py
    def get_section_summary(self, parsed_result: Dict[str, Any]) -> str: 
        """Generate a summary of parsed sections for debugging"""
        if parsed_result['parsing_status'] != 'success': 
            return f"Parsing failed: {parsed_result.get('error', 'Unknown error')}"
        sections = parsed_result['sections']
        contact = parsed_result['contact_info']
        summary = f"Parsed {len(sections)} sections:\n"
        for name, section in sections.items(): 
            #name.title() capitalizes each word in the section name (built-in python) 
            summary += f"- {name.title()}: {len(section.content)} chars (confidence: {section.confidence:.2f})\n"
        summary += "\nContact info found: "
        found_contact: List[str] = []
        for key in contact: 
            if contact[key]: 
                found_contact.append(key)
        #summary += ", ".join(found_contact) if found_contact else "None"
        if found_contact: 
            summary += ", ".join(found_contact)
        else: 
            summary += "None"
        return summary