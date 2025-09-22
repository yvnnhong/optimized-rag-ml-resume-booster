# src/ml/scoring/feature_extractors.py
"""
Feature extraction and engineering for resume-job matching
Converts text and embeddings into numerical features for scoring models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import re
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractedFeatures:
    """Container for all extracted features"""
    text_features: torch.Tensor
    keyword_features: torch.Tensor  
    structure_features: torch.Tensor
    experience_features: torch.Tensor
    similarity_features: torch.Tensor
    ats_features: torch.Tensor
    combined_features: torch.Tensor

class TextualFeatureExtractor:
    """
    Extract features related to text quality, readability, and structure
    """
    
    def __init__(self):
        """Initialize with pre-compiled patterns for efficiency"""
        self.bullet_pattern = re.compile(r'[•\-\*\+]\s+')
        self.number_pattern = re.compile(r'\d+')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        
    def extract_text_features(self, resume_text: str, job_text: str) -> Dict[str, float]:
        """
        Extract text-based features from resume and job description
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            Dictionary of text features
        """
        features = {}
        
        # Resume text quality features
        resume_words = len(resume_text.split())
        resume_sentences = len(re.split(r'[.!?]+', resume_text))
        resume_chars = len(resume_text)
        
        features['resume_word_count'] = float(resume_words)
        features['resume_sentence_count'] = float(resume_sentences)
        features['resume_char_count'] = float(resume_chars)
        features['resume_avg_word_length'] = float(resume_chars / max(resume_words, 1))
        features['resume_avg_sentence_length'] = float(resume_words / max(resume_sentences, 1))
        
        # Job text features
        job_words = len(job_text.split())
        features['job_word_count'] = float(job_words)
        
        # Text complexity features
        features['resume_unique_word_ratio'] = len(set(resume_text.lower().split())) / max(resume_words, 1)
        features['resume_bullet_points'] = float(len(self.bullet_pattern.findall(resume_text)))
        features['resume_numbers_count'] = float(len(self.number_pattern.findall(resume_text)))
        
        # Professional formatting features
        features['has_email'] = float(bool(self.email_pattern.search(resume_text)))
        features['has_phone'] = float(bool(self.phone_pattern.search(resume_text)))
        features['has_url'] = float(bool(self.url_pattern.search(resume_text)))
        
        # Text quality ratios
        features['text_length_ratio'] = float(resume_chars / max(len(job_text), 1))
        features['word_count_ratio'] = float(resume_words / max(job_words, 1))
        
        # Capitalization patterns (professional formatting)
        uppercase_count = sum(1 for c in resume_text if c.isupper())
        features['capitalization_ratio'] = float(uppercase_count / max(resume_chars, 1))
        
        return features

class KeywordFeatureExtractor:
    """
    Extract keyword matching features between resume and job description
    """
    
    def __init__(self):
        """Initialize TF-IDF vectorizer for keyword analysis"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Technical skill patterns
        self.tech_patterns = {
            'programming_languages': r'\b(?:python|java|javascript|typescript|c\+\+|c#|go|rust|php|ruby|swift|kotlin|scala|r|matlab)\b',
            'frameworks': r'\b(?:react|angular|vue|django|flask|spring|express|laravel|rails|pytorch|tensorflow)\b',
            'databases': r'\b(?:mysql|postgresql|mongodb|redis|elasticsearch|cassandra|dynamodb|sqlite)\b',
            'cloud_tools': r'\b(?:aws|azure|gcp|docker|kubernetes|terraform|ansible)\b',
            'tools': r'\b(?:git|github|jira|jenkins|linux|unix)\b'
        }
        
    def extract_keyword_features(self, resume_text: str, job_text: str) -> Dict[str, float]:
        """
        Extract keyword matching features
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            Dictionary of keyword features
        """
        features = {}
        
        # Basic keyword overlap
        resume_words = set(resume_text.lower().split())
        job_words = set(job_text.lower().split())
        
        common_words = resume_words.intersection(job_words)
        features['keyword_overlap_count'] = float(len(common_words))
        features['keyword_overlap_ratio'] = float(len(common_words) / max(len(job_words), 1))
        
        # Technical skills matching
        for category, pattern in self.tech_patterns.items():
            job_skills = set(re.findall(pattern, job_text.lower(), re.IGNORECASE))
            resume_skills = set(re.findall(pattern, resume_text.lower(), re.IGNORECASE))
            
            common_skills = job_skills.intersection(resume_skills)
            features[f'{category}_match_count'] = float(len(common_skills))
            features[f'{category}_match_ratio'] = float(len(common_skills) / max(len(job_skills), 1))
        
        # TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([resume_text, job_text])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            features['tfidf_similarity'] = float(cosine_sim)
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
            features['tfidf_similarity'] = 0.0
        
        # Exact phrase matching
        job_phrases = self._extract_phrases(job_text, min_length=2, max_length=4)
        phrase_matches = sum(1 for phrase in job_phrases if phrase.lower() in resume_text.lower())
        features['phrase_match_count'] = float(phrase_matches)
        features['phrase_match_ratio'] = float(phrase_matches / max(len(job_phrases), 1))
        
        return features
    
    def _extract_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
        """Extract meaningful phrases from text"""
        words = text.split()
        phrases = []
        
        for length in range(min_length, min(max_length + 1, len(words) + 1)):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i + length])
                if len(phrase) > 10 and not phrase.lower().startswith(('the ', 'and ', 'or ', 'but ')):
                    phrases.append(phrase)
        
        return phrases

class StructuralFeatureExtractor:
    """
    Extract features related to resume structure and organization
    """
    
    def __init__(self):
        """Initialize section patterns"""
        self.section_patterns = {
            'experience': r'(?:work\s+)?experience|employment|professional\s+history',
            'education': r'education|academic|qualifications?',
            'skills': r'skills?|competencies|technologies',
            'projects': r'projects?|portfolio',
            'certifications': r'certifications?|licenses?'
        }
        
    def extract_structural_features(self, resume_text: str, job_text: str) -> Dict[str, float]:
        """
        Extract structural features from resume
        
        Args:
            resume_text: Resume text
            job_text: Job description text (for context)
            
        Returns:
            Dictionary of structural features
        """
        features = {}
        
        # Section detection
        sections_found = 0
        for section_name, pattern in self.section_patterns.items():
            if re.search(pattern, resume_text, re.IGNORECASE):
                sections_found += 1
                features[f'has_{section_name}_section'] = 1.0
            else:
                features[f'has_{section_name}_section'] = 0.0
        
        features['total_sections_found'] = float(sections_found)
        features['section_completeness'] = float(sections_found / len(self.section_patterns))
        
        # Resume length and structure
        lines = resume_text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        features['total_lines'] = float(len(lines))
        features['content_lines'] = float(len(non_empty_lines))
        features['empty_line_ratio'] = float((len(lines) - len(non_empty_lines)) / max(len(lines), 1))
        
        # Formatting consistency
        bullet_lines = sum(1 for line in non_empty_lines if re.match(r'^\s*[•\-\*\+]\s+', line))
        features['bullet_line_ratio'] = float(bullet_lines / max(len(non_empty_lines), 1))
        
        # Header detection (capitalized lines, likely section headers)
        header_lines = sum(1 for line in non_empty_lines if line.isupper() and len(line.split()) <= 5)
        features['header_count'] = float(header_lines)
        
        return features

class ExperienceFeatureExtractor:
    """
    Extract features related to work experience and career progression
    """
    
    def __init__(self):
        """Initialize experience-related patterns"""
        self.experience_patterns = {
            'years': r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
            'senior_indicators': r'\b(?:senior|lead|principal|staff|head\s+of|director|manager|vp|vice\s+president)\b',
            'junior_indicators': r'\b(?:junior|entry|associate|intern|trainee|assistant)\b',
            'date_ranges': r'(\d{4})\s*[-–]\s*(\d{4}|present)',
        }
        
    def extract_experience_features(self, resume_text: str, job_text: str) -> Dict[str, float]:
        """
        Extract experience-related features
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            Dictionary of experience features
        """
        features = {}
        
        # Extract years of experience mentioned
        years_mentioned = []
        for match in re.finditer(self.experience_patterns['years'], resume_text, re.IGNORECASE):
            try:
                years = int(match.group(1))
                years_mentioned.append(years)
            except ValueError:
                continue
        
        features['explicit_years_mentioned'] = float(max(years_mentioned) if years_mentioned else 0)
        
        # Calculate experience from date ranges
        calculated_years = self._calculate_years_from_dates(resume_text)
        features['calculated_years_experience'] = float(calculated_years)
        
        # Job requirement analysis
        job_years_required = self._extract_required_years(job_text)
        features['job_years_required'] = float(job_years_required)
        
        # Experience gap analysis
        total_experience = max(calculated_years, max(years_mentioned) if years_mentioned else 0)
        features['experience_gap'] = float(max(0, job_years_required - total_experience))
        features['experience_excess'] = float(max(0, total_experience - job_years_required))
        features['experience_match_ratio'] = float(min(total_experience / max(job_years_required, 1), 2.0))
        
        # Seniority level analysis
        resume_senior_count = len(re.findall(self.experience_patterns['senior_indicators'], resume_text, re.IGNORECASE))
        resume_junior_count = len(re.findall(self.experience_patterns['junior_indicators'], resume_text, re.IGNORECASE))
        
        job_senior_count = len(re.findall(self.experience_patterns['senior_indicators'], job_text, re.IGNORECASE))
        job_junior_count = len(re.findall(self.experience_patterns['junior_indicators'], job_text, re.IGNORECASE))
        
        features['resume_seniority_score'] = float(resume_senior_count - resume_junior_count)
        features['job_seniority_score'] = float(job_senior_count - job_junior_count)
        features['seniority_alignment'] = float(1.0 - abs(features['resume_seniority_score'] - features['job_seniority_score']) / 5.0)
        
        # Career progression indicators
        features['job_count'] = float(self._count_jobs(resume_text))
        features['career_progression_score'] = self._calculate_progression_score(resume_text)
        
        return features
    
    def _calculate_years_from_dates(self, resume_text: str) -> float:
        """Calculate total years of experience from date ranges"""
        date_ranges = re.findall(self.experience_patterns['date_ranges'], resume_text, re.IGNORECASE)
        total_years = 0.0
        
        current_year = 2024  # Update as needed
        
        for start_year, end_year in date_ranges:
            try:
                start = int(start_year)
                end = current_year if end_year.lower() == 'present' else int(end_year)
                years = max(0, end - start)
                total_years += years
            except ValueError:
                continue
        
        return total_years
    
    def _extract_required_years(self, job_text: str) -> float:
        """Extract years of experience required from job description"""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
            r'minimum\s+of\s+(\d+)\s*(?:years?|yrs?)',
            r'at\s+least\s+(\d+)\s*(?:years?|yrs?)'
        ]
        
        years_found = []
        for pattern in patterns:
            matches = re.findall(pattern, job_text, re.IGNORECASE)
            years_found.extend([int(match) for match in matches if match.isdigit()])
        
        return float(max(years_found) if years_found else 0)
    
    def _count_jobs(self, resume_text: str) -> int:
        """Count number of jobs/positions in resume"""
        # Look for job titles followed by company names and date ranges
        job_indicators = [
            r'(?:engineer|developer|analyst|manager|director|specialist|consultant)',
            r'\d{4}\s*[-–]\s*(?:\d{4}|present)',
        ]
        
        job_count = 0
        lines = resume_text.split('\n')
        
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in job_indicators):
                if re.search(r'\d{4}\s*[-–]\s*(?:\d{4}|present)', line):
                    job_count += 1
        
        return job_count
    
    def _calculate_progression_score(self, resume_text: str) -> float:
        """Calculate career progression score based on title progression"""
        # Simplified progression scoring
        senior_terms = ['senior', 'lead', 'principal', 'staff', 'manager', 'director']
        progression_score = 0.0
        
        lines = resume_text.split('\n')
        for i, line in enumerate(lines):
            for j, term in enumerate(senior_terms):
                if term in line.lower():
                    # Earlier positions (later in resume) get less weight
                    position_weight = 1.0 - (i / len(lines))
                    # Higher seniority terms get more weight
                    seniority_weight = (j + 1) / len(senior_terms)
                    progression_score += position_weight * seniority_weight
        
        return float(progression_score)

class SimilarityFeatureExtractor:
    """
    Extract similarity features using embeddings and other measures
    """
    
    def __init__(self):
        """Initialize similarity extractors"""
        pass
    
    def extract_similarity_features(self, 
                                  job_embedding: torch.Tensor,
                                  resume_embedding: torch.Tensor,
                                  job_text: str,
                                  resume_text: str) -> Dict[str, float]:
        """
        Extract similarity features between job and resume
        
        Args:
            job_embedding: Job description embedding
            resume_embedding: Resume embedding
            job_text: Job description text
            resume_text: Resume text
            
        Returns:
            Dictionary of similarity features
        """
        features = {}
        
        # Cosine similarity
        cosine_sim = torch.cosine_similarity(job_embedding, resume_embedding, dim=0)
        features['cosine_similarity'] = float(cosine_sim)
        
        # Euclidean distance (normalized)
        euclidean_dist = torch.norm(job_embedding - resume_embedding)
        features['euclidean_distance'] = float(euclidean_dist)
        features['normalized_euclidean'] = float(euclidean_dist / (torch.norm(job_embedding) + torch.norm(resume_embedding)))
        
        # Dot product similarity
        dot_product = torch.dot(job_embedding, resume_embedding)
        features['dot_product_similarity'] = float(dot_product)
        
        # Manhattan distance
        manhattan_dist = torch.sum(torch.abs(job_embedding - resume_embedding))
        features['manhattan_distance'] = float(manhattan_dist)
        
        # Embedding dimension analysis
        embedding_diff = job_embedding - resume_embedding
        features['mean_embedding_diff'] = float(torch.mean(embedding_diff))
        features['std_embedding_diff'] = float(torch.std(embedding_diff))
        features['max_embedding_diff'] = float(torch.max(torch.abs(embedding_diff)))
        
        # Text length similarities
        job_len = len(job_text.split())
        resume_len = len(resume_text.split())
        features['length_similarity'] = float(1.0 - abs(job_len - resume_len) / max(job_len + resume_len, 1))
        
        return features

class ATSFeatureExtractor:
    """
    Extract features specifically for ATS (Applicant Tracking System) compatibility
    """
    
    def __init__(self):
        """Initialize ATS-specific patterns"""
        self.ats_friendly_formats = [
            r'\.pdf$',
            r'\.docx?$',
            r'\.txt$'
        ]
        
        self.problematic_elements = [
            r'[^\x00-\x7F]',  # Non-ASCII characters
            r'[{}]',  # Curly braces
            r'\|',    # Pipes
            r'[@#$%^&*]',  # Special characters
        ]
    
    def extract_ats_features(self, resume_text: str, job_text: str, 
                           filename: Optional[str] = None) -> Dict[str, float]:
        """
        Extract ATS compatibility features
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            filename: Original filename (for format analysis)
            
        Returns:
            Dictionary of ATS features
        """
        features = {}
        
        # Format compatibility
        if filename:
            is_ats_format = any(re.search(pattern, filename.lower()) 
                               for pattern in self.ats_friendly_formats)
            features['ats_friendly_format'] = float(is_ats_format)
        else:
            features['ats_friendly_format'] = 0.5  # Unknown
        
        # Text parsing friendliness
        non_ascii_chars = len(re.findall(self.problematic_elements[0], resume_text))
        features['non_ascii_char_ratio'] = float(non_ascii_chars / max(len(resume_text), 1))
        
        # Special character analysis
        special_char_count = sum(len(re.findall(pattern, resume_text)) 
                               for pattern in self.problematic_elements[1:])
        features['special_char_density'] = float(special_char_count / max(len(resume_text.split()), 1))
        
        # Keyword density for ATS scanning
        job_keywords = self._extract_important_keywords(job_text)
        keyword_matches = sum(1 for keyword in job_keywords 
                             if keyword.lower() in resume_text.lower())
        features['keyword_density'] = float(keyword_matches / max(len(job_keywords), 1))
        
        # Structure clarity for ATS parsing
        clear_sections = sum(1 for pattern in [
            r'experience|work history',
            r'education|academic',
            r'skills|competencies',
            r'contact|personal information'
        ] if re.search(pattern, resume_text, re.IGNORECASE))
        
        features['section_clarity_score'] = float(clear_sections / 4.0)
        
        # Resume length appropriateness
        word_count = len(resume_text.split())
        if 300 <= word_count <= 800:  # Optimal range for ATS
            features['optimal_length'] = 1.0
        else:
            features['optimal_length'] = float(max(0, 1.0 - abs(word_count - 550) / 550))
        
        return features
    
    def _extract_important_keywords(self, job_text: str) -> List[str]:
        """Extract important keywords from job description for ATS matching"""
        # Focus on technical terms, skills, and qualifications
        important_patterns = [
            r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b',  # Proper nouns
            r'\b(?:python|java|react|aws|sql|excel|project\s+management)\b',  # Common skills
            r'\brequired?\b.*?(?=\.|$)',  # Requirements
            r'\bexperience\s+(?:with|in)\s+([^.]+)',  # Experience requirements
        ]
        
        keywords = []
        for pattern in important_patterns:
            matches = re.findall(pattern, job_text, re.IGNORECASE)
            if isinstance(matches, list):
                keywords.extend([match.strip() for match in matches if len(match.strip()) > 2])
        
        return list(set(keywords))[:20]  # Return top 20 unique keywords

class ComprehensiveFeatureExtractor:
    """
    Main feature extractor that combines all specialized extractors
    """
    
    def __init__(self):
        """Initialize all feature extractors"""
        self.text_extractor = TextualFeatureExtractor()
        self.keyword_extractor = KeywordFeatureExtractor()
        self.structure_extractor = StructuralFeatureExtractor()
        self.experience_extractor = ExperienceFeatureExtractor()
        self.similarity_extractor = SimilarityFeatureExtractor()
        self.ats_extractor = ATSFeatureExtractor()
        
    def extract_all_features(self,
                           resume_text: str,
                           job_text: str,
                           job_embedding: torch.Tensor,
                           resume_embedding: torch.Tensor,
                           filename: Optional[str] = None) -> ExtractedFeatures:
        """
        Extract all features for resume-job matching
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            job_embedding: Job description embedding
            resume_embedding: Resume embedding
            filename: Resume filename (optional)
            
        Returns:
            ExtractedFeatures object with all feature categories
        """
        # Extract features from each category
        text_features = self.text_extractor.extract_text_features(resume_text, job_text)
        keyword_features = self.keyword_extractor.extract_keyword_features(resume_text, job_text)
        structure_features = self.structure_extractor.extract_structural_features(resume_text, job_text)
        experience_features = self.experience_extractor.extract_experience_features(resume_text, job_text)
        similarity_features = self.similarity_extractor.extract_similarity_features(
            job_embedding, resume_embedding, job_text, resume_text
        )
        ats_features = self.ats_extractor.extract_ats_features(resume_text, job_text, filename)
        
        # Convert to tensors
        text_tensor = torch.tensor(list(text_features.values()), dtype=torch.float32)
        keyword_tensor = torch.tensor(list(keyword_features.values()), dtype=torch.float32)
        structure_tensor = torch.tensor(list(structure_features.values()), dtype=torch.float32)
        experience_tensor = torch.tensor(list(experience_features.values()), dtype=torch.float32)
        similarity_tensor = torch.tensor(list(similarity_features.values()), dtype=torch.float32)
        ats_tensor = torch.tensor(list(ats_features.values()), dtype=torch.float32)
        
        # Combine all features
        combined_tensor = torch.cat([
            text_tensor,
            keyword_tensor,
            structure_tensor,
            experience_tensor,
            similarity_tensor,
            ats_tensor
        ])
        
        return ExtractedFeatures(
            text_features=text_tensor,
            keyword_features=keyword_tensor,
            structure_features=structure_tensor,
            experience_features=experience_tensor,
            similarity_features=similarity_tensor,
            ats_features=ats_tensor,
            combined_features=combined_tensor
        )
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get names of all features for interpretability"""
        return {
            'text_features': [
                'resume_word_count', 'resume_sentence_count', 'resume_char_count',
                'resume_avg_word_length', 'resume_avg_sentence_length', 'job_word_count',
                'resume_unique_word_ratio', 'resume_bullet_points', 'resume_numbers_count',
                'has_email', 'has_phone', 'has_url', 'text_length_ratio',
                'word_count_ratio', 'capitalization_ratio'
            ],
            'keyword_features': [
                'keyword_overlap_count', 'keyword_overlap_ratio', 'tfidf_similarity',
                'phrase_match_count', 'phrase_match_ratio'
            ] + [f'{cat}_{metric}' for cat in ['programming_languages', 'frameworks', 'databases', 'cloud_tools', 'tools']
                 for metric in ['match_count', 'match_ratio']],
            'structure_features': [
                'has_experience_section', 'has_education_section', 'has_skills_section',
                'has_projects_section', 'has_certifications_section', 'total_sections_found',
                'section_completeness', 'total_lines', 'content_lines', 'empty_line_ratio',
                'bullet_line_ratio', 'header_count'
            ],
            'experience_features': [
                'explicit_years_mentioned', 'calculated_years_experience', 'job_years_required',
                'experience_gap', 'experience_excess', 'experience_match_ratio',
                'resume_seniority_score', 'job_seniority_score', 'seniority_alignment',
                'job_count', 'career_progression_score'
            ],
            'similarity_features': [
                'cosine_similarity', 'euclidean_distance', 'normalized_euclidean',
                'dot_product_similarity', 'manhattan_distance', 'mean_embedding_diff',
                'std_embedding_diff', 'max_embedding_diff', 'length_similarity'
            ],
            'ats_features': [
                'ats_friendly_format', 'non_ascii_char_ratio', 'special_char_density',
                'keyword_density', 'section_clarity_score', 'optimal_length'
            ]
        }
    
    def get_feature_count(self) -> int:
        """Get total number of features"""
        feature_names = self.get_feature_names()
        return sum(len(names) for names in feature_names.values())