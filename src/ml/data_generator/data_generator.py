#src/ml/data_generator/data_generator.py
"""
Main data generation class that orchestrates the generation of training datasets. Combines
taxonomy, profile generators, and dataset management.
"""

import random
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from .tech_taxonomy import TechSkillsTaxonomy
from .profile_generators import (
    JobRequirement, CandidateProfile, JobDescriptionGenerator,
    CandidateProfileGenerator, TextFormatter, MatchingMetricsCalculator
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample: 
    """Training example with job, resume, and match label."""
    job_description: str
    resume_text: str
    job_requirements: JobRequirement
    candidate_profile: CandidateProfile
    match_label: int  # 1 for match, 0 for no match
    match_score: float  # 0.0 to 1.0
    skill_overlap_percentage: float
    experience_match: bool
    education_match: bool
    missing_critical_skills: List[str]

class TechJobDataGenerator: 
    """Generate synthetic tech job descriptions and matching resumes."""
    def __init__(self, seed: int=42): 
        random.seed(seed)
        np.random.seed(seed)
        #initialize components 
        self.taxonomy = TechSkillsTaxonomy()
        self.job_generator = JobDescriptionGenerator(self.taxonomy)
        self.candidate_generator = CandidateProfileGenerator(self.taxonomy)
        self.text_formatter = TextFormatter()
        self.metrics_calculator = MatchingMetricsCalculator()

    def generate_training_example(
            self, 
            role_type: str, 
            experience_level: str, 
            match_type: str
        ) -> TrainingExample: 
        """Generate a complete training example."""
        #Generate job and candidate: 
        job_req = self.job_generator.generate(role_type, experience_level)
        profile = self.candidate_generator.generate(job_req, match_type)
        
        # Format as text
        job_text = self.text_formatter.format_job_description(job_req)
        resume_text = self.text_formatter.format_resume_text(profile)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_match_metrics(job_req, profile)
        if metrics['match_score'] >= 0.6: 
            match_label = 1
        else: 
            match_label = 0
        return TrainingExample(
            job_description=job_text,
            resume_text=resume_text,
            job_requirements=job_req,
            candidate_profile=profile,
            match_label=match_label,
            match_score=metrics['match_score'],
            skill_overlap_percentage=metrics['skill_overlap_percentage'],
            experience_match=metrics['experience_match'],
            education_match=metrics['education_match'],
            missing_critical_skills=metrics['missing_critical_skills']
        )
    
    def generate_training_dataset(
            self,
            num_examples: int = 1000,
            role_distribution: Optional[Dict[str, float]] = None,
            experience_distribution: Optional[Dict[str, float]] = None,
            match_distribution: Optional[Dict[str, float]] = None
        ) -> List[TrainingExample]: 
        """Generate a balanced training dataset."""
        #Default distributions 
        if role_distribution is None: 
            role_distribution = {
                'software_engineer': 0.4,
                'frontend_engineer': 0.2,
                'data_scientist': 0.2,
                'devops_engineer': 0.1,
                'mobile_developer': 0.1
            }
        if experience_distribution is None:
            experience_distribution = {
                'entry': 0.3,
                'mid': 0.4,
                'senior': 0.25,
                'staff': 0.05
            }
        
        if match_distribution is None:
            match_distribution = {
                'strong_match': 0.3,
                'partial_match': 0.4,
                'weak_match': 0.3
            }
        
        dataset = []

        """
        np.random.choice(a, p=None)
        a = the array/list to choose from
        p = the probability weights for each element
        Note: Converting the keys and values to lists preserves the order. 
        """

        for i in range(num_examples):
            # Sample role type
            role_type = np.random.choice(
                list(role_distribution.keys()),
                p=list(role_distribution.values())
            )
            
            # Sample experience level
            exp_level = np.random.choice(
                list(experience_distribution.keys()),
                p=list(experience_distribution.values())
            )
            
            # Sample match type
            match_type = np.random.choice(
                list(match_distribution.keys()),
                p=list(match_distribution.values())
            )


