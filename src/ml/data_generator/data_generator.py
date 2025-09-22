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