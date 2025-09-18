#src/ml/data_generator/profile_generators.py
"""
Job and resume profile generation logic. 
Handles the creation of synthetic job descriptions and candidate profiles. 
"""
import random 
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from .tech_taxonomy import TechSkillsTaxonomy

logger = logging.getLogger(__name__)

@dataclass
class JobRequirement: 
    """Structured job reuqirement data"""
    job_title: str
    company_type: str
    experience_level: str #entry, mid, senior, staff
    experience_years: int
    required_skills: List[str]
    preferred_skills: List[str]
    education_requirements: List[str]
    certifications: List[str]
    responsibilities: List[str]
    industry: str = "technology"

@dataclass
class CandidateProfile: 
    """Structured candidate resume data."""
    name: str
    email: str
    phone: str
    linkedin: str
    github: str
    summary: str
    experience_years: int
    experience_level: str
    skills: List[str]
    education: List[str]
    certifications: List[str]
    projects: List[Dict[str, str]]
    work_history: List[Dict[str, Any]]

class JobDescriptionGenerator: 
    """Generate synthetic job descriptions"""
    def __init__(self, taxonomy: TechSkillsTaxonomy): 
        self.taxonomy = taxonomy

    def generate(self, role_type: str, experience_level: str) -> JobRequirement: 
        """Generate a synthetic job description."""
        template = self.taxonomy.job_templates[role_type]
        #select job title
        #random.choice randomly selects 1 item from a list/sequence
        job_title = random.choice(template['titles'])
        if experience_level == 'senior': 
            job_title = f"Senior {job_title}"
        elif experience_level == 'staff': 
            job_title = f"Staff {job_title}"

        #generate required skills: 
        required_skills = []
        for skill_path in template['core_skills']: 
            skill_pool = self.taxonomy.get_skills_by_path(skill_path)
            required_skills.extend(random.sample(skill_pool, min(3, len(skill_pool))))
            #extend = add multiple items to a list ; append = add a single item to a list 

        #generate preferred skills
        preferred_skills = []
        for skill_path in template['preferred_skills']: 
            skill_pool = self.taxonomy.get_skills_by_path(skill_path)
            preferred_skills.extend(random.sample(skill_pool, min(2, len(skill_pool))))

        #experience requiremments 
        #the mapping is inclusive of both ends
        exp_mapping = {'entry': (0, 2), 'mid': (3, 5), 'senior': (6, 8), 'staff': (9, 15)}
        min_exp, max_exp = exp_mapping[experience_level]
        experience_years = random.randint(min_exp, max_exp)

        #random.randint(a,b) -> some int between [a,b] 
        #random.random() -> some float in [0.0, 1.0)

        # Education and certifications
        education_reqs = ['Bachelor\'s degree in Computer Science or related field']
        if experience_level in ['senior', 'staff']:
            if random.random() < 0.3: #30% chance of requiring a masters degree
                education_reqs.append('Master\'s degree preferred')

        # Select relevant certifications
        cert_categories = list(self.taxonomy.certifications.keys())
        selected_certs = []
        if random.random() < 0.4:  # 40% chance of requiring certifications
            cat = random.choice(cert_categories)
            selected_certs.extend(random.sample(self.taxonomy.certifications[cat], 
                                              min(2, len(self.taxonomy.certifications[cat]))))

        return JobRequirement(
            job_title=job_title,
            company_type=random.choice(['startup', 'midsize', 'enterprise']),
            experience_level=experience_level,
            experience_years=experience_years,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            education_requirements=education_reqs,
            certifications=selected_certs,
            responsibilities=template['responsibilities'],
            industry='technology'
        )