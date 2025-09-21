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
        #random.choice() -> picks some random item in a given list, uniformly (?)

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

class CandidateProfileGenerator: 
    """Generate synthetic candidate profiles"""
    def __init__(self, taxonomy: TechSkillsTaxonomy): 
        self.taxonomy = taxonomy

    def generate(self, target_job: JobRequirement, match_type: str) -> CandidateProfile: 
        """Generate a candidate profile that matches/doesn't match the job. """
        #Generate basic info 
        first_name = random.choice(self.taxonomy.first_names)
        last_name = random.choice(self.taxonomy.last_names)
        name:str = f"{first_name} {last_name}"
        email = f"{first_name.lower()}.{last_name.lower()}@email.com" #increase variance 
        #determine skill overlap based on match type 
        """
        random.uniform(a, b): returns a random float between a and b (inclusive).
        Both endpoints can be returned.
        It's a uniform distribution (all values are equally likely in that range).
        """
        if match_type == 'strong_match':
            skill_overlap = random.uniform(0.7, 0.9) #70%-90% match
            exp_level_match = True
            exp_years_variance = random.uniform(-0.2, 0.3) #-20% to +30% variance
        elif match_type == 'partial_match':
            skill_overlap = random.uniform(0.4, 0.7) #40%-70%
            exp_level_match = random.choice([True, False])
            exp_years_variance = random.uniform(-0.4, 0.2) #-40% to +20% variance
        else: #weak_match or no_match
            skill_overlap = random.uniform(0.1, 0.4)
            exp_level_match = False
            exp_years_variance = random.uniform(-0.6, -0.1) #below requirements 

        #Generate skills based on overlap 
        all_job_skills: List[str] = list(set(target_job.required_skills 
                                             + target_job.preferred_skills))
        num_matching_skills = int(len(all_job_skills) * skill_overlap)
        #^int truncation floors the decimal 
        matching_skills: List[str] = random.sample(all_job_skills, num_matching_skills)
        #^randomly selects num_matching_skills items from all_job_skills

        """
        START CHECKING FROM BELOW HERE AND GOING OVER COMPREHENSIVELY.
        """

        #next: add some unrelated skills
        unrelated_skills = []
        for category in self.taxonomy.programming_languages.values(): 
            unrelated_skills.extend(category)
        for category in self.taxonomy.frameworks.values(): 
            unrelated_skills.extend(category)

        #remove skills that are already in matching_skills
        filtered_unrelated_skills = []
        for s in unrelated_skills: 
            if s not in matching_skills: 
                filtered_unrelated_skills.append(s)
        unrelated_skills = filtered_unrelated_skills
        num_unrelated = random.randint(2,5)
        additional_skills = random.sample(unrelated_skills, min(num_unrelated, len(unrelated_skills)))
        candidate_skills = matching_skills + additional_skills
        random.shuffle(candidate_skills)

        # Generate experience
        base_exp = target_job.experience_years
        candidate_exp = max(0, int(base_exp * (1 + exp_years_variance)))
        
        # Determine experience level
        if candidate_exp >= 8:
            candidate_level = 'staff' if candidate_exp >= 10 else 'senior'
        elif candidate_exp >= 3:
            candidate_level = 'mid'
        else:
            candidate_level = 'entry'

        # Generate work history
        template = self.taxonomy.resume_templates[candidate_level]
        num_jobs = random.randint(*template['num_jobs_range'])
        
        work_history = []
        current_year = datetime.now().year
        
        for i in range(num_jobs):
            start_year = current_year - candidate_exp + (i * (candidate_exp // max(1, num_jobs)))
            end_year = start_year + random.randint(1, 3) if i < num_jobs - 1 else current_year
            
            work_history.append({
                'company': random.choice(self.taxonomy.company_names),
                'title': f"{random.choice(['Software', 'Backend', 'Frontend'])} {'Engineer' if candidate_level != 'entry' else 'Developer'}",
                'start_year': start_year,
                'end_year': end_year,
                'responsibilities': random.sample([
                    'Developed web applications using modern frameworks',
                    'Collaborated with cross-functional teams',
                    'Participated in code reviews and technical discussions',
                    'Implemented automated testing and CI/CD pipelines',
                    'Optimized application performance and scalability'
                ], 3)
            })
        
        #Generate projects
        num_projects = random.randint(*template['num_projects_range'])
        projects = []
        project_names = ['E-commerce Platform', 'Task Management App', 'Data Visualization Tool', 
                        'Real-time Chat Application', 'Machine Learning Pipeline', 'Mobile Fitness App']
        
        for _ in range(num_projects):
            project_skills = random.sample(candidate_skills, min(4, len(candidate_skills)))
            projects.append({
                'name': random.choice(project_names),
                'description': f"Built using {', '.join(project_skills[:3])}",
                'technologies': project_skills
            })

        #Generate summary
        summary_template = random.choice(template['summary_templates'])
        if matching_skills: 
            primary_skill = matching_skills[0]
        else: 
            primary_skill = candidate_skills[0]
        if len(matching_skills) > 1: 
            secondary_skill = matching_skills[1]
        elif len(candidate_skills) > 1: 
            secondary_skill = candidate_skills[1]
        else:
            secondary_skill = 'web development'
        
        summary = summary_template.format(
            years=candidate_exp,
            job_category='engineer',
            primary_skill=primary_skill,
            secondary_skill=secondary_skill
        )
        
        return CandidateProfile(
            name=name,
            email=email,
            phone=f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            linkedin=f"linkedin.com/in/{first_name.lower()}-{last_name.lower()}",
            github=f"github.com/{first_name.lower()}{last_name.lower()}",
            summary=summary,
            experience_years=candidate_exp,
            experience_level=candidate_level,
            skills=candidate_skills,
            education=[random.choice(template['education_level']) + ' in Computer Science'],
            certifications=random.sample(target_job.certifications, 
                                       min(len(target_job.certifications), random.randint(0, 2))),
            projects=projects,
            work_history=work_history
        )
    
class TextFormatter: 
    """Format job requirements and candidate profiles as text."""

    
        