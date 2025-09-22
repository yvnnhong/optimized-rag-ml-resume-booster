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

            # Generate training example
            example = self.generate_training_example(role_type, exp_level, match_type)
            dataset.append(example)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_examples} training examples")
        return dataset
    
    def save_dataset(
            self,
            dataset: List[TrainingExample],
            output_path: str
        ) -> None: 
        """
        Save dataset to JSON file. 
        """
        #Convert dataclasses to dictionaries: 
        dataset_dict = []
        for example in dataset: 
            example_dict = asdict(example) #asdict converts a dataclass object to a dict
            dataset_dict.append(example_dict)
        #create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
            #json.dump() - Writes the data as JSON to the file
        
        logger.info(f"Saved {len(dataset)} training examples to {output_path}")

    
    def load_dataset(self, input_path: str) -> List[TrainingExample]:  
        """Load dataset from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset_dict = json.load(f)
        
        # Convert dictionaries back to dataclasses
        dataset = []
        for example_dict in dataset_dict:
            # Reconstruct nested dataclasses
            #note: The ** unpacks the dictionary into keyword arguments:
            job_req: JobRequirement = JobRequirement(**example_dict['job_requirements'])
            profile: CandidateProfile = CandidateProfile(**example_dict['candidate_profile'])
            
            example = TrainingExample(
                job_description=example_dict['job_description'],
                resume_text=example_dict['resume_text'],
                job_requirements=job_req,
                candidate_profile=profile,
                match_label=example_dict['match_label'],
                match_score=example_dict['match_score'],
                skill_overlap_percentage=example_dict['skill_overlap_percentage'],
                experience_match=example_dict['experience_match'],
                education_match=example_dict['education_match'],
                missing_critical_skills=example_dict['missing_critical_skills']
            )
            dataset.append(example)
        
        logger.info(f"Loaded {len(dataset)} training examples from {input_path}")
        return dataset
    
    def get_dataset_statistics(self, dataset: List[TrainingExample]) -> Dict[str, Any]:
        """Calculate statistics for the generated dataset"""
        
        if not dataset:
            return {}
        
        # Match distribution
        positive_matches = sum(1 for ex in dataset if ex.match_label == 1)
        negative_matches = len(dataset) - positive_matches
        
        # Role distribution
        role_counts = {}
        for example in dataset:
            role = example.job_requirements.job_title
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Experience level distribution
        exp_level_counts = {}
        for example in dataset:
            level = example.job_requirements.experience_level
            exp_level_counts[level] = exp_level_counts.get(level, 0) + 1
        
        # Score distribution
        scores = [ex.match_score for ex in dataset]
        avg_score = np.mean(scores)
        score_std = np.std(scores)
        
        # Skill overlap distribution
        overlaps = [ex.skill_overlap_percentage for ex in dataset]
        avg_overlap = np.mean(overlaps)
        overlap_std = np.std(overlaps)
        
        return {
            'total_examples': len(dataset),
            'positive_matches': positive_matches,
            'negative_matches': negative_matches,
            'match_ratio': positive_matches / len(dataset),
            'role_distribution': role_counts,
            'experience_level_distribution': exp_level_counts,
            'average_match_score': avg_score,
            'match_score_std': score_std,
            'average_skill_overlap': avg_overlap,
            'skill_overlap_std': overlap_std,
            'score_quartiles': {
                'q25': np.percentile(scores, 25),
                'q50': np.percentile(scores, 50),
                'q75': np.percentile(scores, 75)
            }
        }

    def print_example(self, example: TrainingExample, show_full_text: bool = False):
        """Print a training example in a readable format"""
        
        print(f"=== TRAINING EXAMPLE ===")
        print(f"Match Label: {example.match_label} ({'MATCH' if example.match_label else 'NO MATCH'})")
        print(f"Match Score: {example.match_score:.3f}")
        print(f"Skill Overlap: {example.skill_overlap_percentage:.3f}")
        print(f"Experience Match: {example.experience_match}")
        
        print(f"\n--- JOB REQUIREMENTS ---")
        job = example.job_requirements
        print(f"Title: {job.job_title}")
        print(f"Experience: {job.experience_level} ({job.experience_years}+ years)")
        print(f"Required Skills: {', '.join(job.required_skills[:5])}{'...' if len(job.required_skills) > 5 else ''}")
        print(f"Preferred Skills: {', '.join(job.preferred_skills[:3])}{'...' if len(job.preferred_skills) > 3 else ''}")
        
        print(f"\n--- CANDIDATE PROFILE ---")
        candidate = example.candidate_profile
        print(f"Name: {candidate.name}")
        print(f"Experience: {candidate.experience_level} ({candidate.experience_years} years)")
        print(f"Skills: {', '.join(candidate.skills[:8])}{'...' if len(candidate.skills) > 8 else ''}")
        print(f"Education: {candidate.education[0] if candidate.education else 'N/A'}")
        
        if example.missing_critical_skills:
            print(f"\nMissing Critical Skills: {', '.join(example.missing_critical_skills)}")
        
        if show_full_text:
            print(f"\n--- FULL JOB DESCRIPTION ---")
            print(example.job_description)
            print(f"\n--- FULL RESUME TEXT ---")
            print(example.resume_text)
        
        print("="*50)


# Example usage and testing functions
def generate_sample_dataset():
    """Generate and display a small sample dataset"""
    
    # Initialize generator
    generator = TechJobDataGenerator(seed=42)
    
    # Generate small sample dataset
    print("Generating sample training dataset...")
    dataset = generator.generate_training_dataset(num_examples=50)
    
    # Show statistics
    print("\n=== DATASET STATISTICS ===")
    stats = generator.get_dataset_statistics(dataset)
    print(f"Total examples: {stats['total_examples']}")
    print(f"Positive matches: {stats['positive_matches']} ({stats['match_ratio']:.1%})")
    print(f"Negative matches: {stats['negative_matches']}")
    print(f"Average match score: {stats['average_match_score']:.3f} ± {stats['match_score_std']:.3f}")
    print(f"Average skill overlap: {stats['average_skill_overlap']:.3f} ± {stats['skill_overlap_std']:.3f}")
    
    print(f"\nRole distribution:")
    for role, count in stats['role_distribution'].items():
        print(f"  {role}: {count}")
    
    print(f"\nExperience level distribution:")
    for level, count in stats['experience_level_distribution'].items():
        print(f"  {level}: {count}")
    
    # Show a few examples
    print(f"\n=== SAMPLE EXAMPLES ===")
    
    # Show one strong match
    strong_matches = [ex for ex in dataset if ex.match_label == 1 and ex.match_score > 0.8]
    if strong_matches:
        print(f"\n--- STRONG MATCH EXAMPLE ---")
        generator.print_example(strong_matches[0])
    
    # Show one weak match/no match
    weak_matches = [ex for ex in dataset if ex.match_label == 0]
    if weak_matches:
        print(f"\n--- NO MATCH EXAMPLE ---")
        generator.print_example(weak_matches[0])
    
    return dataset, generator


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate sample dataset
    dataset, generator = generate_sample_dataset()
    
    # Optionally save dataset
    output_path = "data/training/synthetic_dataset_sample.json"
    generator.save_dataset(dataset, output_path)
    print(f"\nDataset saved to {output_path}")
    
    # Test loading
    loaded_dataset = generator.load_dataset(output_path)
    print(f"Successfully loaded {len(loaded_dataset)} examples")

        




