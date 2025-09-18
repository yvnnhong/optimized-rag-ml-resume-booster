# Data Generation Execution Flow

## Overview
Creates synthetic resume-job pairs with ground truth labels for ML model training.

## File Execution Order

### 1. `tech_taxonomy.py` (Foundation)
**Purpose**: Domain knowledge and template definitions. This is the data generation template.
**Contains**: 
- Tech skill categories (programming languages, frameworks, databases, cloud tools)
- Job description templates for different roles (SWE, data scientist, DevOps, etc.)
- Resume templates by experience level (entry, mid, senior, staff)
- Company names and candidate name pools

**Dependencies**: None
**Usage**: `from .tech_taxonomy import TechSkillsTaxonomy`

### 2. `profile_generators.py` (Logic Layer)
**Purpose**: Generation logic using the taxonomy
**Contains**:
- `JobDescriptionGenerator` - Creates job postings from templates
- `CandidateProfileGenerator` - Creates resumes with controlled skill overlap
- `TextFormatter` - Converts structured data to readable text
- `MatchingMetricsCalculator` - Computes ground truth match scores

**Dependencies**: Requires `tech_taxonomy.py`
**Usage**: `from .profile_generators import JobDescriptionGenerator, CandidateProfileGenerator`

### 3. `data_generator.py` (Orchestration)
**Purpose**: Main interface that coordinates everything
**Contains**:
- `TechJobDataGenerator` - High-level API for dataset creation
- Dataset generation with configurable distributions
- Save/load functionality for training data
- Statistics and visualization methods

**Dependencies**: Requires both files above
**Usage**: `from .data_generator import TechJobDataGenerator`

## Execution Flow
```
TechSkillsTaxonomy → JobDescriptionGenerator → TrainingExample
                  ↘ CandidateProfileGenerator ↗
                                ↓
                         TechJobDataGenerator
                                ↓
                      synthetic_dataset.json
```

## Quick Start
```python
from src.ml.data_generator.data_generator import TechJobDataGenerator

generator = TechJobDataGenerator(seed=42)
dataset = generator.generate_training_dataset(num_examples=1000)
generator.save_dataset(dataset, "data/training/synthetic_dataset.json")
```