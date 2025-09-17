# ML-Driven Resume Analyzer - Execution Flow (Updated for Refactored Structure)

## Project Overview
RAG-based resume optimization tool using fine-tuned ML models to analyze resume-job description matches and provide ATS optimization recommendations.

## Architecture Flow

```
User Input (PDF Resume + Job Description)
↓
PDF Extraction & Text Processing
↓
ML-Powered Analysis (Fine-tuned Models)
↓
Scoring & Recommendations
↓
Gradio Interface Output
```

## File Structure & Execution Flow

### 1. Data Generation Phase (Training Setup)
**Files**: `src/ml/data_generator/` (3 files)
- `tech_taxonomy.py` - Domain knowledge (skills, job templates, resume templates)
- `profile_generators.py` - Generation logic (jobs, resumes, formatting, metrics)  
- `data_generator.py` - Main orchestration (dataset creation, I/O, statistics)

**Purpose**: Generate synthetic resume-job pairs with ground truth labels
**Output**: Training dataset with match scores (0-1) and binary labels saved to `data/training/`
**Execution**: Run once to create training data for ML models

### 2. Model Training Phase

#### 2.1 Fine-tuned Embeddings
**Files**: `src/ml/classification/` (3 files)
- `embedding_models.py` - Model architectures (Siamese, multi-task, loss functions)
- `training_pipeline.py` - Training infrastructure (datasets, loops, persistence)
- `classification.py` - High-level API and CLI interface

**Purpose**: Fine-tune sentence transformers on domain-specific resume-job pairs
**Input**: Synthetic training dataset from data_generator
**Output**: Custom embedding model saved to `models/embeddings/`

#### 2.2 Neural Scoring Models
**Files**: `src/ml/scoring/` (9 files)
- `models/` subdirectory:
  - `match_scorer.py` - Overall match scoring model
  - `ats_scorer.py` - ATS compatibility scorer
  - `skill_gap_analyzer.py` - Skill gap analysis with attention
  - `experience_predictor.py` - Experience gap prediction
  - `multitask_scorer.py` - Multi-task learning model
  - `ensemble_scorer.py` - Model ensemble combining
- `loss_functions.py` - Custom loss functions (contrastive, focal, etc.)
- `evaluation.py` - Metrics and model evaluation utilities
- `feature_extractors.py` - Feature engineering from text/embeddings
- `scoring.py` - Main scoring interface and training pipeline

**Purpose**: PyTorch neural networks that predict match scores from features
**Input**: Embeddings + extracted features (skill overlap, experience gaps, etc.)
**Output**: Trained scoring models saved to `models/scorers/`

#### 2.3 Model Utilities
**File**: `src/ml/model_utils.py`
**Purpose**: Loading, saving, and managing trained models across all ML components
**Functions**: Model persistence, version management, inference helpers

### 3. Text Processing Pipeline (Existing, No Changes)

#### 3.1 PDF Extraction
**File**: `src/parser/pdf_extractor.py`
- **Purpose**: Extract raw text from uploaded PDF resumes
- **Input**: PDF file (disk or bytes)
- **Output**: Structured text data with metadata
- **Status**: Already implemented, no changes needed

#### 3.2 Text Processing
**File**: `src/parser/text_processor.py`
- **Purpose**: Clean and normalize extracted resume text
- **Input**: Raw PDF text
- **Output**: Cleaned text ready for section parsing
- **Status**: Already implemented, no changes needed

#### 3.3 Section Parsing
**File**: `src/parser/section_parser.py`
- **Purpose**: Parse resume into sections (experience, skills, education, etc.)
- **Input**: Cleaned text
- **Output**: Structured resume sections with confidence scores
- **Status**: Already implemented, no changes needed

### 4. Job Analysis (Existing, No Changes)

#### 4.1 Job Requirements Extraction
**File**: `src/database/job_analyzer.py`
- **Purpose**: Extract structured requirements from job descriptions
- **Input**: Raw job description text
- **Output**: JobRequirements object (skills, experience, education)
- **Status**: Already implemented, works with ML models

### 5. ML-Powered Matching Engine (Major Refactor)

#### 5.1 Vector Store & Matching
**File**: `src/database/vector_store.py`
- **Purpose**: Orchestrate ML models for resume-job matching
- **Current State**: Uses hardcoded thresholds and basic similarity
- **New Implementation**: 
  - Load trained models from `models/embeddings/` and `models/scorers/`
  - Replace hardcoded logic with ML predictions from `src/ml/` components
  - Use fine-tuned embeddings for semantic similarity
  - Neural scoring for final match predictions

**Key Changes**:
- `match_resume_to_job()` now calls trained models from ML subdirectories
- Dynamic similarity thresholds learned from data
- Multi-dimensional scoring (overall match, ATS score, skill gaps)

### 6. User Interface

#### 6.1 Gradio Interface
**File**: `app.py`
- **Purpose**: Web interface for users to upload resumes and job descriptions
- **Input**: PDF resume file + job description text
- **Output**: Match analysis with recommendations and scores

#### 6.2 Configuration
**File**: `config.py`
- **Purpose**: Centralized configuration for model paths, hyperparameters
- **Contains**: Paths to models in subdirectories, training settings

#### 6.3 Logging
**File**: `src/utils/logging_config.py`
- **Purpose**: Consistent logging across all components

## Execution Flow Sequence

### Training Phase (Run Once)

1. **Generate Training Data**
   ```bash
   python scripts/generate_data.py
   # Uses: src/ml/data_generator/data_generator.py
   # Creates: data/training/synthetic_dataset.json
   ```

2. **Train All Models**
   ```bash
   python scripts/train_models.py
   # Uses: src/ml/classification/ and src/ml/scoring/
   # Creates: models/embeddings/ and models/scorers/
   ```

   Or train individually:
   ```bash
   python -m src.ml.classification.classification --train
   python -m src.ml.scoring.scoring --train
   ```

### Production Phase (Per User Request)

1. **User uploads resume PDF + job description via Gradio**
   ```bash
   python app.py
   # Launches web interface
   ```

2. **Processing Pipeline**:
   - **PDF Extraction**: `src/parser/pdf_extractor.py` → raw text
   - **Text Processing**: `src/parser/text_processor.py` → cleaned text  
   - **Section Parsing**: `src/parser/section_parser.py` → structured resume
   - **Job Analysis**: `src/database/job_analyzer.py` → structured job requirements
   - **ML Matching**: `src/database/vector_store.py` → loads and calls trained models:
     - Fine-tuned embeddings from `src/ml/classification/`
     - Neural scorers from `src/ml/scoring/`
   - **Results**: Match score, missing skills, ATS recommendations

## Import Structure for Refactored Components

```python
# Data generation
from src.ml.data_generator.data_generator import TechJobDataGenerator

# Classification
from src.ml.classification.classification import ResumeJobClassifier

# Scoring  
from src.ml.scoring.scoring import ResumeScorer

# Utilities
from src.ml.model_utils import ModelManager
```

## Key Differences from Original Architecture

### Before (Rule-Based)
- Hardcoded similarity thresholds (0.6, 0.8, 0.95)
- Fixed scoring weights (skills: 0.7, experience: 0.3)
- Simple keyword matching
- No learning capability

### After (ML-Driven with Refactored Structure)
- **Learned similarity thresholds** from training data
- **Neural networks discover** optimal feature combinations
- **Fine-tuned embeddings** understand resume/job context better
- **Continuous improvement** possible with new training data
- **Modular ML components** organized in logical subdirectories

## Model Files Structure
```
models/
├── embeddings/                 # From src/ml/classification/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── scorers/                    # From src/ml/scoring/
│   ├── match_scorer.pth        # Overall match predictor
│   ├── ats_scorer.pth         # ATS compatibility scorer
│   └── model_metadata.json    # Model configs and metrics
```

## ML Source Code Structure
```
src/ml/
├── data_generator/            # Synthetic data creation
│   ├── tech_taxonomy.py
│   ├── profile_generators.py
│   └── data_generator.py
├── classification/            # Fine-tuned embeddings
│   ├── embedding_models.py
│   ├── training_pipeline.py
│   └── classification.py
├── scoring/                   # Neural scoring models
│   ├── scoring_models.py
│   ├── feature_extractors.py
│   └── scoring.py
└── model_utils.py            # Shared utilities
```

## Data Files Structure
```
data/
├── training/
│   └── synthetic_dataset.json  # Generated by data_generator/
├── sample_resumes/             # Test PDFs (existing)
└── sample_jobs/               # Test job descriptions
    ├── software_engineer.txt
    ├── data_scientist.txt
    └── devops_engineer.txt
```

This refactored structure provides better organization while maintaining the same ML-driven functionality, making the codebase more maintainable and scalable.