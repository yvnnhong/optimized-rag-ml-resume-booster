# ML-Driven Resume Analyzer - Execution Flow (Updated)

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
**File**: `src/models/data_generator.py`
- **Purpose**: Generate synthetic resume-job pairs with ground truth labels
- **Output**: Training dataset with match scores (0-1) and binary labels
- **Execution**: Run once to create training data for ML models
- **Key Functions**: 
  - `generate_training_dataset()` - Creates labeled examples
  - Tech skill taxonomies for realistic job descriptions
  - Resume templates with controlled skill overlap percentages

### 2. Model Training Phase

#### 2.1 Fine-tuned Embeddings
**File**: `src/models/classification.py`
- **Purpose**: Fine-tune sentence transformers on domain-specific resume-job pairs
- **Input**: Synthetic training dataset from data_generator
- **Output**: Custom embedding model saved to `models/embeddings/`
- **Key Components**:
  - `ResumeJobClassifier` class with fine-tuning pipeline
  - Custom training loop using sentence-transformers library
  - Evaluation metrics and model persistence

#### 2.2 Neural Scoring Models
**File**: `src/models/scoring.py`
- **Purpose**: PyTorch neural networks that predict match scores from features
- **Input**: Embeddings + extracted features (skill overlap, experience gaps, etc.)
- **Output**: Trained scoring models saved to `models/scorers/`
- **Key Components**:
  - `MatchScorer` - Multi-layer perceptron for overall matching
  - `ATSScorer` - Specialized model for ATS compatibility
  - Feature extraction from text similarities and metadata

#### 2.3 Model Utilities
**File**: `src/models/model_utils.py`
- **Purpose**: Loading, saving, and managing trained models
- **Functions**: Model persistence, version management, inference helpers

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
  - Replace hardcoded logic with ML predictions
  - Use fine-tuned embeddings for semantic similarity
  - Neural scoring for final match predictions

**Key Changes**:
- `match_resume_to_job()` now calls ML models instead of rules
- Dynamic similarity thresholds learned from data
- Multi-dimensional scoring (overall match, ATS score, skill gaps)

### 6. User Interface

#### 6.1 Gradio Interface
**File**: `app.py`
- **Purpose**: Web interface for users to upload resumes and job descriptions
- **Input**: PDF resume file + job description text
- **Output**: Match analysis with recommendations and scores
- **Components**:
  - File upload for resume PDF
  - Text input for job description
  - Results display with visualizations

#### 6.2 Configuration
**File**: `config.py`
- **Purpose**: Centralized configuration for model paths, hyperparameters
- **Contains**: Model file paths, training settings, API configurations

#### 6.3 Logging
**File**: `src/utils/logging_config.py`
- **Purpose**: Consistent logging across all components

## Execution Flow Sequence

### Training Phase (Run Once)

1. **Generate Training Data**
   ```bash
   python scripts/generate_data.py
   # Creates: data/training/synthetic_dataset.json
   ```

2. **Train All Models**
   ```bash
   python scripts/train_models.py
   # Creates: models/embeddings/ and models/scorers/
   ```

   Or train individually:
   ```bash
   python -m src.models.classification --train
   python -m src.models.scoring --train
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
   - **ML Matching**: `src/database/vector_store.py` → calls trained models:
     - Fine-tuned embeddings for semantic similarity
     - Neural scorers for final match prediction
   - **Results**: Match score, missing skills, ATS recommendations

## Key Differences from Original Architecture (see https://github.com/yvnnhong/rag-nlp-resume-booster)

### Before (Rule-Based)
- Hardcoded similarity thresholds (0.6, 0.8, 0.95)
- Fixed scoring weights (skills: 0.7, experience: 0.3)
- Simple keyword matching
- No learning capability

### After (ML-Driven)
- **Learned similarity thresholds** from training data
- **Neural networks discover** optimal feature combinations
- **Fine-tuned embeddings** understand resume/job context better
- **Continuous improvement** possible with new training data

## Model Files Structure
```
models/
├── embeddings/                 # Fine-tuned sentence transformers
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── scorers/                    # PyTorch neural networks
│   ├── match_scorer.pth        # Overall match predictor
│   ├── ats_scorer.pth         # ATS compatibility scorer
│   └── model_metadata.json    # Model configs and metrics
```

## Data Files Structure
```
data/
├── training/
│   └── synthetic_dataset.json  # Generated training examples
├── sample_resumes/             # Test PDFs (existing)
└── sample_jobs/               # Test job descriptions
    ├── software_engineer.txt
    ├── data_scientist.txt
    └── devops_engineer.txt
```

## Dependencies
```txt
# Existing
PyMuPDF, chromadb, sentence-transformers, numpy

# NEW ML dependencies
torch>=2.0.0
scikit-learn>=1.3.0
transformers>=4.30.0
gradio>=3.40.0
pandas>=2.0.0
matplotlib>=3.7.0
```
