# ML-Driven Resume Analyzer

An intelligent resume optimization tool that uses fine-tuned machine learning models to analyze resume-job description matches and provide ATS (Applicant Tracking System) optimization recommendations.

## Overview

This project transforms traditional rule-based resume analysis into a sophisticated ML-driven system that:

- **Generates synthetic training data** for resume-job matching scenarios
- **Fine-tunes sentence transformers** on domain-specific resume-job pairs  
- **Trains neural scoring models** to predict match quality, ATS compatibility, and skill gaps
- **Provides actionable recommendations** for resume improvement

## Architecture

```
User Input → PDF Processing → ML Analysis → Scoring & Recommendations → Gradio Interface
```

The system replaces hardcoded similarity thresholds and fixed scoring weights with learned models that understand nuanced resume-job matching patterns.

## Key Features

- **Synthetic Data Generation**: Creates realistic resume-job pairs with controlled match characteristics
- **Fine-tuned Embeddings**: Domain-specific sentence transformers optimized for resume analysis
- **Neural Scoring**: Multiple specialized models for overall matching, ATS compatibility, and gap analysis
- **Feature Engineering**: 80+ features covering text quality, keywords, structure, experience, and ATS factors
- **Production Ready**: Complete training pipeline with model persistence and web interface

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd optimized-rag-ml-resume-booster
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python scripts/generate_data.py
```

### 3. Train Models

```bash
python scripts/train_models.py
```

### 4. Launch Web Interface

```bash
python app.py
```

## Project Structure

```
optimized-rag-ml-resume-booster/
├── src/ml/                    # ML components
│   ├── data_generator/        # Synthetic data creation
│   ├── classification/        # Fine-tuned embeddings
│   └── scoring/              # Neural scoring models
├── src/parser/               # Text processing pipeline
├── src/database/             # Analysis engines
├── data/                     # Training data and samples
├── models/                   # Trained model artifacts
└── tests/                    # Test suites
```

## Training Pipeline

### Phase 1: Data Generation
Creates synthetic resume-job pairs with ground truth labels:
- Tech skill taxonomies for realistic job descriptions
- Resume templates with controlled skill overlap
- Balanced datasets across experience levels and job types

### Phase 2: Model Training
- **Classification Models**: Fine-tune sentence transformers on resume-job similarity
- **Scoring Models**: Train neural networks for match prediction, ATS scoring, and gap analysis

### Phase 3: Production Deployment
- Load trained models into production pipeline
- Replace rule-based matching with ML predictions
- Serve through web interface

## ML Models

### Classification Models
- **Siamese Networks**: Learn semantic similarity between resumes and jobs
- **Multi-task Learning**: Simultaneous training on multiple objectives
- **Contrastive Learning**: Push similar pairs together, dissimilar pairs apart

### Scoring Models
- **Match Scorer**: Overall compatibility prediction
- **ATS Scorer**: Applicant tracking system optimization
- **Skill Gap Analyzer**: Missing skills identification with attention mechanisms
- **Experience Predictor**: Years and seniority level matching
- **Ensemble Methods**: Combine multiple models for robust predictions

## Features Engineered

- **Text Quality**: Word count, readability, formatting consistency
- **Keyword Matching**: Technical skills, exact phrases, TF-IDF similarity
- **Structure Analysis**: Resume sections, bullet points, contact information
- **Experience Metrics**: Years calculation, seniority indicators, career progression
- **Similarity Measures**: Cosine similarity, embedding distances, length ratios
- **ATS Compatibility**: Format analysis, keyword density, parsing friendliness

## Performance

The ML-driven approach significantly outperforms rule-based baselines:

- **Learned thresholds** adapt to data patterns vs fixed cutoffs
- **Neural feature combination** discovers optimal weightings vs manual tuning  
- **Domain-specific embeddings** understand resume terminology vs generic models
- **Multi-task learning** leverages related objectives vs isolated training

## Technology Stack

- **ML Frameworks**: PyTorch, sentence-transformers, scikit-learn
- **NLP**: SpaCy, transformers, ChromaDB vector storage
- **Web Interface**: Gradio for interactive resume analysis
- **Data Processing**: Pandas, NumPy for feature engineering
- **Document Processing**: PyMuPDF for PDF text extraction

## Development Workflow

### Training New Models
```bash
# Generate fresh training data
python -m src.ml.data_generator.data_generator

# Train classification models
python -m src.ml.classification.classification --data data/training/synthetic_dataset.json --save-path models/embeddings/

# Train scoring models  
python -m src.ml.scoring.scoring --data data/training/synthetic_dataset.json --save-path models/scorers/
```

### Testing Components
```bash
# Test individual components
python -m pytest tests/test_ml/test_data_generator/
python -m pytest tests/test_ml/test_classification/
python -m pytest tests/test_ml/test_scoring/

# Test full pipeline
python -m pytest tests/integration/
```

## Configuration

Key settings in `config.py`:
- Model architectures and hyperparameters
- Training data generation parameters
- File paths for models and data
- Logging and evaluation settings

## API Usage

### Programmatic Access
```python
from src.ml.classification.classification import ResumeJobClassifier
from src.ml.scoring.scoring import ResumeScorer

# Load trained models
classifier = ResumeJobClassifier('models/embeddings/')
scorer = ResumeScorer('models/embeddings/')

# Analyze resume-job pair
results = scorer.score_resume_job_pair(resume_text, job_text)
print(f"Match Score: {results['match_score']:.3f}")
print(f"ATS Score: {results['ats_score']:.3f}")
```

### Web Interface
Upload resume PDF and paste job description through the Gradio interface for interactive analysis with visualizations and detailed recommendations.

## Contributing

1. Follow the established directory structure for new ML components
2. Add comprehensive tests for new functionality  
3. Update documentation for API changes
4. Ensure models can be trained end-to-end with provided scripts

## Author
Creator: Yvonne Hong

## Contact Me 
Linkedin: https://www.linkedin.com/in/yvnnhong
Personal website: https://yvnnhong.github.io/yvonnehong/

## Acknowledgments
Built on top of modern ML libraries including PyTorch, Hugging Face Transformers, and sentence-transformers for state-of-the-art natural language processing capabilities.