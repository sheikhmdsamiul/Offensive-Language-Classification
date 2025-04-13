# Toxic Comment Classification

## Project Overview
This project aims to detect toxic comments in online text data using machine learning. The model classifies text into toxic/non-toxic categories and identifies specific toxicity types (abusive, vulgar, menace, etc.). This is particularly useful for content moderation in online platforms.

## Dataset Description
- **Info**: The dataset consists of 4 files, which are train.csv, test.csv, validation.csv and test_labels.csv
- **Size**: number of Rows and Columns of TrainSet...(23473, 8)
- **Features**:
  - `feedback_text`: Raw comment text
  - `toxic`, `abusive`, `vulgar`, `menace`, `offense`, `bigotry`: Binary labels (0/1)
- **Class Distribution** (Training):
  - Non-toxic: 90.6%
  - Toxic: 9.4%
    
## Model Implementation

### EDA & Preprocessing
1. Visualizing and understanding the data
2. Text cleaning (lowercase, URL removal, special characters)
3. Tokenization and lemmatization
4. Language detection and translation (non-English → English)
5. Undersampling to address class imbalance

### Models Implemented
1. **Logistic Regression (TF-IDF)**
   - Baseline model with n-gram features
   - Best Accuracy : 0.7952

2. **Random Forest**
   - Handled non-linearity better but overfit
   - Validation Accuracy : 0.7405

3. **Bidirectional LSTM with Attention**
   - Architecture:
     ```python
     Input → Embedding → BiLSTM → Attention → Dense → Output
     ```
   - Key hyperparameters:
     - Embedding dim: 128
     - LSTM units: 64
     - Dropout: 0.4
     - Class weights: {0:1, 1:5}

### Optimization Techniques
- Class reweighting
- Focal loss (γ=2.0)
- Learning rate scheduling
- Threshold tuning (optimal: 0.3-0.9)

## How to Run

### Requirements
```bash
pip install -r requirements.txt
