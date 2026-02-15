#  Sports vs Politics Text Classification
### Natural Language Understanding – Assignment (Problem 4)

This project implements a binary text classification system to distinguish between **Sports** and **Politics** documents using classical machine learning techniques.

The goal is to evaluate how effectively traditional machine learning models perform on high-dimensional textual data when combined with TF-IDF feature representation.

---

##  1. Problem Statement

Given a textual document, classify it into one of two categories:

-  Sports  
-  Politics  

The task involves:

- Feature extraction using NLP techniques
- Training multiple machine learning models
- Comparing their quantitative performance
- Analyzing results using evaluation metrics and confusion matrices

---

## 📊 2. Dataset

The dataset is derived from the **20 Newsgroups dataset**, a benchmark dataset widely used for text classification.

### Selected Categories:
- `rec.sport.baseball` → Sports
- `talk.politics.misc` → Politics

### Dataset Statistics:
- Total Documents: **1062**
- Training Samples: **849**
- Testing Samples: **213**
- Data Type: Informal forum-style text

Headers, footers, and quoted replies were removed to reduce metadata bias and improve content-based learning.

---

##  3. Feature Engineering

Raw text cannot be directly processed by machine learning models. Therefore, the documents were converted into numerical feature vectors using:

###  TF-IDF (Term Frequency–Inverse Document Frequency)

Preprocessing steps:
- Lowercasing
- Stopword removal
- Unigram representation
- Vocabulary limited to top 5000 features

TF-IDF helps emphasize discriminative words while reducing the impact of common terms.

---

##  4. Machine Learning Models Implemented

Three classical models were trained and evaluated:

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Linear Support Vector Machine (SVM)**

These models were selected because they are highly effective for high-dimensional sparse text data.

---

##  5. Experimental Results

| Model | Accuracy |
|--------|----------|
| Multinomial Naive Bayes | **99.53%** |
| Logistic Regression | 98.59% |
| Linear SVM | 99.06% |

### Key Observation

Multinomial Naive Bayes achieved the highest accuracy.  
This aligns with literature showing its strength in high-dimensional sparse feature spaces common in NLP tasks.

---

##  6. Confusion Matrices

The confusion matrices show minimal misclassification across both categories.

- Naive Bayes Confusion Matrix
- Logistic Regression Confusion Matrix
- Linear SVM Confusion Matrix

These visualizations confirm strong generalization performance.

---

## 7. How to Run the Project

### Step 1: Install Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn 
