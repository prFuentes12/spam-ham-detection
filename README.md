# 📧 Spam vs Ham Classifier

## 🎯 Project Objective

The goal of this project is to build a classifier that can distinguish between legitimate (ham) and spam emails using the well-known ** Spam dataset**. This is a common and practical problem in text classification with significant applications in digital security, fraud prevention, and improving user experience in email systems.

---

## 🧠 Workflow Overview

1. **Data Loading and Cleaning**:
   - The dataset was downloaded from Kaggle using `kagglehub`.
   - Emails from folders `enron1` through `enron6` were loaded and merged, labeling `ham` as `0` and `spam` as `1`.
   - **600 duplicate rows** were removed and the dataset was checked for **missing values** to ensure data quality.

2. **Text Preprocessing**:
   - Converted text to lowercase.
   - Tokenized using `TreebankWordTokenizer`.
   - Removed punctuation and English stopwords.
   - Cleaned text was saved into a new column.

3. **Text Vectorization**:
   - `TfidfVectorizer` was applied with both unigrams and bigrams (`ngram_range=(1,2)`).
   - Several configurations were tested. The best one was `max_features=10000`.

4. **Models Evaluated**:
   - **Multinomial Naive Bayes**
   - **Logistic Regression**
   - **Support Vector Machine (SVM)**

5. **Model Evaluation**:
   - Data was split into **train/test sets (80/20)**.
   - Metrics calculated: Accuracy, Precision, Recall, F1-score.
   - **Learning curves** were plotted to detect overfitting and assess model generalization.

---

## 📊 Results & Metrics

| Model               | Accuracy | Precision | Recall | F1-score | Overfitting |
|---------------------|----------|-----------|--------|----------|-------------|
| Naive Bayes         | 0.9864   | 0.9833    | 0.9903 | 0.9868   | No          |
| Logistic Regression | 0.9880   | 0.9800    | 0.9971 | 0.9885   | No          |
| SVM                 | **0.9914** | **0.9872** | **0.9962** | **0.9916** | No          |

- **SVM performed the best**, achieving the highest scores without signs of overfitting.
- **Learning curves** show that all three models generalize well (training and validation curves are close).
- **Naive Bayes** is a fast and lightweight model but slightly less accurate.
- **Logistic Regression** offers an excellent balance of performance and simplicity.

---

## 📌 Conclusions

- **Text preprocessing and vectorization** are crucial for good performance in text classification.
- Classic models like **Naive Bayes, Logistic Regression, and SVM** can perform remarkably well with proper setup.
- **SVM is recommended** when accuracy is critical and computational power is available.

This project showcases the full pipeline of spam detection using real-world email data and solid machine learning techniques.
