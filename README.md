Fake Certificate Detection using Machine Learning

Project Overview
This project is a beginner-level Machine Learning project that aims to detect fake vs legitimate certificates using structured certificate data.
The goal of this project is not just high accuracy, but to understand the complete ML workflow:
data preprocessing
feature engineering
model training
evaluation
understanding dataset limitations
This project serves as a baseline version (v1.0) and can be extended further in the future.


Problem Statement
Fake certificates can be created by manipulating:
certificate IDs
names
institution details
issue dates
The objective is to classify a certificate as:
Legitimate (1)
Fake (0)
based on available certificate information.


Dataset Description
The dataset is a synthetically generated dataset containing ~500 certificate records.
Columns used:
certificate_id
student_name
father_name
mother_name
institution_name
issue_date
is_legitimate (target label)

Note:The dataset contains fewer fake samples, which makes the classification problem imbalanced.


Feature Engineering
Instead of directly using raw text data, rule-based numerical features were engineered:
student_father_same
student_mother_same
father_mother_same
institution_match
(matches certificate ID prefix with institution name)
year_match
(certificate year vs issue date year)
These features help convert domain knowledge into model-usable signals.


Model Used
Random Forest Classifier
Chosen because:
handles non-linear patterns
works well with tabular data
beginner-friendly and interpretable
Handling Class Imbalance
class_weight="balanced" was used
Evaluation focused on precision, recall, and F1-score, not just accuracy


Model Evaluation
Accuracy: ~84–88%
Strong performance on legitimate certificates
Limited recall on fake certificates due to:
fewer fake samples
weak distinguishing patterns in synthetic data
Key Learning:
High accuracy does not always mean a good model — especially with imbalanced datasets.


Key Learnings from This Project
ML performance depends heavily on data quality
Feature engineering is often more important than model choice
Accuracy alone is misleading for imbalanced datasets
Sometimes rule-based systems outperform ML when data is weak
Understanding limitations is part of real ML development
