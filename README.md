# 🏥 Healthcare Patient Analytics & Risk Analysis System

An end-to-end exploratory data analysis project analyzing 10,000 healthcare patient records to understand patient demographics, disease patterns, healthcare costs, risk scores, readmissions, correlations, outliers, and statistically significant relationships.

---

## 📌 Project Overview

Healthcare organizations generate large amounts of patient data.

This project analyzes patient-level healthcare data to identify:

- Patient demographic patterns
- Disease distribution
- Admission patterns
- Healthcare treatment costs
- Patient risk scores
- Readmission patterns
- Relationships between healthcare variables
- Statistical differences between patient groups
- Potential outliers in healthcare measurements

The project focuses on data analysis and statistical reasoning using Python.

---

## 🎯 Project Objectives

The main objectives are:

1. Understand the healthcare dataset
2. Clean and validate patient records
3. Analyze patient demographics
4. Study disease distribution
5. Analyze healthcare costs
6. Analyze patient risk scores
7. Study relationships between numerical variables
8. Detect statistical outliers
9. Perform hypothesis testing
10. Generate meaningful healthcare insights

---

## 📊 Dataset

The final cleaned dataset contains:

- 10,000 patient records
- 16 features

### Features

| Feature | Description |
|---|---|
| patient_id | Unique patient identifier |
| age | Patient age |
| gender | Patient gender |
| blood_pressure | Blood pressure measurement |
| heart_rate | Heart rate |
| bmi | Body Mass Index |
| smoking_status | Smoking status |
| diabetes | Diabetes indicator |
| hypertension | Hypertension indicator |
| disease | Primary disease category |
| admission_type | Admission type |
| length_of_stay | Hospital stay duration |
| medication_count | Number of medications |
| treatment_cost | Treatment cost |
| readmission | Readmission indicator |
| risk_score | Patient risk score |

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook
- Git
- GitHub

---

## 🔬 Analysis Performed

### 1. Data Understanding

- Dataset dimensions
- Data types
- Missing values
- Unique values
- Category distributions
- Descriptive statistics

### 2. Data Cleaning

- Missing-value handling
- Duplicate detection
- Patient ID validation
- Data consistency checks

### 3. Exploratory Data Analysis

- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Distribution analysis
- Group-based analysis

### 4. Patient Demographics

Analyzed:

- Age
- Gender
- Smoking status
- Admission type

### 5. Disease Analysis

Analyzed:

- Disease frequency
- Risk score by disease
- Treatment cost by disease
- Length of stay by disease

### 6. Healthcare Cost Analysis

Analyzed:

- Treatment cost distribution
- Treatment cost by disease
- Treatment cost by admission type
- Relationship between cost and risk
- Relationship between cost and length of stay

### 7. Correlation Analysis

Important relationships were investigated using Pearson correlation.

Examples:

- Age vs Risk Score
- Length of Stay vs Treatment Cost
- Risk Score vs Treatment Cost
- Medication Count vs Risk Score

### 8. Outlier Analysis

The IQR method was used to identify potential outliers in numerical variables.

### 9. Statistical Analysis

Statistical tests included:

- Welch's t-test
- One-way ANOVA
- Chi-square test

---

## 📈 Key Findings

The analysis found several measurable relationships.

### Age and Risk

Correlation between age and risk score:

```text
0.537