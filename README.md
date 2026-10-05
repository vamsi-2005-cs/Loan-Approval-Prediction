# Loan Approval Prediction

A machine learning application that predicts loan approval using applicant and financial information. The project includes data preprocessing, classification model evaluation, and an interactive Streamlit interface for predictions and model analysis.

## Overview

The project evaluates multiple machine learning classification algorithms and identifies the best-performing model using 5-fold cross-validation.

### Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Linear Support Vector Machine (SVM)

**Best Model:** Linear SVM  
**Accuracy:** 80.78%

## Features

- Loan approval prediction
- Machine learning model comparison
- Dataset exploration
- Prediction confidence visualization
- Interactive Streamlit interface

## Dataset

The project uses a dataset containing **614 records and 13 attributes** related to applicant and loan information.

Data preprocessing includes handling missing values and encoding categorical features.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Google Colab

## Project Structure

```text
Loan-Approval-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── run.bat
├── run.sh
├── .gitignore
│
├── data/
│   ├── README.md
│   └── train.csv
│
└── notebook/
    └── Loan_Approval_Prediction.ipynb

## Installation

```bash
git clone https://github.com/vamsi-2005-cs/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Windows users can also run:

```text
run.bat
```

## Development

Machine learning development, preprocessing, experimentation, and model evaluation were performed using **Google Colab**.

[View the Google Colab Notebook](https://colab.research.google.com/drive/1Q8xuNgbcz3d0S2NR9v-xw4hRoCXHIfuj)

## Results

Five classification models were evaluated using 5-fold cross-validation. **Linear SVM achieved the highest accuracy of 80.78%** among the evaluated models.

## Author

**Javvadi Venkata Vamsi**

* GitHub: https://github.com/vamsi-2005-cs/
* LinkedIn: https://www.linkedin.com/in/venkatavamsijavvadi/
