# Mental_health_analysis
 
# Mental Health Prediction Tool Documentation

## Overview
The Mental Health Prediction Tool is designed to assess an individual's likelihood of experiencing depression or anxiety based on survey data and machine learning models. The tool uses logistic regression, random forest, and XGBoost models to make predictions and provide actionable recommendations.

## Dataset Preprocessing Steps

1. **Loading Data**:
   - The dataset (`depression_anxiety_data.csv`) is loaded and initial exploratory analysis is performed.
   - Another dataset (`survey.csv`) is cleaned and combined with the main dataset.

2. **Data Cleaning**:
   - Dropped missing values and unnecessary columns like `id`.
   - Encoded categorical variables (e.g., `gender`, `who_bmi`, `depression_severity`, `anxiety_severity`).
   - Converted binary categorical variables to integers.

3. **Feature Engineering**:
   - Created `treatment_status` as a binary target variable.
   - Created `severity_index` by summing `phq_score` and `gad_score`.
   - Created `diagnosis_class` based on `depression_diagnosis` and `anxiety_diagnosis`.
   - Applied missing value imputation using Iterative Imputer.

4. **Data Balancing**:
   - Used resampling techniques to balance the classes before training.

## Model Selection Rationale

Three models were trained and evaluated:

1. **Logistic Regression**:
   - Simple and interpretable model for binary classification.
2. **Random Forest Classifier**:
   - Robust model with built-in feature importance and better handling of imbalanced data.
3. **XGBoost Classifier**:
   - High-performance gradient boosting model with class-weighting to handle imbalance.

Models were evaluated using:
   - **Accuracy**
   - **ROC-AUC Score**
   - **F1-Score**

The best model (Random Forest) was saved as `mental_health_model.pkl` for inference.

## How to Run the Inference Script

### Prerequisites

Ensure you have the required dependencies installed:
```sh
pip install pandas numpy scikit-learn joblib shap xgboost streamlit matplotlib seaborn
```
### Running the inference script
```sh
python3 predict_mental_health.py --phq 12 --gad 8 --age 25 --gender 1
```

### Running the Model
1. Load the trained model:
   ```python
   import joblib
   model = joblib.load('models/mental_health_model.pkl')
   ```
2. Prepare input data:
   ```python
   input_data = {'phq_score': 12, 'gad_score': 10, 'age': 25, 'gender': 1}
   ```
3. Run the prediction:
   ```python
   prediction = model.predict([input_data])
   print(prediction)
   ```

## UI/CLI Usage Instructions

### Running the Streamlit App
```sh
streamlit run mental_health_ui.py
```

### Using the UI
1. Enter PHQ-9 and GAD-7 scores.
2. Select age and gender.
3. Click **Predict** to get diagnosis, severity index, and recommendations.

### Example Output
```
Diagnosis: Comorbid Depression & Anxiety
Severity Index: 22/48
Recommendations:
- Schedule clinical evaluation
- Begin cognitive behavioral therapy (CBT) exercises
```

