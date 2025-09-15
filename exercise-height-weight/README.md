# In-Class Height-Weight Analysis with Classification & Regression Models
This exercise explores **predicting/determining BMI Category from height/weight** (Classification) and **predicting weight from height** (Regression) 

## Setup
1. Clone/Download this repo.
2. Create a Python virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
3. Install dependencies/requirements:
   pip install -r requirements.txt

   ### Requirements
    pandas
    scikit-learn
    matplotlib
   
5. Data Prep
   python code/data-prep/dataPrep.py
   -Strips whitespace from headers, renames the columns to be easier used, converts values to numeric and removes invalid rows.
6. Classifier Model
   python code/custom-classifier-model/customClassifierModel.py
   -Logistic Regression, Random Forrest Classifier
   -Gives BMI Category based on given heights (50,100,150,200,250cm) with metric reporting (Macro F1 Measure/score, ROC Curve, AUC)
7. Regression Model
   python code/custom-regression-model/customRegressionModel.py
   -Linear Regression, Random Forrest Regressor
   -Gives weight predictions at given heights (50,100,150,200,250cm) with metric reporting (R^2, MAE, RMSE)

## Notes
1. Classification classifies data into 4 distinct BMI Categories,
   (C1 = Underweight
    C2 = Healthy
    C3 = Overweight
    C4 = Obese)
2. Regression is a continous weight prediction based on height/weight data.
3. Linear Regression provides more smooth, continous predictions (although more inaccurate) while Random Forrest provides more step-like predictions
   which could possibly change with the introduction of more features (sex, age, etc.) to help the model ensemble differently.  
