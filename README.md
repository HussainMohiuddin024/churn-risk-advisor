📊 Churn Risk Advisor

Customer Churn Prediction using XGBoost and Streamlit

📌 Project Overview

Churn Risk Advisor is a machine learning application that estimates the probability of customer churn using an XGBoost classifier. It helps users explore customer risk, compare what-if scenarios, and score multiple customers using a CSV file.

The application is built with Python, Pandas, Scikit-learn, XGBoost, Joblib, and Streamlit.

🎯 Objectives

* Predict customer churn probability.
* Demonstrate and prevent training-serving encoding inconsistencies.
* Provide customer risk labels using a configurable threshold.
* Compare what-if scenarios, including contract and payment method changes.
* Support batch predictions from CSV files.
* Package the trained model for application use.

✨ Features

* Single Customer Prediction: Enter customer details and estimate churn probability.
* Risk Assessment: View predicted probability and compare it with the selected risk threshold.
* What-if Analysis: Compare one-year contracts, two-year contracts, and automatic credit card payments.
* Batch Scoring: Upload a CSV file to score multiple customers.
* Download Results: Export scored customer data as a CSV file.
* Model Information: View model details and evaluation metrics.

🤖 Model Performance

Metric	Result
Model	XGBClassifier
Feature Columns	30
Cross-validation AUC	0.8502
Cross-validation AUC Standard Deviation	0.0117
Test AUC	0.8483
Model Version	1.0

These metrics are from the project’s recorded model evaluation. AUC measures how well the model distinguishes between customers who churn and those who do not.

🛠️ Technologies Used

* Python
* XGBoost
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

📁 Project Files

* app.py — Streamlit web application.
* churn_model.joblib — Trained machine learning model.
* model_meta.json — Model metadata and feature information.
* requirements.txt — Python dependencies.
* sample_customers.csv — Sample customer data for batch scoring.

🌐 Live Application

Open the deployed application here:

Churn Risk Advisor — Live App:https://churn-risk-advisor-s8pr7und23lxk4wtd3l3z4.streamlit.app/

The application is deployed using Streamlit Community Cloud and connected to the GitHub repository.

🚀 Run Locally

1. Clone the repository

git clone https://github.com/HussainMohiuddin024/churn-risk-advisor.git

2. Open the project folder

cd churn-risk-advisor

3. Install dependencies

python -m pip install -r requirements.txt

4. Start the application

python -m streamlit run app.py

The application should open in your browser. If it does not, visit:

http://localhost:8501

📈 How to Use

1. Open the Single Customer tab.
2. Enter the customer information.
3. Click Predict churn risk.
4. Review the predicted churn probability and risk assessment.
5. Compare alternative scenarios in the what-if analysis.
6. Open Batch Scoring to upload sample_customers.csv.
7. Review the scored customer data and download the results if required.
8. Open Model Information to review model details and evaluation metrics.

⚠️ Limitations

* Predictions are estimates and are not guaranteed outcomes.
* Total charges in the single-customer form are approximated using tenure multiplied by monthly charges.
* Model performance may vary on new or unseen customer data.
* Risk labels depend on the selected threshold and should support, not replace, informed decisions.
* The model is intended for educational and demonstration purposes, not as the sole basis for customer-related decisions.
  
Application Screenshots:

1.Single_Customer

Shows the predicted churn probability, risk band, decision threshold, and what-if analysis.

2.Batch_Scoring

Shows batch prediction results for sample customers.

3. Model_Info

Shows model details, number of features, cross-validation AUC, test AUC, and model version.
👨‍💻 Author

Hussain Mohiuddin

GitHub: HussainMohiuddin024

📄 License

This project was developed for educational and learning purposes.
