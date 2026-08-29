# Telco Customer Churn

Hi! This is a little project where I dig into a telecom company's customer data to understand why customers leave (churn) and what might help keep them around.

## What's in here

I start with the raw customer data, clean it up, explore it, and then train a couple of models to predict who's likely to churn. Along the way I use SHAP to explain what's actually driving those predictions, and I pull it all together into a set of retention recommendations at the end.

## The data

The dataset lives in the `Data` folder as `WA_Fn-UseC_-Telco-Customer-Churn.csv`. It has one row per customer, with things like contract type, payment method, internet service, tenure, and whether they churned.

## How the pipeline works

1. `src/data_loader.py` loads the raw CSV and cleans it up (fixing data types, converting Churn to 0/1, dropping the customer ID).
2. `src/eda.py` explores churn rates across different customer segments and saves the charts to outputs/.
3. `src/feature_engineering.py` builds a few extra features, like tenure buckets and a count of add on services, and one hot encodes everything for modeling.
4. `src/train_model.py` trains a Logistic Regression baseline and an XGBoost model, then saves both the trained model and evaluation results to outputs/.
5. `src/explain_shap.py` uses SHAP to explain what the XGBoost model is picking up on, both overall and for individual customers.
6. `src/main.py` ties the pipeline together.

## Getting started

First, install the dependencies, and then you can run the scipts!

    pip install -r requirements.txt

... for example:

    python src/train_model.py
^^ This is assuming you are in the main folder (`Telco-Customer-Churn`)

Everything gets saved to the outputs folder, including charts, the trained model, and evaluation numbers.

## Results

Check out `outputs/recommendations.md` I wrote for a writeup of what the data and models found, along with some ideas for what the company could actually do about churn.

Thanks for stopping by!
