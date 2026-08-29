# full pipeline: Load -> EDA -> Features -> Train -> Explain
# python src/main.py
from data_loader import load_clean
from eda import run_eda
from train_model import main as train_main
from explain_shap import load_artifacts, global_importance, top_feature_ranking, explain_one_customer

# %%
def main():
    print("Step 1: Load and clean data")
    # load and clean the data
    df = load_clean()
    print(f"{len(df)} rows loaded, churn rate = {df['Churn'].mean():.1%}")

    # run eda
    print("\nStep 2: EDA")
    run_eda(df)

    #featuring
    print("\nStep 3+4: Feature engineering and model training")
    train_main()

    #SHAP
    print("\nStep 5: SHAP explanation")
    model, X_test = load_artifacts()
    shap_values = global_importance(model, X_test)
    top_feature_ranking(shap_values, X_test)
    explain_one_customer(model, X_test, row_idx=0)

    print("\nDone. Check the outputs flder for plot and metrics")

if __name__ == "__main__":
    main()