import pandas as pd
import numpy as np
import shap
from catboost import CatBoostRegressor


class ValuationEngine:
    def __init__(self, model_path: str = "model.cbm"):
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, df: pd.DataFrame) -> dict:
        log_pred = self.model.predict(df)[0]
        price_m2 = np.expm1(log_pred)
        shap_values = self.explainer.shap_values(df)[0]

        impacts = []
        for name, val in zip(df.columns, shap_values):
            if abs(val) > 0.01:
                impacts.append({"feature": name, "value": float(val)})

        return {
            "total_price": int(price_m2 * df['TotalArea'].iloc[0]),
            "price_m2": int(price_m2),
            "impacts": sorted(impacts, key=lambda x: abs(x['value']), reverse=True)[:5]
        }