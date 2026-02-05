import pandas as pd
import numpy as np
import re
import json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from typing import List, Any, Dict


class DataPreprocessor:
    """
    SOLID Data Preparation with Data Quality Safeguard (0.0 to NaN conversion).
    """

    def __init__(self):
        self.cat_features: List[str] = []
        self.district_median_map: Dict[str, float] = {}
        self.global_median: float = 0.0

    @staticmethod
    def _parse_area(value: Any) -> float:
        if pd.isna(value) or str(value).lower() in ['<null>', 'nan', '']: return np.nan
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(value).replace(',', '.'))
        return sum(float(n) for n in nums) if nums else np.nan

    @staticmethod
    def _parse_rooms(value: Any) -> int:
        val = str(value).lower()
        if 'студия' in val: return 1
        res = re.findall(r'\d+', val)
        return int(res[0]) if res else 1

    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df_out = df.copy()

        if is_train:
            df_out = df_out[(df_out['PricePerMeter'] > 100000) & (df_out['PricePerMeter'] < 1500000)]

        impossible_zeros = [
            'LivingArea', 'KitchenArea', 'HallwayArea', 'CeilingHeight',
            'AreaWithoutBalcony', 'TotalArea'
        ]
        for col in impossible_zeros:
            if col in df_out.columns:
                df_out[col] = df_out[col].replace(0, np.nan)

        df_out['bath_val'] = df_out['BathroomArea'].apply(self._parse_area)
        df_out['balc_val'] = df_out['BalconyArea'].apply(self._parse_area)

        df_out['rooms_count'] = df_out['PropertyType'].apply(self._parse_rooms)
        df_out['is_euro'] = df_out['PropertyType'].str.contains('Евро', case=False).astype(int)

        df_out['area_per_room'] = df_out['TotalArea'] / df_out['rooms_count']
        if 'KitchenArea' in df_out.columns:
            df_out['kitchen_ratio'] = df_out['KitchenArea'] / df_out['TotalArea']
        if 'LivingArea' in df_out.columns:
            df_out['living_ratio'] = df_out['LivingArea'] / df_out['TotalArea']

        df_out['HandoverDate'] = df_out['HandoverDate'].replace('Сдан', '01.01.2021')
        handover_dt = pd.to_datetime(df_out['HandoverDate'], format='%d.%m.%Y', errors='coerce')
        df_out['h_year'] = handover_dt.dt.year.fillna(2025).astype(int)
        df_out['months_to_handover'] = ((df_out['h_year'] - 2024) * 12 + handover_dt.dt.month.fillna(1)).astype(int)

        if 'Floor' in df_out.columns and 'FloorsTotal' in df_out.columns:
            df_out['floor_rel'] = df_out['Floor'] / df_out['FloorsTotal']
            df_out['is_first_floor'] = (df_out['Floor'] == 1).astype(int)
            df_out['is_last_floor'] = (df_out['Floor'] == df_out['FloorsTotal']).astype(int)

        if 'CeilingHeight' in df_out.columns:
            df_out['is_high_ceiling'] = (df_out['CeilingHeight'] > 3.0).astype(float)
            df_out['premium_score'] = df_out['CeilingHeight'] * np.log1p(df_out['TotalArea'].fillna(0))

        if is_train:
            self.district_median_map = df_out.groupby('District')['PricePerMeter'].median().to_dict()
            self.global_median = np.median(list(self.district_median_map.values()))

        df_out['district_level'] = df_out['District'].map(self.district_median_map).fillna(self.global_median)

        drop_cols = ['TotalCost', 'Address', 'Number', 'PIBNumber', 'Layout', 'LayoutType',
                     'Phase', 'Section', 'Plot', 'HouseNumber', 'Axis', 'InstallmentUntil',
                     'BathroomArea', 'BalconyArea', 'HandoverDate', 'Complex_encoded']
        df_out.drop(columns=[c for c in drop_cols if c in df_out.columns], inplace=True)
        df_out.drop_duplicates(inplace=True)

        self.cat_features = df_out.select_dtypes(include=['object']).columns.tolist()
        for col in self.cat_features:
            df_out[col] = df_out[col].fillna('Unknown').astype(str)

        return df_out


class CatBoostModelManager:
    """
    Handles model lifecycle: training, validation, and serialization.
    """
    def __init__(self, cat_features: List[str]):
        self.cat_features = cat_features
        self.params = {
            'iterations': 3000,
            'learning_rate': 0.05,
            'depth': 9,
            'l2_leaf_reg': 3,
            'random_strength': 0.8,
            'bagging_temperature': 0.7,
            'loss_function': 'RMSE',
            'eval_metric': 'MAE',
            'random_seed': 42,
            'verbose': 500,
            'early_stopping_rounds': 200
        }
        self.model = None

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
        gkf = GroupKFold(n_splits=5)
        metrics = []

        for i, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            m = CatBoostRegressor(**self.params)
            m.fit(Pool(X_tr, y_tr, cat_features=self.cat_features),
                  eval_set=Pool(X_val, y_val, cat_features=self.cat_features))

            preds = np.expm1(m.predict(X_val))
            actual = np.expm1(y_val)

            metrics.append({
                'MAPE': mean_absolute_percentage_error(actual, preds),
                'MAE': mean_absolute_error(actual, preds),
                'R2': r2_score(actual, preds)
            })
            print(f"Fold {i + 1} metrics calculated.")

        return pd.DataFrame(metrics).mean()

    def train_and_save(self, X: pd.DataFrame, y: pd.Series, path: str = "model.cbm"):
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(Pool(X, y, cat_features=self.cat_features), verbose=500)
        self.model.save_model(path)
        print(f"Model saved to {path}")


class Visualizer:
    @staticmethod
    def plot_all(model, X: pd.DataFrame, y: pd.Series):
        """Generates performance and interpretability reports."""
        preds_log = model.predict(X)
        preds = np.expm1(preds_log)
        actual = np.expm1(y)

        plt.figure(figsize=(20, 6))

        # 1. Regression Plot
        plt.subplot(1, 3, 1)
        sns.regplot(x=actual, y=preds, scatter_kws={'alpha': 0.2, 's': 5}, line_kws={'color': 'red'})
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual Price (RUB)")
        plt.ylabel("Predicted Price (RUB)")

        # 2. Residuals Distribution
        plt.subplot(1, 3, 2)
        sns.histplot(actual - preds, kde=True, color='teal')
        plt.title("Error (Residuals) Distribution")
        plt.xlabel("Prediction Error")

        # 3. SHAP Feature Importance
        plt.subplot(1, 3, 3)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 1. Pipeline execution
    df_raw = pd.read_csv('../data/case_data.csv', low_memory=False)

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.transform(df_raw)

    X = df_clean.drop(columns=['PricePerMeter'])
    y = np.log1p(df_clean['PricePerMeter'])
    groups = df_raw.loc[df_clean.index, 'Complex_encoded']

    # 2. Model Lifecycle
    manager = CatBoostModelManager(preprocessor.cat_features)
    final_stats = manager.cross_validate(X, y, groups)
    print("\nCROSS-VALIDATION STATS:")
    print(final_stats)

    manager.train_and_save(X, y, "model.cbm")

    # 3. Visualization
    print("\nGenerating Visualizations...")
    Visualizer.plot_all(manager.model, X, y)

    # 4. Artifact Metadata
    with open("../model/inference_meta.json", "w") as f:
        json.dump({
            "cat_features": preprocessor.cat_features,
            "district_map": preprocessor.district_median_map,
            "global_median": preprocessor.global_median,
            "features_order": list(X.columns)
        }, f)