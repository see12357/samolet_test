import pandas as pd
import numpy as np
import json
from typing import Any, Dict


class InferencePreprocessor:
    def __init__(self, meta_path: str = "inference_meta.json"):
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.district_map = self.meta["district_map"]
        self.global_median = self.meta["global_median"]
        self.features_order = self.meta["features_order"]

    def _fuzzy_district_search(self, input_name: str) -> float:
        name = input_name.lower()
        for d_name, d_price in self.district_map.items():
            if d_name.lower() in name or name in d_name.lower():
                return d_price
        return self.global_median

    def process(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        data = raw_data.copy()
        features = {col: np.nan for col in self.features_order}

        area = float(data.get('TotalArea', 30.0))
        rooms = int(data.get('rooms_count', 1))

        features.update({
            'TotalArea': area,
            'Floor': int(data.get('Floor', 1)),
            'FloorsTotal': int(data.get('FloorsTotal', 1)),
            'CeilingHeight': float(data.get('CeilingHeight', 2.7)),
            'rooms_count': rooms,
            'is_euro': 1 if 'евро' in str(data.get('PropertyType', '')).lower() else 0,
            'h_year': 2025,
            'district_level': self._fuzzy_district_search(data.get('District', '')),
            'area_per_room': area / rooms,
            'floor_rel': int(data.get('Floor', 1)) / int(data.get('FloorsTotal', 1)),
            'is_high_ceiling': 1 if float(data.get('CeilingHeight', 2.7)) > 3.0 else 0,
            'premium_score': float(data.get('CeilingHeight', 2.7)) * np.log1p(area)
        })

        for cat in self.meta["cat_features"]:
            features[cat] = str(data.get(cat, "Unknown"))

        df = pd.DataFrame([features])
        return df[self.features_order]