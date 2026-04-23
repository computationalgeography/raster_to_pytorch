import numpy as np
import pandas as pd
import pickle
import torch

class DataScaler:
    def __init__(self):
        self.transformers = {}
        self.var_groups = ['dynamic', 'static', 'target']

    def load_transformers(self, scalers_path):
        for var_group in self.var_groups:
            file_path = scalers_path / f'{var_group}_transformer.pkl'
            with open(file_path, 'rb') as f:
                self.transformers[var_group] = pickle.load(f)

    def get_transformers(self):
        return self.transformers

    def get_transformed_zero(self):
        target_transformer = self.transformers.get('target')
        if target_transformer is None:
            raise ValueError("Target transformer has not been fitted or loaded.")
        zero_df = pd.DataFrame({'swe': [0.0]})
        transformed_zero = target_transformer.transform(zero_df)
        return float(transformed_zero[0, 0])

    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    def transform_dataframe(self, df, var_group):
        if not self.transformers:
            raise ValueError("Transformers have not been fitted or loaded.")
        if var_group not in self.transformers:
            raise ValueError(
                "Variable group %r does not have a corresponding transformer."
                " Available groups: %s"
                % (var_group, list(self.transformers.keys()))
            )
        return self.to_tensor(self.transformers[var_group].transform(df))
        
    def log1p_clip(self, x):
        return np.log1p(np.clip(x, 0, None))