import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from sklearn.compose import ColumnTransformer


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect imports to the correct module
        if module == "data_scaler":
            module = "lib.data_scaler"
        # Handle pathlib.PosixPath for cross-platform compatibility
        if module == "pathlib" and name == "PosixPath":
            return Path
        return super().find_class(module, name)

class DataScaler:
    def __init__(self):
        self.transformers = {}
        self.var_groups = ['dynamic', 'static', 'target']

    def log1p_clip(self, X):
        return np.log1p(np.clip(X, 0, None))

    def load_transformers(self, scalers_path):
        for var_group in self.var_groups:
            file_path = scalers_path / f'{var_group}_transformer.pkl'
            with open(file_path, 'rb') as f:
                self.transformers[var_group] = CustomUnpickler(f).load()

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
    
    def get_target_estimator(self):
        target = self.transformers.get('target')
        if target is None:
            raise ValueError("Target transformer has not been fitted or loaded.")
        if isinstance(target, ColumnTransformer):
            for _, est, cols in target.transformers_:
                if len(cols) == 0:
                    continue
                return est
            raise ValueError("No inner estimator found in target ColumnTransformer.")
        return target

    def inverse_transform_target(self, arr):
        est = self.get_target_estimator()
        og_shape = arr.shape
        inv = est.inverse_transform(pd.DataFrame(arr.flatten(), columns=['swe']))
        return inv.reshape(og_shape)