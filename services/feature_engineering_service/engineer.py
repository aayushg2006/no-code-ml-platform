import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    """
    Automates data cleaning, imputation, encoding, and scaling to 
    prepare a dataset for machine learning algorithms.
    """
    
    @staticmethod
    def process_dataset(dataset_id: str, filename: str, target_column: str) -> dict:
        file_ext = filename.split('.')[-1].lower()
        file_path = os.path.join("temp_datasets", f"{dataset_id}.{file_ext}")
        
        # 1. Load the raw data
        if file_ext == 'csv': df = pd.read_csv(file_path)
        elif file_ext == 'xlsx': df = pd.read_excel(file_path)
        elif file_ext == 'json': df = pd.read_json(file_path)
        else: raise ValueError("Unsupported format")

        original_shape = df.shape
        missing_filled = int(df.isnull().sum().sum())

        # 2. Separate Target from Features
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
            
        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()

        # 3. Handle the Target Column (Encode if text)
        target_mapping = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target_column)
            target_mapping = {int(index): label for index, label in enumerate(le.classes_)}

        # 4. Identify Column Types
        numeric_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)
        categorical_cols = list(X.select_dtypes(include=['object', 'category']).columns)

        # --- NEW: High Cardinality Filter ---
        # Automatically drop text columns with more than 100 unique values
        high_cardinality_cols = [col for col in categorical_cols if X[col].nunique() > 100]
        if high_cardinality_cols:
            X = X.drop(columns=high_cardinality_cols)
            # Remove them from our categorical list so we don't try to encode them
            categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]

        # 5. Handle Missing Values in Features
        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

            # 6. Encode Categorical Features safely
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            X = X.astype(int)

        # 7. Scale Numeric Features
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # 8. Recombine and Save the Dataset
        processed_df = pd.concat([X, y], axis=1)
        processed_file_path = os.path.join("temp_datasets", f"{dataset_id}_processed.csv")
        processed_df.to_csv(processed_file_path, index=False)

        # Let the user know if columns were dropped
        drop_msg = f" Dropped {len(high_cardinality_cols)} high-cardinality text columns to prevent memory overflow." if high_cardinality_cols else ""

        return {
            "status": "success",
            "message": "Feature engineering completed successfully." + drop_msg,
            "original_shape": original_shape,
            "new_shape": processed_df.shape,
            "missing_values_handled": missing_filled,
            "target_mapping": target_mapping,
            "processed_file_path": processed_file_path
        }