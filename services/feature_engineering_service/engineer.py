import pandas as pd
import numpy as np
import os
import joblib # NEW: Imported to save our pipeline state
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
        
        if file_ext == 'csv': df = pd.read_csv(file_path)
        elif file_ext == 'xlsx': df = pd.read_excel(file_path)
        elif file_ext == 'json': df = pd.read_json(file_path)
        else: raise ValueError("Unsupported format")

        original_shape = df.shape

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        df = df.dropna(subset=[target_column])
        dropped_target_rows = original_shape[0] - df.shape[0]
        missing_filled = int(df.drop(columns=[target_column]).isnull().sum().sum())
            
        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()

        target_mapping = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target_column)
            target_mapping = {int(index): label for index, label in enumerate(le.classes_)}

        numeric_cols = list(X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns)
        categorical_cols = list(X.select_dtypes(include=['object', 'category']).columns)

        high_cardinality_cols = [col for col in categorical_cols if X[col].nunique() > 100]
        if high_cardinality_cols:
            X = X.drop(columns=high_cardinality_cols)
            categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]

        # Initialize tools
        num_imputer = None
        cat_imputer = None
        scaler = None

        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            X = X.astype(int)

        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Save the processed dataset
        processed_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        processed_file_path = os.path.join("temp_datasets", f"{dataset_id}_processed.csv")
        processed_df.to_csv(processed_file_path, index=False)

        # --- NEW: Save the Pipeline State ---
        # We must save exactly how the data was transformed so we can repeat it on new input!
        pipeline_state = {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "num_imputer": num_imputer,
            "cat_imputer": cat_imputer,
            "scaler": scaler,
            "final_columns": list(X.columns), # Crucial for aligning One-Hot Encoded columns!
            "target_mapping": target_mapping
        }
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline_state, os.path.join("models", f"{dataset_id}_pipeline.joblib"))

        msg = "Feature engineering completed successfully."
        if dropped_target_rows > 0: msg += f" Dropped {dropped_target_rows} rows because the target '{target_column}' was missing."
        if high_cardinality_cols: msg += f" Dropped {len(high_cardinality_cols)} high-cardinality ID/Text columns."

        return {
            "status": "success",
            "message": msg,
            "original_shape": original_shape,
            "new_shape": processed_df.shape,
            "missing_values_handled": missing_filled,
            "target_mapping": target_mapping,
            "processed_file_path": processed_file_path
        }