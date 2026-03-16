import pandas as pd
import os

class DatasetIntelligenceAgent:
    """
    Agent responsible for automatically profiling datasets to determine
    their structure, missing values, and data types.
    """
    
    @staticmethod
    def analyze_dataset(file_path: str, filename: str) -> dict:
        """
        Reads the saved file and performs deep statistical profiling.
        """
        file_ext = filename.split('.')[-1].lower()
        
        # 1. Safely load the dataset from the local storage
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext == 'xlsx':
            df = pd.read_excel(file_path)
        elif file_ext == 'json':
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported format for analysis")

        # 2. Automated Column Type Detection
        # We classify every column so the ML algorithms know how to process them later
        column_types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if 'int' in dtype or 'float' in dtype:
                column_types[col] = 'Numeric'
            elif 'datetime' in dtype:
                column_types[col] = 'Datetime'
            else:
                column_types[col] = 'Categorical/Text'

        # 3. Missing Value Analysis
        # Calculates the exact percentage of missing data for each column
        missing_values = df.isnull().sum()
        missing_percentages = (missing_values / len(df)) * 100
        # Only keep columns that actually have missing values to keep the report clean
        missing_stats = missing_percentages[missing_percentages > 0].to_dict()

        # 4. Duplicate Row Detection
        duplicate_count = int(df.duplicated().sum())

        # 5. Compile the Intelligence Profile
        profile = {
            "column_types": column_types,
            "missing_values": {col: round(val, 2) for col, val in missing_stats.items()},
            "duplicate_rows": duplicate_count,
            "total_rows": len(df)
        }
        
        return profile
        
    @staticmethod
    def suggest_problem_type(file_path: str, filename: str, target_column: str) -> dict:
        """
        Determines if the user wants to do Classification or Regression
        based on the target variable they want to predict.
        """
        file_ext = filename.split('.')[-1].lower()
        
        # 1. Load the dataset
        if file_ext == 'csv': df = pd.read_csv(file_path)
        elif file_ext == 'xlsx': df = pd.read_excel(file_path)
        elif file_ext == 'json': df = pd.read_json(file_path)
        else: raise ValueError("Unsupported format")

        # 2. Extract the specific column the user wants to predict
        target_data = df[target_column].dropna()
        unique_count = target_data.nunique()
        dtype = str(target_data.dtype)

        # 3. Agentic Logic Heuristics
        if 'float' in dtype:
            # Floats (decimals) are almost always continuous regression targets
            problem_type = "Regression"
        elif 'int' in dtype:
            # Integers are tricky. If there are only 2 unique numbers (0 and 1), it's binary classification.
            # If there are many unique numbers, it's likely a regression (like counting days or age).
            if unique_count <= 15:
                problem_type = "Classification (Binary)" if unique_count == 2 else "Classification (Multiclass)"
            else:
                problem_type = "Regression"
        else:
            # Text/Strings are always categorical classification
            problem_type = "Classification (Binary)" if unique_count == 2 else "Classification (Multiclass)"

        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "unique_values": int(unique_count)
        }