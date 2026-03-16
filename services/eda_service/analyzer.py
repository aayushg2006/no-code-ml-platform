import pandas as pd
import numpy as np

class EDAService:
    """
    Handles the heavy mathematical lifting for Exploratory Data Analysis.
    Calculates statistical summaries, correlation matrices, and target-specific insights.
    """
    
    @staticmethod
    def generate_statistics(file_path: str, filename: str, target_column: str = None) -> dict:
        # 1. Safely load the dataset
        file_ext = filename.split('.')[-1].lower()
        if file_ext == 'csv': df = pd.read_csv(file_path)
        elif file_ext == 'xlsx': df = pd.read_excel(file_path)
        elif file_ext == 'json': df = pd.read_json(file_path)
        else: raise ValueError("Unsupported format")

        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        categorical_df = df.select_dtypes(include=['object', 'category'])

        # 2. Descriptive Statistics
        stats_summary = {}
        if not numeric_df.empty:
            desc = numeric_df.describe().replace([np.inf, -np.inf, np.nan], None)
            stats_summary = desc.to_dict()

        # 3. Correlation Matrix
        correlation_matrix = {}
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr = numeric_df.corr().replace([np.inf, -np.inf, np.nan], None)
            correlation_matrix = corr.to_dict()

        # 4. Categorical Summaries
        categorical_summary = {}
        for col in categorical_df.columns:
            counts = categorical_df[col].value_counts().head(10).to_dict()
            categorical_summary[col] = counts

        # --- NEW: 5. Target-Specific Multivariate Analysis ---
        target_analysis = {}
        if target_column and target_column in df.columns:
            target_data = df[target_column]
            
            # Determine if target is numeric or categorical
            if pd.api.types.is_numeric_dtype(target_data):
                target_analysis["type"] = "numeric"
                # For numeric targets, find the top features most correlated with it
                if not numeric_df.empty and target_column in numeric_df.columns:
                    target_corr = numeric_df.corr()[target_column].drop(target_column)
                    target_corr = target_corr.replace([np.inf, -np.inf, np.nan], None)
                    target_analysis["feature_correlations"] = target_corr.sort_values(ascending=False).to_dict()
            else:
                target_analysis["type"] = "categorical"
                # For categorical targets, calculate the mean of numeric features grouped by the target classes
                target_analysis["class_distribution"] = target_data.value_counts().head(10).to_dict()
                if not numeric_df.empty:
                    # Group by target and calculate mean for numeric columns
                    grouped_means = df.groupby(target_column)[numeric_df.columns].mean()
                    grouped_means = grouped_means.replace([np.inf, -np.inf, np.nan], None)
                    target_analysis["grouped_means"] = grouped_means.to_dict(orient="index")

        return {
            "descriptive_statistics": stats_summary,
            "correlation_matrix": correlation_matrix,
            "categorical_summary": categorical_summary,
            "target_analysis": target_analysis
        }