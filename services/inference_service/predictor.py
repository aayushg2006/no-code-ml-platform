import pandas as pd
import joblib
import os

class InferenceService:
    """
    Handles predicting new, unseen data by loading the saved model and 
    applying the exact same mathematical transformations used during training.
    """
    
    @staticmethod
    def predict_single_record(dataset_id: str, input_data: dict) -> dict:
        # 1. Load the frozen model and pipeline state
        model_path = os.path.join("models", f"{dataset_id}_best_model.joblib")
        pipeline_path = os.path.join("models", f"{dataset_id}_pipeline.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
            raise FileNotFoundError("Model or Pipeline state not found. Please train the model first.")
            
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        
        # 2. Convert the user's single row of JSON data into a Pandas DataFrame
        df = pd.DataFrame([input_data])
        
        # 3. Apply the exact same imputers to fill missing values
        if pipeline["num_imputer"] and len(pipeline["numeric_cols"]) > 0:
            # Only impute columns that were actually provided in the input
            cols_to_impute = [c for c in pipeline["numeric_cols"] if c in df.columns]
            if cols_to_impute:
                df[cols_to_impute] = pipeline["num_imputer"].transform(df[cols_to_impute])
                
        if pipeline["cat_imputer"] and len(pipeline["categorical_cols"]) > 0:
            cols_to_impute = [c for c in pipeline["categorical_cols"] if c in df.columns]
            if cols_to_impute:
                df[cols_to_impute] = pipeline["cat_imputer"].transform(df[cols_to_impute])

        # 4. Apply One-Hot Encoding
        if len(pipeline["categorical_cols"]) > 0:
            df = pd.get_dummies(df, columns=[c for c in pipeline["categorical_cols"] if c in df.columns])
            
        # 5. ALIGNMENT FIX: The Dummy Variable Trap
        # Ensure the new dataframe has the exact same columns as the training data.
        # Fill any missing columns (like 'Sex_female' if only 'Sex_male' was triggered) with 0.
        df = df.reindex(columns=pipeline["final_columns"], fill_value=0)
        
        # 6. Apply the exact same Scaler
        if pipeline["scaler"] and len(pipeline["numeric_cols"]) > 0:
            df[pipeline["numeric_cols"]] = pipeline["scaler"].transform(df[pipeline["numeric_cols"]])
            
        # 7. Generate Prediction!
        prediction = model.predict(df)[0] # Extract the single value from the array
        
        # 8. Translate the answer back to text if it was a classification task
        prediction_label = str(prediction)
        if pipeline["target_mapping"]:
            # joblib saves JSON keys as strings, so we ensure the lookup matches
            mapping = {str(k): v for k, v in pipeline["target_mapping"].items()}
            prediction_label = mapping.get(str(prediction), str(prediction))
            
        return {
            "status": "success",
            "raw_prediction": float(prediction),
            "prediction_label": prediction_label
        }