import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

class ModelTrainer:
    """
    The AutoML engine. Automatically trains multiple algorithms, evaluates them, 
    and saves the best performing model.
    """
    
    @staticmethod
    def train_and_evaluate(dataset_id: str, target_column: str, problem_type: str) -> dict:
        # 1. Load the CLEANED dataset from the previous phase
        processed_file = os.path.join("temp_datasets", f"{dataset_id}_processed.csv")
        if not os.path.exists(processed_file):
            raise FileNotFoundError("Processed dataset not found. Please run Feature Engineering first.")
            
        df = pd.read_csv(processed_file)
        
        # 2. Separate Features (X) and Target (y)
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # 3. Train-Test Split (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Initialize the algorithms based on the problem type
        models = {}
        is_classification = "Classification" in problem_type
        
        if is_classification:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42)
            }
            
        # 5. Train all models and record their performance
        leaderboard = []
        best_model_name = None
        best_model_obj = None
        best_score = -float('inf') if is_classification else float('inf') # Maximize accuracy, minimize error
        
        for name, model in models.items():
            # The AI is "learning" here
            model.fit(X_train, y_train)
            
            # The AI takes its "final exam"
            predictions = model.predict(X_test)
            
            # Grade the exam
            if is_classification:
                acc = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted')
                leaderboard.append({"Model": name, "Accuracy": round(acc, 4), "F1-Score": round(f1, 4)})
                
                # Check if this is the best model so far
                if acc > best_score:
                    best_score = acc
                    best_model_name = name
                    best_model_obj = model
            else:
                rmse = mean_squared_error(y_test, predictions, squared=False)
                r2 = r2_score(y_test, predictions)
                leaderboard.append({"Model": name, "RMSE": round(rmse, 4), "R2-Score": round(r2, 4)})
                
                # Check if this is the best model (Lowest RMSE is best)
                if rmse < best_score:
                    best_score = rmse
                    best_model_name = name
                    best_model_obj = model
                    
        # 6. Save the absolute best model to the hard drive for future use
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{dataset_id}_best_model.joblib")
        joblib.dump(best_model_obj, model_path)
        
        # Sort the leaderboard (Highest Accuracy first, or Lowest RMSE first)
        leaderboard = sorted(leaderboard, key=lambda x: x.get("Accuracy", x.get("RMSE")), reverse=is_classification)
        
        return {
            "status": "success",
            "problem_type": problem_type,
            "best_model": best_model_name,
            "best_score": round(best_score, 4),
            "leaderboard": leaderboard,
            "model_path": model_path
        }