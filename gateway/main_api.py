import sys
import os

# --- PATH CONFIGURATION ---
# This ensures Python can find the 'services' directory from the root project folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional

# Import our new Dataset Service Processor
from services.dataset_service.processor import DatasetProcessor
from agents.dataset_agent import DatasetIntelligenceAgent
from services.eda_service.analyzer import EDAService
from services.feature_engineering_service.engineer import FeatureEngineer
from services.training_service.trainer import ModelTrainer

app = FastAPI(
    title="No-Code ML Platform API Gateway",
    description="Central hub for routing frontend requests to ML microservices.",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"status": "online", "message": "API Gateway is running."}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Receives the dataset, reads it into memory, and forwards it to the Dataset Service.
    """
    try:
        # 1. Read the raw file bytes into memory
        file_content = await file.read()
        
        # 2. Forward the bytes and filename to the Dataset Service Processor
        # This will save the file to 'temp_datasets/' and extract the metadata
        processing_result = DatasetProcessor.save_and_validate(file_content, file.filename)
        
        # 3. Check if the processor encountered an error (like a corrupted file)
        if processing_result["status"] == "error":
            raise HTTPException(status_code=400, detail=processing_result["message"])
            
        # 4. If successful, return the rich metadata back to the frontend
        return JSONResponse(
            status_code=200, 
            content={
                "message": "Dataset successfully uploaded and processed!",
                "metadata": processing_result["metadata"]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze-dataset/{dataset_id}")
async def analyze_dataset(dataset_id: str, filename: str):
    """
    Triggers the Dataset Intelligence Agent to profile the uploaded dataset.
    """
    try:
        # Construct the path to where the DatasetProcessor saved the file
        file_ext = filename.split('.')[-1].lower()
        file_path = os.path.join("temp_datasets", f"{dataset_id}.{file_ext}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found on server.")
            
        # Trigger the AI Agent
        profile = DatasetIntelligenceAgent.analyze_dataset(file_path, filename)
        
        return JSONResponse(status_code=200, content={"profile": profile})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/determine-problem/{dataset_id}")
async def determine_problem(dataset_id: str, filename: str, target_column: str):
    """
    Analyzes the selected target column and determines the ML problem type.
    """
    try:
        file_ext = filename.split('.')[-1].lower()
        file_path = os.path.join("temp_datasets", f"{dataset_id}.{file_ext}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found on server.")
            
        result = DatasetIntelligenceAgent.suggest_problem_type(file_path, filename, target_column)
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/eda-report/{dataset_id}")
async def get_eda_report(dataset_id: str, filename: str, target_column: Optional[str] = None):
    """
    Triggers the EDA Service to calculate descriptive statistics, correlations, 
    and target-specific multivariate analysis if a target is provided.
    """
    try:
        file_ext = filename.split('.')[-1].lower()
        file_path = os.path.join("temp_datasets", f"{dataset_id}.{file_ext}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found on server.")
            
        # Pass the target_column directly to the upgraded EDA service
        report = EDAService.generate_statistics(file_path, filename, target_column)
        return JSONResponse(status_code=200, content=report)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/engineer-features/{dataset_id}")
async def engineer_features(dataset_id: str, filename: str, target_column: str):
    """
    Triggers the Feature Engineering Service to clean, encode, and scale the dataset.
    """
    try:
        result = FeatureEngineer.process_dataset(dataset_id, filename, target_column)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/train-models/{dataset_id}")
async def train_models(dataset_id: str, target_column: str, problem_type: str):
    """
    Triggers the AutoML pipeline to train, evaluate, and save models.
    """
    try:
        result = ModelTrainer.train_and_evaluate(dataset_id, target_column, problem_type)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)