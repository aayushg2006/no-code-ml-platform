import pandas as pd
import os
import uuid

# Define our local storage path for the raw data
STORAGE_DIR = "temp_datasets"
os.makedirs(STORAGE_DIR, exist_ok=True)

class DatasetProcessor:
    """
    Handles dataset ingestion, storage, and validation using pandas.
    """
    @staticmethod
    def save_and_validate(file_content: bytes, filename: str) -> dict:
        # 1. Generate a unique ID so multiple users don't overwrite each other's files
        dataset_id = str(uuid.uuid4())
        file_extension = filename.split('.')[-1].lower()
        
        # Create a unique file path safely stored in our local directory
        save_path = os.path.join(STORAGE_DIR, f"{dataset_id}.{file_extension}")
        
        # 2. Save the raw bytes to the hard drive
        with open(save_path, "wb") as f:
            f.write(file_content)
            
        # 3. Read the dataset using pandas for integrity checks
        try:
            if file_extension == 'csv':
                df = pd.read_csv(save_path)
            elif file_extension == 'xlsx':
                df = pd.read_excel(save_path)
            elif file_extension == 'json':
                df = pd.read_json(save_path)
            else:
                raise ValueError("Unsupported file format")
                
            # 4. Integrity Check: Ensure the dataset actually contains data
            if df.empty:
                raise ValueError("The uploaded dataset is completely empty.")
                
            # 5. Generate the Dataset Metadata
            metadata = {
                "dataset_id": dataset_id,
                "filename": filename,
                "num_rows": int(df.shape[0]),
                "num_columns": int(df.shape[1]),
                "columns": df.columns.tolist(),
                "file_size_bytes": os.path.getsize(save_path)
            }
            
            return {"status": "success", "metadata": metadata}
            
        except Exception as e:
            # If pandas fails to read the file, it is corrupted. 
            # We delete the bad file to save storage space.
            if os.path.exists(save_path):
                os.remove(save_path)
            return {"status": "error", "message": str(e)}