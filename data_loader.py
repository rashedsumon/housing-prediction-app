import os
import shutil
import pandas as pd
import kagglehub

def load_housing_data():
    """
    Downloads the housing dataset via kagglehub if it doesn't exist locally,
    and returns it as a pandas DataFrame.
    """
    local_csv_path = "Housing.csv"
    
    # 1. Check if the file is already available locally
    if not os.path.exists(local_csv_path):
        print("Dataset not found locally. Downloading from Kaggle...")
        try:
            # Download latest version
            path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")
            downloaded_csv = os.path.join(path, "Housing.csv")
            
            # Copy file to the project root directory for easy access
            shutil.copy(downloaded_csv, local_csv_path)
            print(f"Dataset successfully downloaded and saved to: {local_csv_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from Kaggle: {e}")
            
    # 2. Read and return the dataset
    df = pd.read_csv(local_csv_path)
    return df

if __name__ == "__main__":
    # Test script execution
    data = load_housing_data()
    print("Data loaded successfully! Shape:", data.shape)
    print("Columns available:", data.columns.tolist())