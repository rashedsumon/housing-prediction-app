# model.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_loader import load_housing_data

def train_and_save_model():
    print("Starting model training pipeline...")
    
    # 1. Load Data
    df = load_housing_data()
    
    # 2. Preprocess Categorical Variables
    # Convert binary columns (yes/no) to 1/0
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
        
    # Convert nominal categorical column ('furnishingstatus') using One-Hot Encoding
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
    
    # Ensure boolean columns from get_dummies convert to integers (1/0)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # 3. Separate Features (X) and Target (y)
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Save feature names order to ensure the web app structure matches exactly
    model_features = X.columns.tolist()
    
    # 4. Split and Train
    # FIX: Changed test_test_split=0.2 to test_size=0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R^2 Score: {train_score:.4f}")
    print(f"Testing R^2 Score: {test_score:.4f}")
    
    # 5. Export Model Artifacts
    artifacts = {
        'model': model,
        'features': model_features,
        'binary_cols': binary_cols
    }
    
    with open('housing_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("Model and preprocessing metadata successfully saved to 'housing_model.pkl'!")

if __name__ == "__main__":
    train_and_save_model()