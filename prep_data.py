import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def get_prepared_data(data_path="data/IMDB top 1000.csv"):
    """Load and preprocess movie data for revenue prediction."""
    
    # Load and clean raw data
    data = get_raw_data(data_path)
    
    # Use only relevant columns
    relevant_cols = ['Duration', 'Rate', 'Metascore', 'Info']
    data = data[relevant_cols].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # --- Data Cleaning Pipeline ---
    def process_gross(df):
        """Extract and clean gross revenue values from Info column"""
        # Extract gross from Info column
        df["Gross"] = df["Info"].str.extract(r'Gross: \$?([\d,]+)M?')
        # Remove commas and convert to float
        df["Gross"] = df["Gross"].replace(',', '', regex=True).astype(float)
        # If values are in millions (M), multiply by 1,000,000
        df.loc[df["Info"].str.contains('M'), "Gross"] *= 1e6
        # Remove rows with missing gross values
        df = df.dropna(subset=["Gross"])
        # Log-transform to handle skewed distribution
        df["Gross"] = np.log1p(df["Gross"])
        return df.drop(columns=["Info"])

    def add_features(df):
        """Create engineered features"""
        # Convert Duration to numeric (remove ' min' and convert to int)
        df["Duration"] = df["Duration"].str.replace(' min', '').astype(int)
        # Convert Rate to float
        df["Rate"] = df["Rate"].astype(float)
        # Convert Metascore to float, replacing '-' with NaN
        df["Metascore"] = pd.to_numeric(df["Metascore"].replace('-', np.nan), errors='coerce')
        # Interaction features
        df["Rate_to_Duration"] = df["Rate"] / df["Duration"]
        return df.fillna(df.mean(numeric_only=True))

    # --- Execute Pipeline ---
    data = process_gross(data)
    data = add_features(data)
    
    # --- Prepare Final Data ---
    # Separate features and target
    features = data.drop(columns=["Gross"])
    target = data["Gross"]

    # Normalize numerical features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Convert to tensors
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    target_tensor = torch.tensor(target.values, dtype=torch.float32).unsqueeze(1)

    print(f"Number of features: {features.shape[1]}")

    return features_tensor, target_tensor

def get_all_titles(data_path="data/IMDB top 1000.csv"):
    """Get list of movie titles without preprocessing"""
    data = get_raw_data(data_path)
    return data["Title"]

def get_raw_data(path="data/IMDB top 1000.csv"):
    """Load raw data from CSV file(s)"""
    if path.endswith(".csv"):
        # Single file load
        data = pd.read_csv(path)
    else:
        # Original multi-file merge logic
        import os
        data = pd.DataFrame()
        for file in os.listdir(path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, file))
                if data.empty:
                    data = df
                else:
                    # Merge on 'Title' instead of 'Series_Title'
                    data = data.merge(df, on="Title", how="outer")
    
    print("Columns in the DataFrame:", data.columns.tolist())
    
    # Check if 'Title' exists, if not, use the first column as the title
    if 'Title' not in data.columns:
        print("'Title' not found. Using first column as title.")
        data = data.rename(columns={data.columns[0]: 'Title'})
    
    # Basic cleaning for merge safety
    return data.dropna(subset=["Title"]).reset_index(drop=True)


