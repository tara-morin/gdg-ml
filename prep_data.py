import pandas as pd
import numpy as np
import torch
import re
from sklearn.preprocessing import StandardScaler
import os

def get_raw_data(path="data"):
    """Robust CSV merger with enhanced title handling"""
    def standardize_title(df):
        """Find and rename title column across variations"""
        # Convert all columns to lowercase first
        df.columns = df.columns.str.lower()
        
        # List of possible title column names
        title_patterns = ['title', 'name', 'movie', 'film']
        
        # Check for existing title column
        for pattern in title_patterns:
            if pattern in df.columns:
                df = df.rename(columns={pattern: 'title'})
                return df
        
        # Fallback: Use first text column
        str_cols = df.select_dtypes(include='object').columns
        if len(str_cols) > 0:
            print(f"⚠️ Using '{str_cols[0]}' as title column")
            return df.rename(columns={str_cols[0]: 'title'})
        
        raise ValueError("No viable title column found")

    merged = pd.DataFrame()
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            # Read and standardize columns
            df = pd.read_csv(os.path.join(path, file))
            df = standardize_title(df)
            
            # Clean title values
            df['title'] = (
                df['title']
                .astype(str)
                .str.strip()
                .str.title()
                .replace(['Nan', 'N/A', 'None', ''], pd.NA)
            )
            df = df.dropna(subset=['title'])
            
            # Merge data
            if merged.empty:
                merged = df
            else:
                merged = pd.merge(
                    merged,
                    df,
                    on='title',
                    how='outer',
                    suffixes=('', f'_DROP_{file}')
                ).filter(regex='^(?!.*_DROP)')

    # Final cleaning
    merged = merged.loc[:, ~merged.columns.duplicated()]
    print("✅ Merged columns:", merged.columns.tolist())
    return merged

def process_gross(df):
    """Handle gross revenue from multiple potential sources"""
    try:
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        
        if 'gross' in df.columns:
            df.loc[:, "gross"] = pd.to_numeric(df["gross"].replace('[$,]', '', regex=True), errors='coerce')
        elif 'Gross' in df.columns:
            df.loc[:, "gross"] = pd.to_numeric(df["Gross"].replace('[$,]', '', regex=True), errors='coerce')
            df = df.drop('Gross', axis=1)
        
        # Remove rows where gross is NaN or 0
        df = df[df["gross"].notna() & (df["gross"] > 0)]
        df.loc[:, "gross"] = np.log1p(df["gross"])
        return df
    except Exception as e:
        print(f"Error in process_gross: {str(e)}")
        raise

def add_features(df):
    """Create powerful features from raw columns"""
    try:
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Convert duration to numeric, handling NaN values
        if 'duration' in df.columns:
            df['duration'] = pd.to_numeric(
                df['duration'].astype(str).str.extract(r'(\d+)')[0], 
                errors='coerce'
            )

        # Handle certificate column
        if 'certificate' in df.columns:
            df['certificate'] = df['certificate'].astype(str)
            df['certificate'] = df['certificate'].apply(
                lambda x: re.search(r'([A-Za-z-]+)', x).group(1) 
                if pd.notnull(x) and re.search(r'([A-Za-z-]+)', x) 
                else None
            )

        # Director impact score
        if 'director' in df.columns:
            director_counts = df['director'].value_counts(normalize=True)
            df['director_impact'] = df['director'].map(director_counts)

        # Genre expansion - Fixed version
        if 'genre' in df.columns:
            df['genre'] = df['genre'].fillna('')
            # Split genres and create a list of all unique genres
            all_genres = set()
            for genres in df['genre'].str.split(', '):
                if isinstance(genres, list):
                    all_genres.update(genres)
            
            # Create dummy variables for each genre
            for genre in all_genres:
                if genre:  # Skip empty strings
                    df[f'genre_{genre}'] = df['genre'].str.contains(genre, case=False, na=False).astype(int)

        # Handle numeric columns
        numeric_cols = ['rate', 'metascore', 'score', 'votes', 'budget']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill missing values with median
                df[col] = df[col].fillna(df[col].median())

        return df

    except Exception as e:
        print(f"Error in add_features: {str(e)}")
        raise

def process_gross(df):
    """Handle gross revenue from multiple potential sources"""
    try:
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        
        if 'gross' in df.columns:  # Check for lowercase column name
            df["gross"] = pd.to_numeric(df["gross"].replace('[$,]', '', regex=True), errors='coerce')
        elif 'Gross' in df.columns:  # Check for uppercase column name
            df["gross"] = pd.to_numeric(df["Gross"].replace('[$,]', '', regex=True), errors='coerce')
            df = df.drop('Gross', axis=1)
        else:
            # Extract from Info column
            df["gross"] = df["info"].str.extract(r'Gross:.*?\$?([\d,.]+)\s?(?:M|million)', flags=re.IGNORECASE)
            df["gross"] = pd.to_numeric(df["gross"].str.replace(',', ''), errors='coerce')
            million_mask = df["info"].str.contains('million', case=False, na=False)
            df.loc[million_mask, "gross"] *= 1e6

        # Fill missing values with median
        df["gross"] = df["gross"].fillna(df["gross"].median())
        df["gross"] = np.log1p(df["gross"])
        return df
    except Exception as e:
        print(f"Error in process_gross: {str(e)}")
        raise

def get_prepared_data(data_path="data"):
    """Load and prepare data for model training"""
    try:
        # Load raw data
        data = get_raw_data(data_path)
        
        # Process gross revenue
        data = process_gross(data)
        
        # Select relevant columns
        keep_columns = [
            'title', 'certificate', 'duration', 'genre', 
            'rate', 'metascore', 'gross', 'budget',
            'director', 'votes', 'year', 'score'
        ]
        
        # Keep only columns that exist in the dataset
        existing_columns = [col for col in keep_columns if col in data.columns]
        data = data[existing_columns].copy()
        
        # Add features
        data = add_features(data)
        
        # Drop non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if 'gross' not in numeric_data.columns:
            raise ValueError("Target column 'gross' not found in dataset")
            
        # Prepare features and target
        features = numeric_data.drop(columns=['gross'])
        target = numeric_data['gross']
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        print(f"Final feature columns: {features.columns.tolist()}")
        
        return torch.tensor(features_scaled, dtype=torch.float32), \
               torch.tensor(target.values, dtype=torch.float32).unsqueeze(1)

    except Exception as e:
        print(f"Error in get_prepared_data: {str(e)}")
        raise

def test_data_preparation():
    """Test function to verify data preparation pipeline"""
    try:
        # Test raw data loading
        raw_data = get_raw_data()
        print(f"✅ Raw data loaded successfully with {len(raw_data)} rows")
        print(f"✅ Columns present: {raw_data.columns.tolist()}")
        
        # Test for missing values
        missing_values = raw_data.isnull().sum()
        print("\nMissing values per column:")
        print(missing_values[missing_values > 0])
        
        # Print data types
        print("\nData types:")
        print(raw_data.dtypes)
        
        # Test feature preparation
        features, target = get_prepared_data()
        print(f"\n✅ Features shape: {features.shape}")
        print(f"✅ Target shape: {target.shape}")
        
        # Additional verification
        print("\nVerification:")
        print(f"Features NaN count: {torch.isnan(features).sum().item()}")
        print(f"Target NaN count: {torch.isnan(target).sum().item()}")
        print(f"Features min value: {features.min().item():.2f}")
        print(f"Features max value: {features.max().item():.2f}")
        print(f"Target min value: {target.min().item():.2f}")
        print(f"Target max value: {target.max().item():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_data_preparation()