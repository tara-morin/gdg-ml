import pandas as pd
import numpy as np
import torch

# Load dataset and return cleaned and preprocessed features and target
# TODO: use more of the columns, preprocess them in a different way,
#       or even include new features (e.g. from other datasets)

# columns: 'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate',
#          'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
#          'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes'

# columns used in template: 'Released_Year', 'Certificate', 'Runtime', 'Genre' (kind of',
# 'IMDB_Rating', 'Meta_score', 'No_of_Votes'
def get_prepared_data(data_path="data/IMBD top 1000.csv"):

    # Load raw data
    # this function tries to combine all .csv files in the data folder
    # it matches them up using the "Series_Title" column
    # if you want to use additional datasets, make sure they have a "Series_Title" column
    # if not, you will need additional logic to join the datasets
    # do not rename the column by hand, add code before this point to rename it
    # remember: we will not manually modify your datasets, so your code must do any formatting automatically
    data = get_raw_data(data_path)

    # Drop columns in text format (not used in the demo, may be useful to you)
    data = data.drop(columns=["Poster_Link", "Series_Title", "Overview", "Director", "Star1", "Star2", "Star3", "Star4"])

    # Create separate columns for all genres
    def process_genres(data):
        # Split genres into list
        genres = data["Genre"].str.split(",").explode().str.strip()
        
        # Create binary columns
        genre_dummies = pd.get_dummies(genres, prefix="Genre").groupby(level=0).max()
        
        # Merge with original data
        return pd.concat([data.drop("Genre", axis=1), genre_dummies], axis=1)

    data = process_genres(data)

    def process_gross(data):
        # Clean numeric format
        data["Gross"] = data["Gross"].replace('[$,]', '', regex=True).astype(float)
        
        # Log-transform to handle skew
        data["Gross"] = np.log1p(data["Gross"])  # Better for regression
        return data

    data = process_gross(data)
    
    # Convert categorical columns to one-hot encoding
    data = pd.get_dummies(data)

    # Define features and target
    features = data.drop(columns=["Gross"])
    target = data["Gross"]

    # Convert to numpy arrays
    features = np.array(features)
    target = np.array(target).reshape(-1, 1)

    # Convert to torch tensors
    features = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    return features, target

def get_all_titles(data_path="data"):
    data = get_raw_data(data_path)
    return data["Series_Title"]

def get_raw_data(path="data"):
    # read in every csv file, join on "Series_Title"
    # return the raw data
    import os
    files = os.listdir(path)
    data = pd.DataFrame()
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            df = df.dropna()
            # join on "Series_Title"
            if data.empty:
                data = df
            else:
                data = data.merge(df, on="Series_Title")
    return data