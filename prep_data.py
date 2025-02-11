import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def get_prepared_data(data_path="data"):

    
    df = get_raw_data(data_path)

    #drop excessive/non-helpful features
    df.drop('Poster_Link', axis=1, inplace=True)
    df.drop('Overview', axis=1, inplace=True)

    #editing data:
    df.rename(columns={'Series_Title': 'Movie'}, inplace=True) #changing name to movie is easier
    labels=df['Movie'].copy()
    df.loc[df['Movie'] == 'Apollo 13', 'Released_Year'] = '1995' #filling in missing data here.

    #make sure numeric columns are converted from strings
    df['Released_Year']= df['Released_Year'].astype(int)

    #dropping rows where gross or certificate is missing
    df.dropna(subset=['Gross'], inplace=True)
    df.dropna(subset=['Certificate'], inplace=True)

    #converting strings to numbers where applicable
    df['Gross'] = df['Gross'].replace('[,]', '', regex=True).astype(int)
    df['Runtime'] = df['Runtime'].str.replace('min', '').str.strip().astype(int)
    
    #drop labels so they are not one hot encoded
    labels=df['Movie'].copy()
    df.drop('Movie', axis=1, inplace=True)

    #creating the pipeline
    num_pipeline= Pipeline([
        ('scaler',StandardScaler()),
        ('imputer', SimpleImputer(strategy='median'))
    ])


    #now selecting only the features I want (make sure to update pipeline)
    df.drop('Genre', axis=1, inplace=True)
    df.drop('Star3',axis=1,inplace= True)
    df.drop('Star4',axis=1,inplace= True)
    df.drop('Director',axis=1,inplace= True)
    df.drop('Meta_score',axis=1,inplace= True)
    df.drop('IMDB_Rating',axis=1,inplace= True)


    pipeline= ColumnTransformer([
        ("cat",OneHotEncoder(),['Certificate','Star1','Star2']),
        ("num",num_pipeline,['Gross','No_of_Votes','Released_Year'])
    ])

    df_transformed= pipeline.fit_transform(df).toarray()
    feature_names= pipeline.get_feature_names_out()
    df_transformed= pd.DataFrame(df_transformed, columns=feature_names)

    # allowed_actors = ["Tom Hanks", "Keanu Reeves", "Tom Cruise"]
    # df_transformed = filter_actors(df_transformed, allowed_actors)

    gross=df_transformed['num__Gross'].copy()
    df_transformed.drop('num__Gross', axis=1, inplace=True)

    # Define features and target
    features = df_transformed
    target = gross

    # Convert to numpy arrays
    features = np.array(features)
    target = np.array(target).reshape(-1, 1)

    # Convert to torch tensors
    features = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    return features, target

def filter_actors(df, allowed_actors):
    allowed_columns = [f'cat__{actor.replace(" ", "_")}' for actor in allowed_actors]
    
    actor_columns = [col for col in df.columns if col.startswith("cat__")]
  
    columns_to_keep = [col for col in df.columns if col in allowed_columns or col not in actor_columns]
    
    return df[columns_to_keep]

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

def get_real_numbers(data_path="data"):
    df = get_raw_data(data_path)

    #editing data:
    df.rename(columns={'Series_Title': 'Movie'}, inplace=True) #changing name to movie is easier
    labels=df['Movie'].copy()
    df.loc[df['Movie'] == 'Apollo 13', 'Released_Year'] = '1995' #filling in missing data here.

    #make sure numeric columns are converted from strings
    df['Released_Year']= df['Released_Year'].astype(int)

    #dropping rows where gross or certificate is missing
    df.dropna(subset=['Gross'], inplace=True)
    df.dropna(subset=['Certificate'], inplace=True)

    #converting strings to numbers where applicable
    df['Gross'] = df['Gross'].replace('[,]', '', regex=True).astype(int)
    return df['Gross']

def unscale_predicted_vals(output):
    df = get_raw_data("data")
    df['Gross'] = df['Gross'].replace('[,]', '', regex=True).astype(int)

    scaler_target = StandardScaler()
    target_scaled = scaler_target.fit_transform(df['Gross'].to_numpy().reshape(-1, 1))  # Scaling target
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    return scaler_target.inverse_transform(output)
