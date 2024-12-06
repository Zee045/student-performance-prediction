import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(file_path):
    """
    Load and preprocess the dataset.

    Args:
    - file_path (str): Path to the dataset file.

    Returns:
    - pd.DataFrame: Preprocessed dataset.
    """
    data = pd.read_csv(file_path, sep=';')
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['sex', 'Mjob', 'Fjob'], drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    numeric_features = ['age', 'absences']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    return data
