import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    """
    Preprocess the dataset.

    Args:
    - file_path (str): Path to the dataset file.

    Returns:
    - pd.DataFrame: Preprocessed dataset.
    """
    data = pd.read_csv(file_path, sep=';')
    
    # Encode categorical variables
    categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                           'Mjob', 'Fjob', 'reason', 'guardian', 
                           'schoolsup', 'famsup', 'paid', 'activities', 
                           'nursery', 'higher', 'internet', 'romantic']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Scale numerical features
    numeric_columns = ['age', 'absences']
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data
