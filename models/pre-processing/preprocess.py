import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, test_size=0.2, random_state=42):
    X = data.drop("alzheimers_status", axis=1)
    y = data["alzheimers_status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_file = "path/to/your/data.csv"
    data = load_data(data_file)
    X_train, X_test, y_train, y_test = preprocess_data(data)
