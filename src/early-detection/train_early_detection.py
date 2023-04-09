import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=["label"])
    y = data["label"]
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load and preprocess data
    X, y = load_data("data/biomarker_data/processed_data/training_data.csv")
    X_scaled = preprocess_data(X)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(input_shape=(X_train.shape[1],))
    checkpoint_callback = ModelCheckpoint("models/personalized_treatment/checkpoints/best_model.h5", save_best_only=True, monitor='val_loss')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[checkpoint_callback])

    # Save the trained model
    model.save("models/personalized_treatment/final_model/model.h5")

if __name__ == "__main__":
    main()
