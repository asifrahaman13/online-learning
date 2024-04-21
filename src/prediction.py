import torch

print(torch.__version__)
from src.model_creation import PricePredictor
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


def predict_output(cost, demand, recession, economy, competition, market_size):
    # Scale numerical variables
    scaler = StandardScaler()

    # Assuming X_new is a pandas DataFrame containing new input data for prediction
    # Preprocess the new input data
    encoder_recession = LabelEncoder()
    encoder_recession.classes_ = np.array(["False", "True"])  # Set the categories explicitly

    # Preprocessing
    # Encode categorical variables
    encoder = LabelEncoder()

    # Load the saved model for prediction
    model = PricePredictor(6)
    model.load_state_dict(torch.load("price_predictor_model.pth"))
    model.eval()

    # Assuming X_new is a pandas DataFrame containing new input data for prediction
    # Preprocess the new input data
    X_new = pd.DataFrame(
        {
            "Demand": demand,
            "Cost": cost,
            "Recession": recession,
            "Economy": economy,
            "Competition": competition,
            "Market Size": market_size,
        }
    )
    X_new["Recession"] = encoder_recession.fit_transform(X_new["Recession"])

    # Assuming X_new is a pandas DataFrame containing new input data for prediction
    # Preprocess the new input data
    encoder_economy = LabelEncoder()
    encoder_economy.classes_ = np.array(["Strong", "Moderate", "Weak"])  # Set the categories explicitly
    X_new["Economy"] = encoder_economy.fit_transform(X_new["Economy"])
    X_new["Competition"] = encoder.fit_transform(X_new["Competition"])
    X_new[["Demand", "Cost", "Market Size"]] = scaler.fit_transform(X_new[["Demand", "Cost", "Market Size"]])
    X_new_tensor = torch.tensor(X_new.values, dtype=torch.float32)

    # Make predictions using the loaded model
    with torch.no_grad():
        y_pred = model(X_new_tensor)

    # Convert predictions to a numpy array
    predictions = y_pred.numpy()

    # Print the predictions
    print("Predictions:")
    print(predictions)

    return predictions.tolist()
