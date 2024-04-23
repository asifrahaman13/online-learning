import torch

print(torch.__version__)
from src.model_creation import PricePredictor
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model_va import PricePredictor

# Load the trained model
model = PricePredictor(6)
model.load_state_dict(torch.load("price_predictor_model.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_output(cost, demand, recession, economy, competition, market_size):
    print(cost, demand, recession, economy, competition, market_size)
    # cost, demand, recession, economy, competition, market_size=cost[0], demand[0], recession[0], economy[0], competition[0], market_size[0]

    # Preprocess the input data
    input_data = {
        "Demand": [demand, 4784, 1500, 2500, 3500, 2000, 4200, 1800, 3000, 2700],
        "Cost": [cost, 2486, 700, 1200, 1800, 900, 2000, 800, 1500, 1300],
        "Recession": [
            recession,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
        ],
        "Economy": [
            economy,
            "Medium",
            "Strong",
            "Strong",
            "Medium",
            "Weak",
            "Strong",
            "Medium",
            "Weak",
            "Strong",
        ],
        "Competition": [
            competition,
            "Low",
            "Low",
            "High",
            "Medium",
            "High",
            "Low",
            "Medium",
            "High",
            "Low",
        ],
        "Market Size": [
            market_size,
            3502,
            8000,
            12000,
            15000,
            9000,
            18000,
            7000,
            14000,
            10000,
        ],
    }
    input_df = pd.DataFrame(input_data)

    encoder = LabelEncoder()
    input_df["Recession"] = encoder.fit_transform(input_df["Recession"])
    input_df["Economy"] = encoder.fit_transform(input_df["Economy"])
    input_df["Competition"] = encoder.fit_transform(input_df["Competition"])

    scaler = StandardScaler()
    input_df[["Demand", "Cost", "Market Size"]] = scaler.fit_transform(
        input_df[["Demand", "Cost", "Market Size"]]
    )

    # Convert input data to PyTorch tensor
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

    # Get the predicted prices
    with torch.no_grad():
        predicted_prices = model(input_tensor).cpu().numpy().flatten()
    prediction = 0
    # Print the predicted prices
    for i, price in enumerate(predicted_prices):
        print(f"Predicted price for sample {i + 1}: {price:.4f}")
        print(type(price))
        # return price
        prediction = int(price)
        return prediction
        # prediction= price:.4f

    return prediction


predict_output(2959, 3986, True, "Weak", "High", 3500)
