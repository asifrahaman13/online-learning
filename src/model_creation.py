import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Define a simple neural network for price prediction
class PricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model():
    # Load the dataset
    data = pd.read_csv("/media/asifr/work/ml_project/backend/src/service_pricing_data.csv")

    # Preprocessing
    # Encode categorical variables
    encoder = LabelEncoder()
    data["Recession"] = encoder.fit_transform(data["Recession"])
    data["Economy"] = encoder.fit_transform(data["Economy"])
    data["Competition"] = encoder.fit_transform(data["Competition"])

    # Scale numerical variables
    scaler = StandardScaler()
    data[["Demand", "Cost", "Market Size"]] = scaler.fit_transform(
        data[["Demand", "Cost", "Market Size"]]
    )

    # Split the dataset into training and testing sets
    X = data.drop("Price", axis=1)
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

    # Model parameters
    input_dim = X_train.shape[1]

    # Create an instance of the model
    model = PricePredictor(input_dim)

    def train_model():

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training the model
        num_epochs = 1000
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Save the trained model
        torch.save(model.state_dict(), "/media/asifr/work/ml_project/backend/src/price_predictor_model.pth")

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            test_loss = criterion(y_pred, y_test_tensor)
            print(f"Test Loss: {test_loss.item():.4f}")

    count = 0
    while True:

        count = count + 1
        # if count == 3:
        #     break
        print("First training model......")
        import time

        train_model()
        time.sleep(10)


if __name__ == "__main__":
    train_model()
