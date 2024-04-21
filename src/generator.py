import pandas as pd
import numpy as np


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of samples
    num_samples = 1000

    # Generate synthetic data for demand, cost, price, recession, economy, competition, and market size
    demand = np.random.randint(50, 200, size=num_samples)
    cost = np.random.randint(10, 50, size=num_samples)
    price = 50 + 2 * demand - 1.5 * cost + np.random.normal(0, 10, size=num_samples)
    recession = np.random.choice([True, False], size=num_samples)
    economy = np.random.choice(["Strong", "Moderate", "Weak"], size=num_samples)
    competition = np.random.choice(["High", "Medium", "Low"], size=num_samples)
    market_size = np.random.randint(1000, 10000, size=num_samples)

    # Create a DataFrame
    data = pd.DataFrame(
        {
            "Demand": demand,
            "Cost": cost,
            "Price": price,
            "Recession": recession,
            "Economy": economy,
            "Competition": competition,
            "Market Size": market_size,
        }
    )

    # Save the DataFrame to a CSV file
    data.to_csv("backend/service_pricing_data.csv", index=False)

    print("Dummy CSV file generated successfully.")


if __name__ == "__main__":
    main()
