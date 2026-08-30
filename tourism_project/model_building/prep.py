
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH)

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split the dataset into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save train/test files expected by the GitHub Actions workflow
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)

ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print(f"Xtrain shape: {Xtrain.shape}")
print(f"Xtest shape: {Xtest.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"ytest shape: {ytest.shape}")
