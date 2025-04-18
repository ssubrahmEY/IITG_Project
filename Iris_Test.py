# Iris_Test.py - This script loads a pre-trained model, makes predictions on test data, and evaluates the model's accuracy.

# Import necessary libraries

from Iris_Train import train_model
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Specify the path to save the model
model_path = "models/iris_model.pkl"

# Load the Iris dataset for splitting test data
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained model if it exists
try:
    # Try to load the pre-trained model
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
except FileNotFoundError:
    # If the model doesn't exist, train a new one and save it
    print(f"Model not found. Training a new model...")
    model, _, _ = train_model(output_path=model_path)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")