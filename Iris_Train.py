# Train.py 
# This script trains a Random Forest classifier on the Iris dataset and saves the model to a file.
# It also includes a function to load the model and make predictions on new data.

# Import the required libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

def train_model(output_path="models/iris_model.pkl"):
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    print("Training the model...")
    model.fit(X_train, y_train)
    print

    # Save the trained model to the specified output path
    try:
    # Your training and saving code here
        joblib.dump(model, output_path)
        print(f"Model trained and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Return the model and test data
    return model, X_test, y_test

# Call the function in the main block
if __name__ == "__main__":
    train_model()