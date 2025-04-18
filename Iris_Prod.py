import os
from Iris_Train import train_model
import joblib
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import numpy as np


def load_production_data(file_path):
    """
    Loads production data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing production data.

    Returns:
        DataFrame: Loaded production data.
    """
    return pd.read_csv(file_path)

def predict_and_display_results(production_df, features_array, model, images_dir):
    """
    Predicts the class for each image, displays the results with images, 
    and calculates accuracy, precision, and recall.

    Args:
        production_df (DataFrame): DataFrame containing production data with actual classes and filenames.
        features_array (ndarray): Array of features for prediction.
        model (object): Trained model used for prediction.
        images_dir (str): Directory containing the image files.

    Returns:
        dict: Dictionary containing accuracy, precision, and recall metrics.
    """
    # Predict the class for each image using the model
    production_df["predicted_class"] = model.predict(features_array)

    # Display the results as a table with images
    fig, ax = plt.subplots(len(production_df), 1, figsize=(5, len(production_df) * 3))
    for i, row in production_df.iterrows():
        # Open the image file
        image_path = os.path.join(images_dir, row["filename"])
        image = Image.open(image_path)

        # Display the image with prediction and actual class
        ax[i].imshow(image)
        ax[i].axis("off")
        ax[i].set_title(
            f"Actual: {row['actual_class']}, Predicted: {row['predicted_class']}"
        )

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Calculate metrics
    accuracy = accuracy_score(production_df["actual_class"], production_df["predicted_class"])
    precision = precision_score(production_df["actual_class"], production_df["predicted_class"], average="weighted")
    recall = recall_score(production_df["actual_class"], production_df["predicted_class"], average="weighted")

    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Plot the confusion matrix
    conf_matrix = confusion_matrix(production_df["actual_class"], production_df["predicted_class"]
)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Production Data")
    plt.show()

    return {"accuracy": accuracy, "precision": precision, "recall": recall}

def main():
    """
    Main function to load data, predict classes, display results, and calculate metrics.
    """
    # Define paths and model
    # Production test section

    print("\nProduction Test Results:")

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the absolute path to the images directory
    images_dir = os.path.join(script_dir, "images")

    # Define the image data with filenames pointing to the images directory
    image_data = [
        {"filename": os.path.join(images_dir, "Seimage1.jpeg"), "actual_class": 0},
        {"filename": os.path.join(images_dir, "Viimage2.jpeg"), "actual_class": 2},
        {"filename": os.path.join(images_dir, "Viimage2.png"), "actual_class": 2},
        {"filename": os.path.join(images_dir, "Veimage5.jpeg"), "actual_class": 1},
        {"filename": os.path.join(images_dir, "Viimage4.jpg"), "actual_class": 2},
        {"filename": os.path.join(images_dir, "Seimage3.jpeg"), "actual_class": 0},
    ]

    # Prepare a DataFrame for production data
    production_df = pd.DataFrame(image_data)

  
    # Specify the path to save the model
    model_path = "models/iris_model.pkl"

    # Load the pre-trained model if it exists
    try:
        # Try to load the pre-trained model
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        # If the model doesn't exist, train a new one and save it
        print(f"Model not found. Training a new model...")
        # Train the model and save it
        model, _, _ = train_model(output_path=model_path)



    # Extract features for prediction
    custom_features_array = np.array([
    [5.7, 3.8, 1.7, 0.2],  # Example values for the first image
    [7.3, 2.9, 6.6, 2.1],  # Example values for the second image
    [6.9, 2.8, 5.1, 2.4],  # Example values for the third image
    [6.0, 3.2, 3.8, 1.3],  # Example values for the fourth image
    [7.5, 2.7, 6.3, 2.5],  # Example values for the fifth image
    [4.8, 3.4, 1.5, 0.1],  # Example values for the sixth image  
])

    # Assign your custom array to features_array
    features_array = custom_features_array

    print(f"Custom Features array of production data is {features_array}")

    # Predict and display results
    metrics = predict_and_display_results(production_df, features_array, model, images_dir)

    # Save metrics to a file (optional)
    metrics_output_path = "outputs/metrics_output.json"  # Replace with actual path
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_output_path}")

if __name__ == "__main__":
    main()