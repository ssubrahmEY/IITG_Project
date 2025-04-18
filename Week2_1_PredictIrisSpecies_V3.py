import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

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

    return {"accuracy": accuracy, "precision": precision, "recall": recall}

def main():
    """
    Main function to load data, predict classes, display results, and calculate metrics.
    """
    # Define paths and model
    production_data_path = "/path/to/production_data.csv"  # Replace with actual path
    images_dir = "/path/to/images_dir"  # Replace with actual path
    model = load_trained_model()  # Replace with your model loading logic

    # Load production data
    production_df = load_production_data(production_data_path)

    # Extract features for prediction
    features_array = extract_features(production_df)  # Replace with your feature extraction logic

    # Predict and display results
    metrics = predict_and_display_results(production_df, features_array, model, images_dir)

    # Save metrics to a file (optional)
    metrics_output_path = "/path/to/metrics_output.json"  # Replace with actual path
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_output_path}")

if __name__ == "__main__":
    main()