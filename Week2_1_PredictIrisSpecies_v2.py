import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Load the Iris dataset
iris = load_iris()

# Create a DataFrame with features
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the class names to the DataFrame
iris_df['class'] = iris.target

# Rename the columns for better readability
iris_df.rename(columns={
    'sepal length (cm)': 'SL (cm)',
    'sepal width (cm)': 'SW (cm)',
    'petal length (cm)': 'PL (cm)',
    'petal width (cm)': 'PW (cm)',
}, inplace=True)

# Separate features (X) and target (y)
X = iris_df[['SL (cm)', 'SW (cm)', 'PL (cm)', 'PW (cm)']]
y = iris_df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

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


# Modify the features array to match the 4 features of each species
# Define your custom features array (replace with your own values)
# Assuming there are 4 features per image and the number of rows matches production_df
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

# Predict the class for each image using the model
production_df["predicted_class"] = model.predict(features_array)

# Display the results as a table with images
fig, ax = plt.subplots(len(production_df), 1, figsize=(5, len(production_df) * 3))
for i, row in production_df.iterrows():
    image = Image.open(os.path.join(images_dir, row["filename"]))
    ax[i].imshow(image)
    ax[i].axis("off")
    ax[i].set_title(
        f"Actual: {row['actual_class']}, Predicted: {row['predicted_class']}"
    )
plt.tight_layout()
plt.show()

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(production_df["actual_class"], production_df["predicted_class"])
precision = precision_score(production_df["actual_class"], production_df["predicted_class"], average="weighted")
recall = recall_score(production_df["actual_class"], production_df["predicted_class"], average="weighted")

# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
# Plot the confusion matrix
conf_matrix = confusion_matrix(
    production_df["actual_class"], production_df["predicted_class"]
)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Production Data")
plt.show()