# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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