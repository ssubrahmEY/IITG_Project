#Testing the exercises in class

# Testing the exercises in class

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame with features
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the class names to the DataFrame
iris_df['class'] = [str(iris.target_names[label]) for label in iris.target]

# Rename the columns
iris_df.rename(columns={
    'sepal length (cm)': 'SL (cm)',
    'sepal width (cm)': 'SW (cm)',
    'petal length (cm)': 'PL (cm)',
    'petal width (cm)': 'PW (cm)',
    'class': 'Species',
}, inplace=True)

# Print the characteristics of the dataframe
print(f"The summary of the dataframe is {iris_df.info()}")
print(f"The statistical summary of the dataframe is {iris_df.describe()}")

# Print the first 5 rows and last 5 rows
print(iris_df.head())
print(iris_df.tail())

# Find the unique species and their count
unique_species = iris_df['Species'].unique().tolist()
unique_species_count = len(unique_species)

# Print the unique species and their count
print(f"Number of unique species: {unique_species_count}")
print(f"Unique species: {unique_species}")

# Display statistical summary per species
grouped_summary = iris_df.groupby('Species').describe()
print(f"Statistical summary per species:\n{grouped_summary}")

import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

# KDE plot for sepal length with legend explicitly shown for the hue displayed for each of the species
plt.figure(figsize=(8, 6))
kde_plot = sns.kdeplot(
    data=iris_df, 
    x='SL (cm)', 
    hue='Species', 
    fill=True, 
    common_norm=False, 
    alpha=0.5
)

# Add title and labels
plt.title('KDE Plot for Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Density')

# Show the plot
plt.show()

# KDE plot with histogram for sepal length with legend explicitly shown for the hue displayed for each of the species

import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with three subplots laid out horizontally
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# List of species
species_list = iris_df['Species'].unique()

# Loop through each species and create a histogram with KDE for each
for i, species in enumerate(species_list):
    sns.histplot(
        data=iris_df[iris_df['Species'] == species], 
        x='SL (cm)', 
        kde=True, 
        ax=axes[i], 
        color=sns.color_palette("husl")[i]
    )
    axes[i].set_title(f'{species} - Sepal Length')
    axes[i].set_xlabel('Sepal Length (cm)')
    axes[i].set_ylabel('Density')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()