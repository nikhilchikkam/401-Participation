import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Function to perform EDA
def perform_eda(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Display basic information about the dataset
    print("Data Information:")
    data.info()

    # Display basic statistics
    print("\nData Description:")
    print(data.describe())

    # Plotting distributions of key demographic variables
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data['age'], bins=20, kde=True, color='orange')
    plt.title('Age Distribution')

    plt.subplot(2, 2, 2)
    sns.histplot(data['educ'], bins=10, kde=True, color='blue')
    plt.title('Education Level Distribution')

    plt.subplot(2, 2, 3)
    sns.histplot(data['fsize'], bins=5, kde=True, color='green')
    plt.title('Family Size Distribution')

    plt.subplot(2, 2, 4)
    sns.countplot(x='marr', data=data, palette='Set2')
    plt.title('Marital Status')

    plt.tight_layout()
    plt.show()

    # Correlation heatmap to explore relationships between variables
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


# Define the path to your dataset
file_path = '401k_data.csv'

# Call the EDA function
perform_eda(file_path)
