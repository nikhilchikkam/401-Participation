from doubleml.datasets import fetch_401K

# Fetch the 401(k) dataset and return it as a pandas DataFrame
data = fetch_401K(return_type='DataFrame', polynomial_features=False)

# Display the fetched data
print(data.head())

# Save the dataset to a CSV file
data.to_csv('401k_data.csv', index=False)

print("Data saved to 401k_data.csv")