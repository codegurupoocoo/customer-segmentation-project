# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import os

# Define file paths
customer_file = r'C:\Users\hp\OneDrive\Desktop\ass ds\Customers.csv'
product_file = r'C:\Users\hp\OneDrive\Desktop\ass ds\Products.csv'
transaction_file = r'C:\Users\hp\OneDrive\Desktop\ass ds\Transactions.csv'
lookalike_file = r'C:\Users\hp\OneDrive\Desktop\ass ds\Lookalike.csv'

# Check if files exist
for file in [customer_file, product_file, transaction_file]:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        exit()

# Load the datasets
customers = pd.read_csv(customer_file)
products = pd.read_csv(product_file)
transactions = pd.read_csv(transaction_file)

# Display the first few rows of each dataset
print(customers.head())
print(products.head())
print(transactions.head())

# Check for missing values
print("Missing values in Customers:\n", customers.isnull().sum())
print("Missing values in Products:\n", products.isnull().sum())
print("Missing values in Transactions:\n", transactions.isnull().sum())

# Display summary statistics
print(customers.describe(include='all'))
print(products.describe())
print(transactions.describe())

# Convert TransactionDate to datetime
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], errors='coerce')

# Handle any potential missing dates
transactions.dropna(subset=['TransactionDate'], inplace=True)

# Example: Sales over time
sales_over_time = transactions.groupby('TransactionDate')['TotalValue'].sum()

plt.figure(figsize=(12, 6))
plt.plot(sales_over_time)
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# Merge datasets to get a complete view
merged_data = pd.merge(transactions, customers, on='CustomerID', how='inner')
customer_profiles = merged_data.groupby(['CustomerID', 'Region', 'SignupDate']).agg({
    'TotalValue': 'sum',
    'Quantity': 'sum'
}).reset_index()

# Create a feature matrix
features = customer_profiles[['TotalValue', 'Quantity']]

# Clustering with KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
customer_profiles['Cluster'] = kmeans.fit_predict(features)

# Calculate DB Index
db_index = davies_bouldin_score(features, customer_profiles['Cluster'])
print(f'Davies-Bouldin Index: {db_index}')

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalValue', y='Quantity', hue='Cluster', data=customer_profiles, palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Total Value')
plt.ylabel('Quantity')
plt.legend(title='Cluster')
plt.show()

# Create similarity matrix for lookalike model
similarity_matrix = cosine_similarity(features)

# Function to get top 3 lookalikes
def get_lookalikes(customer_id, similarity_matrix):
    customer_idx = customer_profiles[customer_profiles['CustomerID'] == customer_id].index[0]
    similar_indices = np.argsort(similarity_matrix[customer_idx])[-4:-1]  # Top 3 lookalikes
    return customer_profiles.iloc[similar_indices]['CustomerID'].tolist()

# Generate lookalikes for the first 20 customers
lookalikes = {}
for customer_id in customer_profiles['CustomerID'][:20]:
    lookalikes[customer_id] = get_lookalikes(customer_id, similarity_matrix)

# Save to CSV
lookalike_df = pd.DataFrame.from_dict(lookalikes, orient='index').reset_index()
lookalike_df.columns = ['CustomerID', 'Lookalike1', 'Lookalike2', 'Lookalike3']
lookalike_df.to_csv(lookalike_file, index=False)

print(f"Lookalikes saved to: {lookalike_file}")
