import numpy as np
import pandas as pd

# Load the dataset
file_path = 'flipkart_com-ecommerce_sample.csv'
df = pd.read_csv(file_path)

# Handle missing values
df['retail_price'].fillna(df['retail_price'].mean(), inplace=True)
df['discounted_price'].fillna(df['discounted_price'].mean(), inplace=True)
df['image'].fillna('No Image', inplace=True)
df['description'].fillna('No Description', inplace=True)
df['brand'].fillna('Unknown', inplace=True)
df['product_specifications'].fillna('No Specifications', inplace=True)
df['product_name'] = df['product_name'].fillna('').astype(str)
df['description'] = df['description'].fillna('').astype(str)
df['product_category_tree'] = df['product_category_tree'].fillna('Unknown').astype(str)
df_subset = df.head(20000)

# Heuristic Search Function
def heuristic_search(query, df_subset, top_n=5):
    query = query.lower().split()    
    # Initialize a score column
    df_subset['heuristic_score'] = 0
    # Keyword Matching
    for word in query:
        df_subset['heuristic_score'] += df_subset['product_name'].str.lower().str.contains(word).astype(int)
        df_subset['heuristic_score'] += df_subset['description'].str.lower().str.contains(word).astype(int)
    # Price Heuristic
    if "cheap" in query:
        df_subset = df_subset.sort_values(by='discounted_price', ascending=True)
    elif "expensive" in query:
        df_subset = df_subset.sort_values(by='discounted_price', ascending=False)
    category_match = df_subset['product_category_tree'].str.lower().str.contains('|'.join(query))
    df_subset.loc[category_match, 'heuristic_score'] += 1
    df_sorted = df_subset.sort_values(by='heuristic_score', ascending=False)     # Sort by heuristic score
    return df_sorted.head(top_n)[['product_name', 'description', 'product_category_tree', 'discounted_price']]     # Return the top N results

# Example heuristic search query
query = "cheap women clothing"
heuristic_results = heuristic_search(query, df_subset)
print("Heuristic search results:")
print(heuristic_results)
