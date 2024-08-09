# import numpy as np
# import pandas as pd
# import faiss
# from transformers import BertTokenizer, TFBertModel
# import tensorflow as tf

# # Load the dataset
# file_path = 'flipkart_com-ecommerce_sample.csv'
# df = pd.read_csv(file_path)

# # Handle missing values
# df['retail_price'].fillna(df['retail_price'].mean(), inplace=True)
# df['discounted_price'].fillna(df['discounted_price'].mean(), inplace=True)
# df['image'].fillna('No Image', inplace=True)
# df['description'].fillna('No Description', inplace=True)
# df['brand'].fillna('Unknown', inplace=True)
# df['product_specifications'].fillna('No Specifications', inplace=True)
# df['product_name'] = df['product_name'].fillna('').astype(str)
# df['description'] = df['description'].fillna('').astype(str)
# df['product_category_tree'].fillna('Unknown Category', inplace=True)  # Handle missing values for product_category_tree

# # Ensure there are no null values in the columns used for embeddings
# required_columns = ['product_name', 'description', 'product_category_tree']
# for col in required_columns:
#     if df[col].isnull().any():
#         raise ValueError(f"Column '{col}' contains null values. Please handle them before proceeding.")

# # Define function to get BERT embeddings
# def get_bert_embeddings(text):
#     if not isinstance(text, str) or text == "":
#         raise ValueError("Input text must be a non-empty string.")
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = TFBertModel.from_pretrained('bert-base-uncased')
#     inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
#     outputs = model(inputs)
#     pooled_output = outputs.last_hidden_state[:, 0, :].numpy()
#     return pooled_output

# df_subset = df.head(20000).copy()

# # Generate embeddings for product names and descriptions for the subset
# df_subset['name_embeddings'] = df_subset['product_name'].apply(get_bert_embeddings)
# df_subset['description_embeddings'] = df_subset['description'].apply(get_bert_embeddings)

# # Create a combined column for embedding
# df_subset['combined_info'] = df_subset.apply(
#     lambda row: f"{row['product_name']} {row['description']} {row['product_category_tree']}",
#     axis=1
# )

# # Generate embeddings for the combined information
# df_subset['combined_embeddings'] = df_subset['combined_info'].apply(get_bert_embeddings)

# # Convert embeddings to a numpy array
# combined_embeddings_matrix = np.vstack(df_subset['combined_embeddings'].values)

# # Create a FAISS index
# d = combined_embeddings_matrix.shape[1]  # Dimension of embeddings
# index = faiss.IndexFlatL2(d)  # Using L2 (Euclidean) distance for similarity

# # Add embeddings to the index
# index.add(combined_embeddings_matrix)

# # Save the index to a file
# faiss.write_index(index, 'product_embeddings.index')

# # Save the dataframe to a file (to preserve associated metadata)
# df_subset.to_pickle('product_metadata.pkl')
import numpy as np
import pandas as pd
import faiss
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

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
df['product_category_tree'].fillna('Unknown Category', inplace=True)  # Handle missing values for product_category_tree

# Ensure there are no null values in the columns used for embeddings
required_columns = ['product_name', 'description', 'product_category_tree']
for col in required_columns:
    if df[col].isnull().any():
        raise ValueError(f"Column '{col}' contains null values. Please handle them before proceeding.")

# Define function to get BERT embeddings in batches
def get_bert_embeddings_batch(text_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text_list, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    pooled_output = outputs.last_hidden_state[:, 0, :].numpy()
    return pooled_output

df_subset = df.head(20000).copy()

# Create a combined column for embedding
df_subset['combined_info'] = df_subset.apply(
    lambda row: f"{row['product_name']} {row['description']} {row['product_category_tree']}",
    axis=1
)

# Generate embeddings in batches of 32
batch_size = 32
combined_embeddings = []

for i in range(0, len(df_subset), batch_size):
    batch_texts = df_subset['combined_info'].iloc[i:i + batch_size].tolist()
    batch_embeddings = get_bert_embeddings_batch(batch_texts)
    combined_embeddings.append(batch_embeddings)

# Convert the list of batches to a numpy array
combined_embeddings_matrix = np.vstack(combined_embeddings)

# Add the embeddings to the DataFrame
df_subset['combined_embeddings'] = list(combined_embeddings_matrix)

# Create a FAISS index
d = combined_embeddings_matrix.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)  # Using L2 (Euclidean) distance for similarity

# Add embeddings to the index
index.add(combined_embeddings_matrix)

# Save the index to a file
faiss.write_index(index, 'product_embeddings.index')

# Save the dataframe to a file (to preserve associated metadata)
df_subset.to_pickle('product_metadata.pkl')
