# Enhancing Search Experience for Zepto

**Author:** Sarthak Kumar  
**Email:** [sarthakkumar3110@gmail.com](mailto:sarthakkumar3110@gmail.com)  
**Reg. No:** 20215089  

## Project Links
- **Google Colab:** [Link to Colab Notebook](https://colab.research.google.com/drive/12tEiKYBJaETcox9C81yaV-8z6t36DChs?usp=sharing)
- **Video Demonstration:** [Link to Video](https://drive.google.com/file/d/1IxEitB3kR1-ne7PEWUIBL-WtQJcmyOSk/view?usp=sharing)

## Table of Contents
1. [Introduction](#introduction)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Semantic Search with BERT and FAISS](#semantic-search-with-bert-and-faiss)
7. [Frontend with Streamlit](#frontend-with-streamlit)
8. [How to Run](#how-to-run)
9. [Future Enhancements](#future-enhancements)
10. [Conclusion](#conclusion)

## Introduction
In this project, we aim to enhance the search experience for Zepto by implementing a semantic search system. Traditional search methods often rely on keyword matching, which can result in irrelevant search results. To overcome this, we use BERT embeddings and FAISS for efficient and accurate semantic search.

## Objective
The objective is to develop a search system that understands the context and meaning of queries, providing users with more relevant search results. The system should be able to handle complex queries and return results that match the intent behind the query, rather than just keyword matches.

## Dataset
The dataset used in this project contains product information, including retail prices, discounted prices, images, descriptions, brands, product specifications, and product names. This dataset is processed to create vector embeddings for each product, which are then used for semantic search.

## Data Preprocessing
Data preprocessing involves the following steps:
1. **Filling Missing Values:** Columns like `retail_price`, `discounted_price`, `image`, `description`, `brand`, `product_specifications`, and `product_name` are processed to fill missing values.
2. **Normalization:** Numerical columns like `retail_price` and `discounted_price` are normalized to bring them to a common scale.
3. **Tokenization:** Text columns are tokenized and processed to remove stop words, punctuation, and special characters.

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) is performed to understand the distribution and relationships within the data. Key insights include:
- **Univariate Analysis:** Examines individual columns, such as the distribution of `retail_price` and `discounted_price`.
- **Bivariate Analysis:** Looks at relationships between two variables, such as `retail_price` vs. `discounted_price`.
- **Multivariate Analysis:** Investigates interactions between multiple variables to understand the overall trends in the data.

## Semantic Search with BERT and FAISS
We use BERT (Bidirectional Encoder Representations from Transformers) to create embeddings for the product descriptions. These embeddings capture the semantic meaning of the text, allowing us to compare the similarity between different products. FAISS (Facebook AI Similarity Search) is then used to efficiently store and search through these embeddings, enabling fast and accurate retrieval of relevant products based on user queries.

### Steps:
1. **Embedding Generation:** BERT model generates embeddings for each product description.
2. **Indexing with FAISS:** The embeddings are stored in a FAISS index, which allows for efficient similarity searches.
3. **Search Implementation:** Given a user query, the system generates an embedding and searches the FAISS index for similar products.

## Frontend with Streamlit
We developed a simple frontend using Streamlit to demonstrate the functionality of the semantic search system. The frontend allows users to input search queries and view the results in real-time, showcasing the relevance and accuracy of the search results.

## How to Run
To run the project locally:
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using the command `streamlit run app.py`.
4. Open the provided local URL in your browser to interact with the app.

## Future Enhancements
- **Improved UI:** Enhancing the user interface to make it more intuitive and user-friendly.
- **Advanced Search Features:** Implementing filters and sorting options to refine search results.
- **Real-time Data Updates:** Allowing the system to update product data in real-time for dynamic search results.

## Conclusion
This project demonstrates the potential of semantic search in improving user experience. By leveraging BERT embeddings and FAISS, we can create a search system that understands the intent behind queries and returns highly relevant results, significantly enhancing the traditional search experience.
