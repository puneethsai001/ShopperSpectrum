import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

segment_map = {
    0: 'At-Risk',
    1: 'Regular',
    2: 'High Value'
}

retail_df = pd.read_csv('Cleaned_online_retail.csv')

# Create User-Item Matrix
user_item_matrix = retail_df.pivot_table(index='CustomerID',
                                         columns='Description',
                                         values='Quantity',
                                         aggfunc='sum',
                                         fill_value=0)

# Compute Cosine Similarity between Products
item_similarity = cosine_similarity(user_item_matrix.T)

item_similarity_df = pd.DataFrame(item_similarity,
                                  index=user_item_matrix.columns,
                                  columns=user_item_matrix.columns)

# Load KMeans model
with open('Customer_Cluster_Model.pkl', 'rb') as f:
    loaded_kmeans = pickle.load(f)

# Load Scaler
with open('Scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Function to recommend similar products
def recommend_similar_products(item_name, top_n=6):

    if item_name not in item_similarity_df.index:
        print(f"Product '{item_name}' not found in dataset.")
        return []

    # Get similarity scores
    sim_scores = item_similarity_df[item_name].sort_values(ascending=False)

    similar_products = pd.DataFrame(sim_scores.head(top_n))
    return similar_products

# Function to classify using loaded model
def classify_customer(r, f, m):
    new_rfm = pd.DataFrame([[r, f, m]], columns=['Recency', 'Frequency', 'Monetary'])
    new_rfm_scaled = loaded_scaler.transform(new_rfm)
    cluster = loaded_kmeans.predict(new_rfm_scaled)[0]
    segment = segment_map.get(cluster, "Unknown")
    return cluster, segment

# Title
st.set_page_config(page_title="Shopper Spectrum Project", layout="wide")
st.title("Shopper Spectrum")
st.caption("by Puneeth Sai")

# Sidebar for toggling between pages
page = st.sidebar.selectbox("Choose a page", ["Customer Segmentation", "Product Recommendation"])

# Customer Segmentation Page
if page == "Customer Segmentation":
    st.header("Customer Segmentation")

    rec = st.number_input("Recency (days)", min_value=0)
    fre = st.number_input("Frequency (number of purchases)", min_value=0)
    mon = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button('Predict Cluster'):
        c, s = classify_customer(rec, fre, mon)
        st.write(c)
        st.write(s)


# Product Recommendation Page
elif page == "Product Recommendation":
    st.header("Product Recommendation")
    product_name = st.text_input("Enter product name")
    product_name = product_name.upper()

    if st.button("Get Recommendations"):
        recommendations = recommend_similar_products(product_name)

        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            st.subheader(f"Top Recommendations for '{product_name}':")
            recommendations.columns = ['Similarity Score']
            st.dataframe(recommendations)
        else:
            st.error("Product not found or no similar items.")





