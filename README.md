# ShopperSpectrum

The project mainly deals with online retail data where it contains the records of customers, invoices, products, dates, etc. The main objective is to analyze the data using exploratory data analysis to find some insights with graphs, cluster the customers based on high value, at risk and regular by computing RFM values and using KMeans clustering taking n=3 (Found optimal from the elbow curve) and generate a product heatmap based on the correlation between them by generating user-item pivot table and calculating the cosine similarity between them and plotting it.

In addition to this analysis, the clustering model which is to be exported and the algorithm to find the similarity (using cosine similarity) is used to develop a web application using StreamLit that has two pages. First is the customer segmentation module that essentially takes recency, frequency and monetary as the input and predicts which cluster (High value, At risk or Regular) the customer belongs to based on the data trained using the dataset. Second is the product recommendation module that takes the product name (Not case-sensitive) as the input and generates a list of 5 similar products based on the cosine similarity that is calculated during the heatmap plot.

The Final outputs are the python notebook **Shopper_Spectrum.ipynb** that contains the main analysis phase, model creation and testing and the StreamLit python file **Index.py** that consists of importing the model and using it to develop the application with the mentioned features.

Other Files
* online_retail.csv - Initial dataset
* Cleaned_online_retail.csv - Cleaned dataset obtained after data preprocessing
* Customer_Cluster_Model.pk - KMeans model trained using the dataset
* Scaler.pkl - Scaler used on the dataset