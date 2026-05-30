import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation using Machine Learning")
st.write("Segment customers based on demographics and purchasing behavior")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Handle categorical columns
    label_encoder = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])

    st.subheader("Data Summary")
    st.write(df.describe())

    # Feature selection
    features = st.multiselect(
        "Select Features for Clustering",
        options=df.columns,
        default=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    )

    if len(features) > 1:
        X = df[features]

        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select number of clusters
        k = st.slider("Select Number of Clusters", 2, 10, 5)

        # KMeans Model
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters

        st.subheader("Clustered Data")
        st.dataframe(df.head())

        # Visualization
        if len(features) >= 2:
            fig = px.scatter(
                df,
                x=features[0],
                y=features[1],
                color=df['Cluster'].astype(str),
                title='Customer Segments'
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Insights")

        cluster_summary = df.groupby('Cluster')[features].mean()
        st.dataframe(cluster_summary)

        st.success("Customer segmentation completed successfully!")

else:
    st.info("Please upload a CSV dataset to continue")