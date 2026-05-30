import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'data/Mall_Customers.csv'
df = pd.read_csv(file_path)

# Encode categorical data
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# Select features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Save model
with open('models/kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

print('KMeans model saved successfully!')