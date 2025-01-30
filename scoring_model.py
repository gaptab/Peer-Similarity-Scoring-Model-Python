import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Generating Dummy Data
np.random.seed(42)
num_assets = 10  # Number of assets

# Creating a synthetic dataset of financial KPIs
data = pd.DataFrame({
    "Asset_ID": [f"Asset_{i}" for i in range(1, num_assets + 1)],
    "Revenue": np.random.randint(100, 1000, num_assets),
    "Profit_Margin": np.random.uniform(0.05, 0.3, num_assets),
    "Debt_to_Equity": np.random.uniform(0.1, 1.5, num_assets),
    "Return_on_Assets": np.random.uniform(0.02, 0.15, num_assets),
    "Market_Cap": np.random.randint(500, 5000, num_assets)
})

# Normalizing financial indicators for similarity calculation
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=["Asset_ID"]))

# Computing Cosine Similarity Matrix
similarity_matrix = cosine_similarity(scaled_data)

# Creating a DataFrame for similarity scores
similarity_df = pd.DataFrame(similarity_matrix, index=data["Asset_ID"], columns=data["Asset_ID"])

# Visualizing the Similarity Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Peer Similarity Score (Cosine Similarity)")
plt.show()

# Saving the dataset and similarity scores to CSV
data.to_csv("asset_kpi_data.csv", index=False)
similarity_df.to_csv("peer_similarity_scores.csv")
