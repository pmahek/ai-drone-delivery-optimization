import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

# load dataset
data = pd.read_csv("dataset.csv")

# -------------------------
# DATA CHECKING
# -------------------------

print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

# features and target
X = data[["distance_km","weight_kg","priority","weather"]]
y = data["delivery_time"]

# -------------------------
# KMEANS CLUSTERING
# -------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# -------------------------
# REGRESSION MODELS
# -------------------------

lr = LinearRegression()
lr.fit(X, y)

rf = RandomForestRegressor()
rf.fit(X, y)

# -------------------------
# SAVE MODELS
# -------------------------

pickle.dump(kmeans, open("kmeans.pkl","wb"))
pickle.dump(lr, open("linear_regression.pkl","wb"))
pickle.dump(rf, open("random_forest.pkl","wb"))

print("\nModels trained successfully!")