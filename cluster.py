import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

with open("ProductNames.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

if "description" in df.columns:
    df["description"] = df["description"].fillna("")
else:
    df["description"] = ""

df["text"] = df["name"].astype(str) + " " + df["description"].astype(str)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)
X = vectorizer.fit_transform(df["text"])

scores = {}
print("Silhouette scores by k:")
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores[k] = score
    print(f"k = {k}: {score:.4f}")

best_k = max(scores, key=scores.get)
print(f"\nBest k by silhouette: {best_k} (score = {scores[best_k]:.4f})")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df["cluster"] = clusters

for c in range(best_k):
    print(f"\n=== Cluster {c} ===")
    print(df[df["cluster"] == c]["name"])

pca = PCA(n_components=2, random_state=42)
X_dense = X.toarray()
X_2d = pca.fit_transform(X_dense)

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap="tab10", alpha=0.7)
plt.title(f"Product Clusters (PCA 2D, k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
