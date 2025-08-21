import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import nltk
import string

# Download stopwords (only first time needed)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("data/Fake.csv")

print("Columns in dataset:", data.columns)

# Combine title + text
data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")

# Clean text function
stop_words = set(stopwords.words("english"))
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data["clean_content"] = data["content"].apply(clean_text)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["clean_content"])

# Dimensionality reduction for visualization
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

# KMeans clustering
k = 5  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
data["cluster"] = kmeans.fit_predict(X_reduced)

# Show cluster counts
print("\nCluster distribution:")
print(data["cluster"].value_counts())

# Plot clusters
plt.figure(figsize=(8,6))
sns.countplot(x=data["cluster"], palette="viridis")
plt.title("Fake News Clusters")
plt.xlabel("Cluster")
plt.ylabel("Number of Articles")
plt.show()

# Show sample articles from each cluster
for i in range(k):
    print(f"\n🔹 Sample articles from Cluster {i}:")
    print(data[data["cluster"] == i]["title"].head(3).tolist())
