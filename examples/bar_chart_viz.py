import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

df = pd.read_csv('data/sample_docs.csv')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
feature_names = vectorizer.get_feature_names_out()

# Average TF-IDF across all documents
avg_scores = X.mean(axis=0).A1
top_indices = avg_scores.argsort()[-10:][::-1]
top_terms = [feature_names[i] for i in top_indices]
top_values = avg_scores[top_indices]

plt.figure(figsize=(8, 5))
plt.barh(top_terms[::-1], top_values[::-1], color='skyblue')
plt.xlabel('TF-IDF Score')
plt.title('Top 10 TF-IDF Terms')
plt.tight_layout()
plt.show()
