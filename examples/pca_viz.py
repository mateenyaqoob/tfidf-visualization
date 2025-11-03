import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

df = pd.read_csv('data/sample_docs.csv')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

pca = PCA(n_components=2)
reduced = pca.fit_transform(X.toarray())

plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=pd.factorize(df['category'])[0], cmap='viridis')
for i, txt in enumerate(df['category']):
    plt.annotate(txt, (reduced[i,0], reduced[i,1]))
plt.title('PCA Visualization of Documents')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
