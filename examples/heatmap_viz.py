import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/sample_docs.csv')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

plt.figure(figsize=(10, 6))
sns.heatmap(tfidf_df, cmap='YlGnBu')
plt.title('TF-IDF Heatmap (Documents × Terms)')
plt.xlabel('Terms')
plt.ylabel('Documents')
plt.show()
