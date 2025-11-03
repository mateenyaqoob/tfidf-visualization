import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/sample_docs.csv')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

sns.clustermap(tfidf_df, cmap='coolwarm', figsize=(10,8))
plt.title('TF-IDF Cluster Map')
plt.show()
