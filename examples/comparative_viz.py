import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/sample_docs.csv')

tech_docs = df[df['category'] == 'Technology']['text']
business_docs = df[df['category'] == 'Business']['text']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(pd.concat([tech_docs, business_docs]))
terms = vectorizer.get_feature_names_out()
tfidf_matrix = pd.DataFrame(X.toarray(), columns=terms)

tech_avg = tfidf_matrix.iloc[:len(tech_docs)].mean()
bus_avg = tfidf_matrix.iloc[len(tech_docs):].mean()

diff = (tech_avg - bus_avg).sort_values(ascending=False)[:10]

plt.figure(figsize=(8,5))
diff.plot(kind='bar', color='coral')
plt.title('Top Terms (Tech vs Business)')
plt.ylabel('TF-IDF Difference')
plt.show()
