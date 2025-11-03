import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('data/sample_docs.csv')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
feature_names = vectorizer.get_feature_names_out()

tfidf_scores = X.toarray().sum(axis=0)
word_freq = dict(zip(feature_names, tfidf_scores))

wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('TF-IDF WordCloud')
plt.show()
