import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam.csv")

X = df['message']
y = df['label']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X,y)

print(model.predict(vectorizer.transform(["Win money now"])))
