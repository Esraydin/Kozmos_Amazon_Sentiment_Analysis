# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:32:28 2025

@author: aydin
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x: '%.2f' % x)
pd.set_option('display.width',200)

#%%
# TEXT PREPROCESSING 


df = pd.read_excel("amazon.xlsx")
df.head()
df.info()

# Normalizing Case Folding

df['Review'] = df['Review'].str.lower()

# Punctuations
df['Review'] = df['Review'].str.replace('[^\w\s]','')

# Numbers 
df['Review'] = df['Review'].str.replace('\d','')

# Stopwords
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


#Rarewords / Custom Words

sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# Lemmazation

df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df['Review'].head(10)


#%%
# Visualization

# Barplot

from collections import Counter

# Tüm incelemeleri birleştirip kelime frekansını hesaplayın
all_words = " ".join(df['Review']).split()
word_counts = Counter(all_words)

# DataFrame'e dönüştürün
tf = pd.DataFrame(word_counts.items(), columns=["words", "tf"])

# 500'den fazla frekanslı kelimeleri filtreleyin
tf_filtered = tf[tf["tf"] > 500]

# Çubuk grafik çizimi
tf_filtered.plot.bar(x="words", y="tf", figsize=(12, 6))
plt.show()


# Wordcloud

text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                       max_words=100,
                       background_color="white").generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Sentiment Analysis

df.head()

sia = SentimentIntensityAnalyzer()

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["Sentiment_label"] =df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df.groupby("Sentiment_label")["Star"].mean()


#%%

# Train-Test-Split

x_train, x_test, y_train, y_test = train_test_split(df["Review"],
                                                    df["Sentiment_label"],
                                                    random_state = 42)

# TF - IDF Word Level
tf_idf_word_vectorizer = TfidfVectorizer().fit(x_train)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(x_train)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(x_test)

# Modelling (Logistic Regression)

log_model = LogisticRegression().fit(x_train_tf_idf_word, y_train)

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, y_test))

cross_val_score(log_model,x_test_tf_idf_word, y_test, cv=5).mean()


random_review = pd.Series(df["Review"].sample(1).values)
yeni_yorum = CountVectorizer().fit(x_train).transform(random_review)
pred = log_model.predict(yeni_yorum)
print(f'Review: {random_review[0]} \n Prediction: {pred}')

# Random Forest Classification

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, y_train) # String verilerin vektörize edilmesi.
cross_val_score(rf_model, x_test_tf_idf_word, y_test, cv=5, n_jobs=1).mean()












