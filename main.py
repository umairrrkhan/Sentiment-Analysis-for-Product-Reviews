import pandas as pd
import numpy as np
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load the CSV file into a pandas dataframe
df = pd.read_csv('reviews.csv')

# drop any rows with missing data
df.dropna(inplace=True)

# select only the 'review_text' and 'sentiment' columns
df = df[['review_text', 'sentiment']]

# preprocess the text data by removing special characters, stop words, and converting all text to lowercase
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['review_text'] = df['review_text'].apply(preprocess_text)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review_text'], df['sentiment'], test_size=0.2, random_state=42)

# convert the text data into numerical vectors using the bag of words technique
cv = CountVectorizer(max_features=5000)
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# use the TF-IDF technique to convert the text data into numerical vectors
tv = TfidfVectorizer(max_features=5000)
X_train = tv.fit_transform(X_train.tolist())
X_test = tv.transform(X_test.tolist())

# train the model using a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# use the trained model to predict the sentiment of the test dataset
y_pred = clf.predict(X_test)

# evaluate the performance of the model using various metrics
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('Classification Report: \n', classification_report(y_test, y_pred))

# We can now train the model using a Naive Bayes classifier

clf = MultinomialNB()
clf.fit(X_train, y_train)

# We can use the trained model to predict the sentiment of the test dataset

y_pred = clf.predict(X_test)

# we can evaluate the performance of the model using various metrics

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('Classification Report: \n', classification_report(y_test, y_pred))

# We can now use the trained model to predict the sentiment of new reviews

new_reviews = ['This product is amazing', 'This product is terrible', 'I love this product', 'I hate this product']
new_reviews = [preprocess_text(review) for review in new_reviews]
new_reviews_vectors = tv.transform(new_reviews)
new_reviews_sentiment = clf.predict(new_reviews_vectors)

print('Predicted Sentiment of new reviews: ', new_reviews_sentiment)

#Let's visualize the confusion matrix using a heatmap

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Let's visualize the most important features for each sentiment

feature_names = cv.get_feature_names_out()

def show_most_informative_features(model, vect, sentiment, n=20):
    feature_names = vect.get_feature_names_out()
    if sentiment == 1:
        coef = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        coef = model.feature_log_prob_[0] - model.feature_log_prob_[1]
    coefs_with_fns = sorted(zip(coef, feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n+1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))

print('Most informative features for positive sentiment: ')
show_most_informative_features(clf, cv, 1)

print('\nMost informative features for negative sentiment: ')
show_most_informative_features(clf, cv, 0)

