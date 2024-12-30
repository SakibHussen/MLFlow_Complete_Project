#importing dependencies
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#initializing the object
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# A bsic text processing function with variations in preprocessing like stemming / lemmatization
def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha()]
    # filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    # filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    # filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_words)

# basic trainin model func with variations for vectorizing
def train_model(x_train, y_train, n, c, d):    
    # Create a Vectorizer to convert text data to numerical features
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)
    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=n, criterion=c, max_depth=d)
    # Train the classifier on the training data
    rf_classifier.fit(x_train_vectorized, y_train)
    pred = rf_classifier.predict(x_train_vectorized)
    acc = accuracy_score(pred, y_train)
    return vectorizer, rf_classifier, acc

# evaluation function
def eval_met(actual, pred):
    acc = accuracy_score(actual, pred)
    prc = precision_score(actual, pred, pos_label='spam')
    rec = recall_score(actual, pred, pos_label='spam')
    f1 = f1_score(actual, pred, pos_label='spam')
    return acc, prc, rec, f1


