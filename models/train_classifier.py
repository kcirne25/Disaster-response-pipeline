# Importing libraries
import nltk
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Function to load data from database
def load_data(database_filepath):
    """
    Load data from database.

    Args:
        database_filepath (str): Filepath of the database.

    Returns:
        tuple: A tuple containing features (X), labels (Y), and category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse_table', engine)
    
    # Removing 'child_alone' column as it is not meaningful
    df = df.drop(['child_alone'], axis=1)
    
    # Replacing value '2' with '1' on related column
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x)
    
    X = df['message']
    Y = df.iloc[:, 4:]
    
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

# Tokenization function
def tokenize(text):
    """
    Tokenize text.

    Args:
        text (str): Input text.

    Returns:
        list: List of clean tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Custom transformer to extract starting verb
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# Function to build a machine learning pipeline
def build_pipeline(clf=AdaBoostClassifier()):
    """
    Build a machine learning pipeline.

    Args:
        clf: Classifier to be used in the pipeline.

    Returns:
        Pipeline: Machine learning pipeline.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])

    return pipeline

# Function to train the pipeline
def train_pipeline(pipeline, X_train, y_train):
    """
    Train the machine learning pipeline.

    Args:
        pipeline (Pipeline): Machine learning pipeline.
        X_train (pd.Series): Training features.
        y_train (pd.DataFrame): Training labels.

    Returns:
        GridSearchCV: Trained model.
    """
    parameters = {
        'clf__estimator__learning_rate': [0.5, 1.0],
        'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3)
    cv.fit(X_train, y_train)

    return cv

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the machine learning model.

    Args:
        model: Trained model.
        X_test (pd.Series): Testing features.
        y_test (pd.DataFrame): Testing labels.
        category_names (list): List of category names.
    """
    y_pred_test = model.predict(X_test)

    # classification report
    print(classification_report(y_test.values, y_pred_test, target_names=category_names))

# Function to save the model
def save_model(model, model_filepath):
    """
    Save the trained model.

    Args:
        model: Trained model.
        model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
