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
import sys


def load_data(database_filepath):
    '''
    This function is to import data from .db files.
    inout:
        database_filepath: path to the .db file
    Output:
        X: Extracted varibale columns (messages, genre) 
        Y: Extracted category columns
        category_names: column name of the categories
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse_table', engine)
    X = df['message']
    y = df[df.columns[4:]]
    return X, y, y.columns

def tokenize(text):
    '''
    This function is for tokenizing a given text 
    (eg. each message from X in the model). Steps include
    tokenizing sentences into words, lemmatizing (words root), 
    normalizing (all to lower case), removing stop words and empty spaces
    
    Input:
        text: a given text (a message at a time)
    output:
        clean_tokens: cleaned tokens to be used in the model        
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    This function creates a model. We make the model
    first by defining a pipeline that encloses all the
    transforming steps to make the features ready for
    the estimator/predictor. 
    
    The transformation is done in 2 parallel steps:
    1) to the text column (messages) to turn them into 
    vectors for the model to use.
    2) to the genre column to turn it into dummy columns
    
    The result of both steps are then concatenated together 
    and fed to the model. As the model should do a 
    multi-classification based on the input features, 
    a AdaBoostClassifier is used inside a MultiOutputClassifier
    to fit the classification per target.
    
    The pipeline and the parameters that we need to tune in the model
    are then given to a GridSearchCV to do the cross validation.
    
    Input: Non
    Output: a tuned and optimized classification model (cv)
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {
        'clf__estimator__learning_rate': [0.5, 1.0],
        'clf__estimator__n_estimators': [10, 20]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance.
    
    Args:
        model (sklearn.pipeline.Pipeline): Trained machine learning model.
        X_test (pd.DataFrame): Test features.
        Y_test (pd.DataFrame): True labels for test data.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print(f'Category: {category}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model.

    Args:
        model: Trained model.
        model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
