import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
from statistics import mean
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
# from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib 
import pickle

def load_data(database_filepath):
    """
    This function is performed to load data from sql type data file into
    pandas dataframe. Output from this function is separated into
    features (X), labels (Y) and genre (category names).
    
    input parameter :
    database_filepath : sql file address
    
    output parameter:
    X : Features matrix
    Y : label of dataset, consist of 36 different classes
    category_name : genre of message
    
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('InsertTableName',engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_name = df.iloc[:,3]
    return(X, Y, category_name)

def tokenize(text):
    """
    This function convert raw text into separated word token.
    
    input parameter : 
    text : string text that user want to convert into token
    
    output parameter:
    tokens : list of tokens from text.
    """
    raw_tokens = word_tokenize(text.lower()) # change text into separate words
    lemmatizer = WordNetLemmatizer() # make lemmatizer object
    
    tokens = []
    
    # make lemma from separated words
    for token in raw_tokens:
        tokens.append(lemmatizer.lemmatize(token).strip())
    
    return tokens


def build_model():
    """
    This procedure is made to make pipeline with natural language process and process
    genre column. This model is based on logistic regression.
    
    output parameter:
    model_pipeline : made pipeline model
    """
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(OneVsRestClassifier(LogisticRegression())))
           ])
    
    parameters = { 
                'clf__estimator__estimator__C' : [0.1, 0.3, 1.0]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv

def make_prediction_report(y_true, y_prediction):
    """
    This function is used to make prediction report of inputted parameters,
    output from this function is printed report in monitor and list of
    precision, recall and f1 score.
    
    Input parameters:
    y_true : true of output value from dataset
    y_predicted : predicted output value from trained model calculation
    
    
    Output paramters:
    precision_list = list contain precision values for each output class
    recall_list = list contain recall values for each output class
    f1_list = list contain f1-score values for each output class
    """
    
    precision_list =[]
    recall_list =[]
    f1_list =[]
    # print classification report each column
    for col in range (len(list (y_true.columns))):
        print(y_true.columns[col])
        report = classification_report(y_true.iloc[:,col], y_prediction[:,col]) 
        print(report)
        splited_rep = report.split()     # split report into list
        precision_list.append(float(splited_rep[-4]))     # take precision from report
        recall_list.append(float(splited_rep[-3]))     # take precision from report
        f1_list.append(float(splited_rep[-2]))     # take precision from report
        print()
    
    # return precision, recall and f1 list
    return precision_list, recall_list, f1_list

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function is used to eavluate trained model.
    
    input parameter:
    model : model pipeline
    X_test : dataset features for testing
    Y_test : label for testing
    category_name : genre category
    
    """
    
    # make prediciton of test set
    y_pred_test = model.predict(X_test)

    test_prec_list, test_rec_list, test_f1_list =  make_prediction_report(Y_test, y_pred_test)
    
    categories = list(Y_test.columns)
    
    # make dataframe for each category report value
    cat_name = pd.DataFrame(categories, columns = ['category_name'])
    pre_cat = pd.DataFrame(test_prec_list, columns = ['precision'])
    rec_cat = pd.DataFrame(test_rec_list, columns = ['recall'])
    f1_cat = pd.DataFrame(test_f1_list, columns = ['f1_score'])
    
    cat_name = cat_name.join([pre_cat])
    cat_name = cat_name.join([rec_cat])
    cat_name = cat_name.join([f1_cat])
    
    print('performance recap for all categories')
    print(cat_name)
    
    print()
    # print overall / average precision, recall and f1 score of all categories
    print('average value for all parameters')
    print('average precision : {}'.format(mean(test_prec_list)))
    print('average recall : {}'.format(mean(test_rec_list)))
    print('average f1-score : {}'.format(mean(test_f1_list)))

    print('best parameter :')
    print(model.best_params_)


def save_model(model, model_filepath):
    #save model with pickle
    pickle.dump(model.best_estimator_, open('classifier.pkl', 'wb'))


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