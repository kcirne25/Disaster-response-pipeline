# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

# Loading, merging data and creating a dataframe
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how ='outer', on =['id'])
    return df

# Cleaning data by splitting, dropping duplicates, replacing values 2 by 1 on related column
def clean_data(df):
    categories = df['categories'].str.split(';', expand=True)
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    categories['related'] = categories['related'].replace(to_replace=2, value=1)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df

# Saving data using sqlalchemy
def save_data(df, database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponse_table', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
