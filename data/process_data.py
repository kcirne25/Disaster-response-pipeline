# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets.

    Args:
        messages_filepath (str): Filepath of messages dataset.
        categories_filepath (str): Filepath of categories dataset.

    Returns:
        tuple: A tuple containing two dataframes (messages, categories).
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories

def merge_data(messages, categories):
    """
    Merge messages and categories datasets.

    Args:
        messages (pd.DataFrame): Messages dataframe.
        categories (pd.DataFrame): Categories dataframe.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged dataframe.

    Args:
        df (pd.DataFrame): Merged dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Use the first row of the categories dataframe to create column names for the categories data
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename columns of categories with new column names
    categories.columns = category_colnames

    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Drop the original categories column from df
    df = df.drop('categories', axis=1)

    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataset into an sqlite database.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        database_filename (str): Name of the database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse_table', engine, index=False)

# Example usage
messages_filepath = 'messages.csv'
categories_filepath = 'categories.csv'
database_filename = 'DisasterResponse.db'

messages, categories = load_data(messages_filepath, categories_filepath)
df = merge_data(messages, categories)
cleaned_df = clean_data(df)
save_data(cleaned_df, database_filename)
