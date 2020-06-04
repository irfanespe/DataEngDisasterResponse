import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to load messages and categories data, and 
    load it into merged dataframe.
    
    input parameter:
    messages_filepath : message file location
    categories_filepath : categories file location
    
    output :
    df : merged message and categories dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')
    
    return df

def clean_data(df):
    """
    This function extract values inc category column into
    separated binary columns, and delete duplication data.
    
    input paramter :
    df : merged raw messages and categories dataframe
    
    output :
    df : df with extracted category values and deleted
         duplication data
    
    """
    
    # create a dataframe of the 36 individual category columns
    categories_col = df.categories.str.split(pat=';', expand = True)
    row = list(categories_col.loc[0,:])

    # use this row to extract a list of new column names for categories.
    # up to the second to last character of each string with slicing
    categories_col.columns = list(map(lambda x : x[:-2], row))
    
    # extract binary values from each values in cell
    for column in categories_col:
        # set each value to be the last character of the string
        categories_col[column] = categories_col[column].str[-1]

        # convert column from string to numeric
        categories_col[column] = categories_col[column].astype('int')
        
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories_col)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('InsertTableName', engine, index=False)  


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