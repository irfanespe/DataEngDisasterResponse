# Disaster Response Pipeline Project

URL for this repository : https://github.com/irfanespe/DataEngineeringWithNLPML-

### Dataset
In this project used 2 data file, categories and message file. Categories file contains id and categories of message.
The other data file, message file, contains id message, translated message, original message and message genre. 
Both data file have same rows, 26248. There are several overview image of this dataset when both data file
are merged. 

![distribution](/dist_mes_gen.png)
Format: ![Alt Text](url)

Each genre has different categories distribution. Below shown top 5 categories in each genre.
![top_dir](/top_dir.png)
Format: ![Alt Text](url)

![top_news](/top_news.png)
Format: ![Alt Text](url)

![top_soc](/top_soc.png)
Format: ![Alt Text](url)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

