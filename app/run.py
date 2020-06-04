import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/YourDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # make graph one
    graph_one = []
    graph_one.append(
      Bar(
      x = genre_names,
      y = genre_counts,
      )
    )
    
    layout_one = dict(title = 'Distribution of Message Genres',
                    xaxis = dict(title = 'Genre',),
                    yaxis = dict(title = 'Count'),
                )
    
    # get category data grouped by genre     
    cat_per_genre = df.groupby('genre').sum().drop('id', axis = 1)
    
    # get sorted category data from direct genre    
    cat_in_direct = pd.DataFrame(cat_per_genre.iloc[0,:].sort_values(ascending = False))
    
    # get top 5 indexes in direct genre    
    top_direct_lab = list(cat_in_direct.iloc[:5,:].index.values)
    
    # get big 5 values in direct genre    
    top_direct_val = (cat_in_direct.iloc[:5,0].tolist())
    
    # make graph two
    graph_two = []
    graph_two.append(
      Bar(
       x=top_direct_val,
       y=top_direct_lab,
       orientation='h'
      )
    )
    
    layout_two = dict(title = 'Top 5 Categories in Direct Genre',
                    xaxis = dict(title = 'Count'),
                    yaxis = dict(title = 'Category', tickangle = -45),
                )
    
    # get sorted category data from news genre    
    cat_in_news = pd.DataFrame(cat_per_genre.iloc[1,:].sort_values(ascending = False))
    
    # get top 5 indexes in news genre    
    top_news_lab = list (cat_in_news.iloc[:5,:].index.values)
    
    # get top 5 values in news genre    
    top_news_val = list (cat_in_news.iloc[:5,0].tolist())
    
    # make graph three
    graph_three = []
    graph_three.append(
      Bar(
      x = top_news_val,
      y = top_news_lab,
      orientation = 'h'
      )
    )
    
    layout_three = dict(title = 'Top 5 Categories in News Genre',
                    xaxis = dict(title = 'Count'),
                    yaxis = dict(title = 'Category', tickangle = -45),
                )
    
    # get sorted category data from social genre    
    cat_in_soc = pd.DataFrame(cat_per_genre.iloc[2,:].sort_values(ascending = False))
    
    # get top 5 indexes in social genre    
    top_social_lab = list (cat_in_soc.iloc[:5,:].index.values)
    
    # get top 5 values in social genre    
    top_social_val = list (cat_in_soc.iloc[:5,0].tolist())
    
    # make graph three
    graph_four = []
    graph_four.append(
      Bar(
      x = top_social_val,
      y = top_social_lab,
      orientation = 'h'
      )
    )
    
    layout_four = dict(title = 'Top 5 Categories in Social Genre',
                    xaxis = dict(title = 'Count'),
                    yaxis = dict(title = 'Category', tickangle = -45),
                )
    
    # make graphs and layouts into one dictionary    
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    graphs.append(dict(data=graph_four, layout=layout_four))
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
    
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()