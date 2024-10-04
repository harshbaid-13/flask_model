import nltk
from textblob import TextBlob
from wordcloud import  STOPWORDS
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

from nltk.data import find

def setup_nltk():
    try:
        # Check if stopwords data is already downloaded
        find('corpora/stopwords.zip')
        print("Stopwords already downloaded.")
    except LookupError:
        # If stopwords are not downloaded, download them
        print("Downloading stopwords...")
        nltk.download('stopwords')

setup_nltk()  # Call this function to check and download NLTK data if necessary

stop_words=set(STOPWORDS)
stop_words=stop_words.union(set(stopwords.words('english')))
stop_words.add('im')
stop_words.add('u')
stop_words.add('got')
def clean_text(text):
    text = text.str.lower()    # lowercase 
    text = text.str.strip()  # removing all spaces (useless)
    #text=text.apply()
    #GoogleTranslator(source='auto', target='en').translate() 
    for a in text.index:
        print(round(a/pd.Series(text.index).iloc[-1],3), end='\r')
        #text[a] = lemmatizer.lemmatize(text[a])
        #text[a] = ps.stem(text[a])
        text[a] = re.sub(r"[-()\"#/@;:{}`+=~|_.!?,'0-9]", " ", text[a])        # getting rid of special characters and numbers
        text[a] = ' '.join( [w for w in text[a].split() if len(w)>1] )
        text[a] = re.sub(r' +', ' ', text[a])
        text[a] = re.sub(r'[^a-zA-Z_]', ' ', text[a])
        text[a]=re.sub('not good','bad',text[a])
        text[a]=re.sub('not so good','bad',text[a])
        text[a]=re.sub('not at all good','bad',text[a])
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))  # removing stopwords
    text = text.apply(lambda x: " ".join(x for x in x.split() if wordnet.synsets(x)))   # keeping words with some meaning, can be refined for respective business over time  
    
    return(text)

import pickle
from flask import Flask, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections.abc import Sequence
# Initialize Flask app
app = Flask(__name__)

# Load the trained model from local directory
with open("lda_topic_mining_playstore2.pkl", "rb") as model_file:
    lda_model = pickle.load(model_file)
with open("lda_topic_mining_vocab2.pkl", "rb") as model_file:
    vect = pickle.load(model_file)
countvect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1,2 ), min_df=4, encoding='latin-1',max_features=800,vocabulary=vect)

# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json
        # Assuming input is a list of features
        input_string = data['input_string']
        input_string_cleaned=clean_text(pd.Series(input_string))
        df1_count = countvect.transform(input_string_cleaned)
        dtm_df1=df1_count.toarray()
        dtm_df1=pd.DataFrame(dtm_df1, columns=countvect.get_feature_names_out())
        df1_count = countvect.transform(input_string_cleaned)
        dtm_df1=df1_count.toarray()
        dtm_df1=pd.DataFrame(dtm_df1, columns=countvect.get_feature_names_out())
        data_lda_count=lda_model.transform(df1_count)
        doc_topics1 = pd.DataFrame(data_lda_count)
        topics = 'TOPIC_' + pd.Series(list(range(0,20))).astype(str)
        doc_topics1.columns = topics.tolist()
        dominating_topic = []
        sub_topic = []
        for i in range(0,len(doc_topics1)):
            r = doc_topics1.loc[i,]
            dom_topic = r.sort_values(ascending=False).head(1).index[0] if r.sort_values(ascending=False).head(1)[0]>=0.5 else 'Not able as below 0.5, else it will be '+r.sort_values(ascending=False).head(1).index[0]
            sub_tp = " + ".join(list(r.sort_values(ascending=False)[1:20].index))
            dominating_topic.append(dom_topic)
            sub_topic.append(sub_tp)
        print(dominating_topic)
        print(sub_topic)


        # Return the prediction as a JSON response
        return {'prediction': dominating_topic[0]}
    except Exception as e:
        return {'error': str(e)}

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
