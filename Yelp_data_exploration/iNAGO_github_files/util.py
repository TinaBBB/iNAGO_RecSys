import pickle
import json
import math
from scipy.sparse import csc_matrix, save_npz, load_npz
import numpy as np
import pandas as pd
import geopy.distance
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm import tqdm
from scipy.sparse import csr_matrix
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer
stop_words = set(stopwords.words("english"))
new_words = ['not_the']
stop_words = stop_words.union(new_words)

def calculate_distance(coord1, coord2):
    return geopy.distance.distance(coord1, coord2).km

def calculate_add_distance(df):
    return df

def save_npz_data(path, obj):
    save_npz(path, obj)

def load_npz_data(path, name):
    return load_npz(path + name)

def load_POI_lat_long_df(path, name):
    return pd.read_pickle(path + name)

def save_POI_lat_long_df(path, obj):
    pickle_df(obj, path)

def get_POI_df(path):
    with open(path, encoding = 'utf-8') as json_file:
        data = json_file.readlines()
        data = list(map(json.loads, data))

        df = pd.DataFrame(data)
        df.rename(columns={'stars': 'business_stars', 'name': 'business_name',
                           'review_count': 'business_review_count'}, inplace=True)
    return df

#Get toronto business df
def get_POI_df_toronto(path):
    with open(path) as json_file:
        data = json_file.readlines()
        data = list(map(json.loads, data))
    data = data[0]
    
    df = pd.DataFrame(data)
    
    #renaming the columns 
    df.rename(columns={'rating': 'business_stars', 'name': 'business_name',
                       'review_count_y': 'business_review_count'}, inplace=True)
    return df

def filter_by_city(df, city="Toronto"):
    df = df.loc[df['city'] == city]
    return df

def select_restaurants(df):
    df = df.dropna(subset=['attributes'])
    df.reset_index(drop=True, inplace=True)

    restaurant_indexes = df['attributes'].apply(pd.Series).dropna(subset=['GoodForMeal']).index.values.tolist()
    df = df.iloc[restaurant_indexes, :]
    df.reset_index(drop=True, inplace=True)

    df['business_num_id'] = df.business_id.astype('category').\
        cat.rename_categories(range(0, df.business_id.nunique()))
    df['business_num_id'] = df['business_num_id'].astype('int')
    return df

def return_POI_lat_long_df(df):
    df = df[["business_id", "business_num_id", "latitude", "longitude"]].sort_values('business_num_id', ascending=True)
    
    df['latitude'] = df['latitude'].astype('float')
    df['longitude'] = df['longitude'].astype('float')
    return df

def pickle_df(df, filename):
    df.to_pickle(filename)

def get_review_df(path):
    with open(path, encoding = 'utf-8') as json_file:
        data = json_file.readlines()        
        data = list(map(json.loads, data))

    review_df = pd.DataFrame(data)
    review_df.rename(columns={'stars': 'review_stars', 'cool': 'review_cool', 'funny': 'review_funny',
                              'useful': 'review_useful'}, inplace=True)
    #Only returning the following columns, with addition of review text
    return review_df[['business_id', 'date', 'review_stars', 'user_id']]
    #return review_df

#Reading Toronto review csv data     
def get_review_df_toronto(path):
    with open(path, encoding = 'utf-8') as json_file:
        data = json_file.readlines()        
        data = list(map(json.loads, data))
        data = data[0]
    
    review_df = pd.DataFrame(data)
    review_df.rename(columns={'stars': 'review_stars', 'text': 'review_text'}, inplace=True)
    
    #Only returning the following columns, with addition of review text
    return review_df[['business_id', 'date', 'review_stars', 'user_id', 'review_text']]
#    return review_df
    

def binarized_star(df):
    df['review_stars_binary'] = (df['review_stars'] > 3).astype(int)
    return df

def merge_df(df1, df2, on_column, how_merge, columns, sort_column):
    return pd.merge(df1, df2, on=on_column, how=how_merge)[columns].sort_values(by=sort_column)

def pandas_to_dict(df, key_col, value_col):
    return pd.Series(df[value_col].values,index=df[key_col]).to_dict()

""" Currently the train set and test set are splited by 80/20.
    In practice, they can be splited in multiple ways.
    For example, we can use last 6 months data as test set and any data before
    that will be train set. In this case, we can remove all restaurants that
    users have been to in test set from prediction so that users don't get
    recommendations for POIs they have recently been to.
"""

#Now spliting for both binary and raw review star
def train_test_split(df):
    num_reviews = len(df)
    train = df[:math.ceil(0.8*num_reviews)][["business_num_id", "user_num_id", "review_text", "review_stars", "review_stars_binary"]]
    test = df[math.ceil(0.8*num_reviews):][["business_num_id", "user_num_id","review_text", "review_stars", "review_stars_binary"]]
    return train, test

#Computing the UI matrix accoridng to the given df, default binary setting is true 
def df_to_sparse(df, num_rows, num_cols, binarySetting=True):
    
    # Assume the dataset is in order of 'business_num_id', 'user_num_id', 'review_stars'
    row = np.asarray(df["user_num_id"])
    col = np.asarray(df["business_num_id"])

    if (binarySetting == True):
        data = np.asarray(df["review_stars_binary"]).astype(np.int)
    else:
        data = np.asarray(df["review_stars"]).astype(np.int)
    return csc_matrix((data, (row, col)), shape=(num_rows, num_cols))

def save_user_id_dict_pickle(path, obj, name):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_user_id_dict_pickle(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#Produce an IK matrix for passed in dataframe, keywords = max_features_num set to 5000
def get_I_K(df_train, max_features_num = 5000):
    corpus = preprocess(df_train)
    dict_text = {}
    for i in range(len(corpus)):
        if df_train['business_num_id'][i] not in dict_text:
            dict_text[df_train['business_num_id'][i]] = corpus[i]
        else:
            temp = dict_text[df_train['business_num_id'][i]]
            temp = temp + corpus[i]
            dict_text[df_train['business_num_id'][i]] = temp
    list_text = []
    for key in range(0,max(list(dict_text.keys()))+1) :
        if key not in dict_text.keys():
            list_text.extend([""])
        else:
            list_text.extend([dict_text[key]])
    vectorizer = TfidfVectorizer(max_df=0.9,stop_words = stop_words, max_features=max_features_num, ngram_range=(1,1))
    X_cleaned = vectorizer.fit_transform(list_text).toarray()
    return csr_matrix(X_cleaned)

#preprocess funtion processes the review text for each business into item * 1 list 
def preprocess(df, reset_list = [',','.','?',';','however','but']):
    corpus = []
    for i in tqdm(range(df.shape[0])):
        text = df['review_text'][i]
        change_flg = 0
        #Convert to lowercase
        text = text.lower()
        
        ##Convert to list from string, loop through the review text
        text = text.split()
        
        #any sentence that encounters a not, the folloing words will become not phrase until hit the sentence end
        for j in range(len(text)):
            #Make the not_ hack
            if text[j] == 'not':
                change_flg = 1
#                 print 'changes is made after ', i
                continue
            #if was 1 was round and not hit a 'not' in this round
            if change_flg == 1 and any(reset in text[j] for reset in reset_list):
                text[j] = 'not_' + text[j]
                change_flg = 0
#                 print 'reset at ', i
            if change_flg == 1:
                text[j] = 'not_' + text[j]
        
        #Convert back to string
        text = " ".join(text)
        
        #Remove punctuations
#       text = re.sub('[^a-zA-Z]', ' ', text)
        
        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
        
        # remove special characters and digits
        text = re.sub("(\\d|\\W)+"," ",text)
        
        ##Convert to list from string
        text = text.split()
        
        ##Stemming
        #ps = PorterStemmer()
        
        #Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
                stop_words] 
        text = " ".join(text)
        corpus.append(text)
    return corpus

