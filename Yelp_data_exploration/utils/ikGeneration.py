# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:47:43 2020

@author: hexiaoni
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from scipy.sparse import csr_matrix

#Stemming and Lemmatisation
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
# Get corpus and CountVector
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('wordnet')
nltk.download('stopwords')
lem = WordNetLemmatizer()
stem = PorterStemmer()
stop_words = set(stopwords.words("english"))
new_words = ['not_the']
stop_words = stop_words.union(new_words)

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
        text=re.sub("(\\d|\\W)+"," ",text)
        
        ##Convert to list from string
        text = text.split()
        
        ##Stemming
        # ps=PorterStemmer()
        
        #Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
                stop_words] 
        text = " ".join(text)
        corpus.append(text)
    return corpus

def ikGeneration(df):
    
    corpus = preprocess(df)
    #Creating a dictionary to store business: review text
    dict_text = {}
    for i in range(len(corpus)):
        if df['business_num_id'][i] not in dict_text:
            dict_text[df['business_num_id'][i]] = corpus[i]
        else:
            temp = dict_text[df['business_num_id'][i]]
            temp = temp + corpus[i]
            dict_text[df['business_num_id'][i]] = temp
            
        #Create a list for the review text, where the row dimension = total business ids
    list_text = []
    for key in range(0,max(list(dict_text.keys()))+1) :
        if key not in dict_text.keys():
            list_text.extend([""])
        else:
            list_text.extend([dict_text[key]])
            
        #Get the X vector, where dimension is #business vs #terms like IK
    vectorizer = TfidfVectorizer(max_df=0.9,stop_words=stop_words, max_features=5000, ngram_range=(1,1))
    X_cleaned = vectorizer.fit_transform(list_text).toarray()
    X_cleaned_sparse = csr_matrix(X_cleaned)
    IK_MATRIX = X_cleaned_sparse
    
    return IK_MATRIX