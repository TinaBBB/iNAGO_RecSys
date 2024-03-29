# -*- coding: utf-8 -*-
"""models

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gTuSEuSuUe4byLkTVYVhGHRg8PwM6HwP
"""

from tqdm import tqdm
#Stemming and Lemmatisation
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re 
import nltk
import numpy as np
import statistics as stats
from scipy.sparse import csr_matrix, load_npz, save_npz
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from evaluation.metrics import evaluate
# Get corpus and CountVector
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('wordnet')
nltk.download('stopwords')
lem = WordNetLemmatizer()
stem = PorterStemmer()
stop_words = set(stopwords.words("english"))
new_words = ['not_the']
stop_words = stop_words.union(new_words)

#Should 'because' added?
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
        #ps=PorterStemmer()
        
        #Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
                stop_words] 
        text = " ".join(text)
        corpus.append(text)
    return corpus

def train(matrix_train):
    similarity = cosine_similarity(X=matrix_train, Y=None, dense_output=True)
    return similarity

def get_I_K(df, X, row_name = 'business_num_id', binary = True, shape = (121994,6000)):
    """
    get the item-keyphrase matrix
    """
    rows = []
    cols = []
    vals = []
    #For each review history
    for i in tqdm(range(X.shape[0])):
        #Get the array of frequencies for document/review i 
        arr = X[i].toarray() 
        nonzero_element = arr.nonzero()[1]  # Get nonzero element in each line, keyphrase that appears index 
        length_of_nonzero = len(nonzero_element) #number of important keyphrase that appears
        
        # df[row_name][i] is the item idex
        #Get a list row index that indicates the document/review
        rows.extend(np.array([df[row_name][i]]*length_of_nonzero)) ## Item index
        #print(rows)
        
        #Get a list of column index indicating the key phrase that appears in i document/review
        cols.extend(nonzero_element) ## Keyword Index
        if binary:
            #Create a bunch of 1s
            vals.extend(np.array([1]*length_of_nonzero))
        else:
            #If not binary 
            vals.extend(arr[arr.nonzero()])    
    return csr_matrix((vals, (rows, cols)), shape=shape)


#Get a UI matrix if it's not item_similarity based or else IU
def predict(matrix_train, k, similarity, item_similarity_en = False):
    prediction_scores = []
    
    #inverse to IU matrix
    if item_similarity_en:
        matrix_train = matrix_train.transpose()
        
    #for each user or item, depends UI or IU 
    for user_index in tqdm(range(matrix_train.shape[0])):
        # Get user u's prediction scores for all items
        #Get prediction/similarity score for each user 1*num or user or num of items
        vector_u = similarity[user_index]

        # Get closest K neighbors excluding user u self
        #Decending accoding to similarity score, select top k
        similar_users = vector_u.argsort()[::-1][1:k+1]
        
        # Get neighbors similarity weights and ratings
        similar_users_weights = similarity[user_index][similar_users]
        
        #similar_users_weights_sum = np.sum(similar_users_weights)
        #print(similar_users_weights.shape)
        #shape: num of res * k
        similar_users_ratings = matrix_train[similar_users].toarray()
              
        prediction_scores_u = similar_users_ratings * similar_users_weights[:, np.newaxis]
        #print(prediction_scores_u)
        
        
        prediction_scores.append(np.sum(prediction_scores_u, axis=0))
        
    res = np.array(prediction_scores)
    
    if item_similarity_en:
        res = res.transpose()
    return res

#Get a UI matrix if it's not item_similarity based or else IU
def predictUU(matrix_train, k, chooseWeigthMethod, similarity1=None, similarity2=None, similarity3=None, similarity4=None, similarity5=None, item_similarity_en = False):
    prediction_scores = []
    #Convert from list to ndarray, add an axis
    if isinstance(chooseWeigthMethod, list):
        chooseWeigthMethod = np.array(chooseWeigthMethod)[:, np.newaxis]
   
    "make sure that when passing in chooseWeightMethod, the weight must be aligned with similarity metrices, even if set to None"
    "They should add to 1 as well"
    #inverse to IU matrix
    if item_similarity_en:
        matrix_train = matrix_train.transpose()
        
    #for each user or item, depends UI or IU 
    for user_index in tqdm(range(matrix_train.shape[0])):
    #for user_index in tqdm(range(10,20)):
        
        numberSimilarMatrix = 0
        # Get user u's prediction scores for all items 
        #Get prediction/similarity score for each user 1*num or user or num of items
        if similarity1 is not None:
            vector_u1 = similarity1[user_index]
            numberSimilarMatrix += 1
        else:
            vector_u1 = [0]*matrix_train.shape[0]
            
        if similarity2 is not None:
            vector_u2 = similarity2[user_index]
            numberSimilarMatrix += 1
        else:
            vector_u2 = [0]*len(vector_u1)
            
        if similarity3 is not None:
            vector_u3 = similarity3[user_index]
            numberSimilarMatrix += 1
        else:
            vector_u3 = [0]*len(vector_u1)
            
        if similarity4 is not None:
            vector_u4 = similarity4[user_index]
            numberSimilarMatrix += 1
        else:
            vector_u4 = [0]*len(vector_u1)
        
        if similarity5 is not None:
            vector_u5 = similarity5[user_index]
            numberSimilarMatrix += 1
        else:
            vector_u5 = [0]*len(vector_u1)
        
        #Temperary vector that stacks all 4 vectors together
        tempVector = np.array([vector_u1,vector_u2,vector_u3,vector_u4, vector_u5])
        
        if chooseWeigthMethod is None:
            #Get the similarity score from the first similarity matrix anyways 
            vector_u = vector_u1.copy()
            
        #If we are choosing the max, min, avg or similarity scores
        if chooseWeigthMethod is not None:
            if chooseWeigthMethod == 'max':
                vector_u = tempVector.max(axis=0)
            elif chooseWeigthMethod == 'min':
                vector_u = tempVector.min(axis=0)
            elif chooseWeigthMethod == 'average':
                vector_u = tempVector.mean(axis=0)
            elif isinstance(chooseWeigthMethod, np.ndarray):
                #Validate that number of weights passed in equals number of matrices
                #assert(len(chooseWeigthMethod) == numberSimilarMatrix)
                #Get the new combined similarity vector 
                weighted_u = tempVector * chooseWeigthMethod
                vector_u =np.sum(weighted_u, axis=0)
                #print((vector_u == vector_u4).sum())
                
        similar_users = vector_u.argsort()[::-1][1:k+1]
        
        # Get neighbors similarity weights and ratings
        #similar_users_weights = similarity1[user_index][similar_users]
        similar_users_weights = vector_u[similar_users]
        
        #similar_users_weights_sum = np.sum(similar_users_weights)
        #print(similar_users_weights.shape)
        #shape: num of res * k
        similar_users_ratings = matrix_train[similar_users].toarray()
              
        prediction_scores_u = similar_users_ratings * similar_users_weights[:, np.newaxis]
        #print(prediction_scores_u)
        
        prediction_scores.append(np.sum(prediction_scores_u, axis=0))
        
    res = np.array(prediction_scores)
    
    if item_similarity_en:
        res = res.transpose()
    return res
    #return vector_u


def predictII(matrix_train, k, similarity, item_similarity_en = False):
    prediction_scores = []
    
    #inverse to IU matrix
    if item_similarity_en:
        matrix_train = matrix_train.transpose()
        
    #for each user or item, depends UI or IU 
    for user_index in tqdm(range(matrix_train.shape[0])):
        # Get user u's prediction scores for all items
        #Get prediction/similarity score for each user 1*num or user or num of items
        vector_u = similarity[user_index]

        # Get closest K neighbors excluding user u self
        #Decending accoding to similarity score, select top k
        similar_users = vector_u.argsort()[::-1][1:k+1]
        
        # Get neighbors similarity weights and ratings
        similar_users_weights = similarity[user_index][similar_users]
        
        #similar_users_weights_sum = np.sum(similar_users_weights)
        #print(similar_users_weights.shape)
        #shape: num of res * k
        similar_users_ratings = matrix_train[similar_users].toarray()
              
        prediction_scores_u = similar_users_ratings * similar_users_weights[:, np.newaxis]
        #print(prediction_scores_u)
        
        
        prediction_scores.append(np.sum(prediction_scores_u, axis=0))
        
    res = np.array(prediction_scores)
    
    if item_similarity_en:
        res = res.transpose()
    return res



#Preidction score is UI or IU?
def prediction(prediction_score, topK, matrix_Train):

    prediction = []

    #for each user
    for user_index in tqdm(range(matrix_Train.shape[0])):
        
        #take the prediction scores for user 1 * num res
        vector_u = prediction_score[user_index]
        
        #The restuarant the user rated
        vector_train = matrix_Train[user_index]
        
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(vector_u, vector_train, topK=topK)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return np.vstack(prediction)

#topK: the number of restuarants we are suggesting 
#if vector_train has number, then the user has visited
def sub_routine(vector_u, vector_train, topK=500):

    #index where non-zero
    train_index = vector_train.nonzero()[1]
    
    vector_u = vector_u
    
    #get topk + num rated res prediction score descending, top index 
    candidate_index = np.argpartition(-vector_u, topK+len(train_index))[:topK+len(train_index)]
    
    #sort top prediction score index in range topK+len(train_index) into vector_u`
    vector_u = candidate_index[vector_u[candidate_index].argsort()[::-1]]
    
    #deleted the rated res from the topk+train_index prediction score vector for user u 
    #Delete the user rated res index from the topk+numRated index
    vector_u = np.delete(vector_u, np.isin(vector_u, train_index).nonzero()[0])

    #so we only include the top K prediction score here
    return vector_u[:topK]

def individualKNNPrediction (similarityMatrix, predictionMatrix, kRange, validOrTestMatrix):
    "Declaration for kRange = range(50,120,10)"
    similarity = train(similarityMatrix)
    MAP10 = {}
    for kValue in kRange:
        user_item_prediction_score = predict(predictionMatrix, kValue, similarity, item_similarity_en= False)
        user_item_predict = prediction(user_item_prediction_score, 50, predictionMatrix)
        user_item_res = evaluate(user_item_predict, validOrTestMatrix)
        
        
        MAP10[kValue] = user_item_res.get('MAP@10')[0]
        
    return MAP10

def simpleKNNPrediction (similarityMatrix, predictionMatrix, kValue, validOrTestMatrix):
    "Declaration for kRange = range(50,120,10)"
    similarity = train(similarityMatrix)
    user_item_prediction_score = predict(predictionMatrix, kValue, similarity, item_similarity_en= False)
    user_item_predict = prediction(user_item_prediction_score, 50, predictionMatrix)
    user_item_res = evaluate(user_item_predict, validOrTestMatrix)

    MAP10= user_item_res.get('MAP@10')[0]
        
    return MAP10

def computeUUCombination(rtrain, rtrain_userAvg, userVisitMatrix, rtrain_implicit, combinationDict, SimilarityMatrixIndex, kTune, method='max'):

    prediction1 = {}
    prediction2 = {}
    prediction3 = {}
    prediction4 = {}

    for combination, indexList in combinationDict.items():
        #Loop through the similarity matrices 
        for index in SimilarityMatrixIndex.keys():
            if index in indexList:
                if index == 1: 
                    similarityOne = SimilarityMatrixIndex[1][0]
                elif index == 2:
                    similarityTwo = SimilarityMatrixIndex[2][0]
                elif index == 3:
                    similarityThree = SimilarityMatrixIndex[3][0]
                elif index == 4:
                    similarityFour = SimilarityMatrixIndex[4][0]
            else:
                if index == 1: 
                    similarityOne = SimilarityMatrixIndex[1][1]
                elif index == 2:
                    similarityTwo = SimilarityMatrixIndex[2][1]
                elif index == 3:
                    similarityThree = SimilarityMatrixIndex[3][1]
                elif index == 4:
                    similarityFour = SimilarityMatrixIndex[4][1]

        user_item_prediction_score1 = predictUU(rtrain, kTune, similarityOne, similarityTwo, similarityThree, similarityFour, chooseWeigthMethod=method, item_similarity_en= False)
        user_item_predict1 = prediction(user_item_prediction_score1, 50, rtrain)
        user_item_res1 = evaluate(user_item_predict1, rvalid)
        prediction1[combination] = user_item_res1.get('MAP@10')[0]

        user_item_prediction_score2 = predictUU(rtrain_userAvg, kTune, similarityOne, similarityTwo, similarityThree, similarityFour, chooseWeigthMethod=method, item_similarity_en= False)
        user_item_predict2 = prediction(user_item_prediction_score2, 50, rtrain_userAvg)
        user_item_res2 = evaluate(user_item_predict2, rvalid_userAvg)
        prediction2[combination] = user_item_res2.get('MAP@10')[0]

        user_item_prediction_score3 = predictUU(userVisitMatrix, kTune, similarityOne, similarityTwo, similarityThree, similarityFour, chooseWeigthMethod=method, item_similarity_en= False)
        user_item_predict3 = prediction(user_item_prediction_score3, 50, userVisitMatrix)
        user_item_res3 = evaluate(user_item_predict3, rvalid_implicit)         
        prediction3[combination] = user_item_res3.get('MAP@10')[0]                

        user_item_prediction_score4 = predictUU(rtrain_implicit, kTune, similarityOne, similarityTwo, similarityThree, similarityFour, chooseWeigthMethod=method, item_similarity_en= False)
        user_item_predict4 = prediction(user_item_prediction_score4, 50, rtrain_implicit)
        user_item_res4 = evaluate(user_item_predict4, rvalid_implicit)
        prediction4[combination] = user_item_res4.get('MAP@10')[0]
        
    plotingCombination(prediction1, prediction2, prediction3, prediction4, kTune, method)