# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:19:55 2020

@author: hexiaoni
"""

from tqdm import tqdm

import numpy as np

from utils.evaluate import evaluate

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

#Preidction score is UI or IU?
def prediction(prediction_score, matrix_Train, topK=50):

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

#One hot encoding for new users, passed in 1 * number items prediction score vector
def prediction_oneHotEncode(prediction_score, initial_record):
    prediction = []

    #for each user
    #for user_index in tqdm(range(matrix_Train.shape[0])):
        
    #take the prediction scores for user 1 * num res
    vector_u =prediction_score

    #The restuarant the user rated 1 * num res
    vector_train = initial_record

    if len(vector_train.nonzero()[0]) > 0:
        vector_predict = sub_routine_modified(vector_u, vector_train)
    else:
        vector_predict = np.zeros(50, dtype=np.float32)

    prediction.append(vector_predict)

    return prediction

#topK: the number of restuarants we are suggesting 
#if vector_train has number, then the user has visited
def sub_routine(vector_u, vector_train, topK=50):

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


def prediction_modified(prediction_score, matrix_Train, user_id, topK = 50):
    prediction = []

    #for each user
    for user_index in tqdm(range(matrix_Train.shape[0])):
        
        #take the prediction scores for user 1 * num res
        vector_u = prediction_score[user_index]
        
        #The restuarant the user rated
        vector_train = matrix_Train[user_index]
        
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine_modified(vector_u, vector_train)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return prediction[user_id]


def sub_routine_modified(vector_u, vector_train):

    #index where non-zero
    train_index = vector_train.nonzero()[1]
    
    vector_u = vector_u
    
    #get topk + num rated res prediction score descending, top index 
    candidate_index = np.argpartition(-vector_u, -1)
    
    #sort top prediction score index in range topK+len(train_index) into vector_u`
    vector_u = candidate_index[vector_u[candidate_index].argsort()[::-1]]
    
    #deleted the rated res from the topk+train_index prediction score vector for user u 
    #Delete the user rated res index from the topk+numRated index
    vector_u = np.delete(vector_u, np.isin(vector_u, train_index).nonzero()[0])

    #so we only include the top K prediction score here
    return vector_u


#Passing in the trained similarity matrx, for the purpose of cross validation
def individualKNNPrediction (similarityMatrix, predictionMatrix, kRange, validOrTestMatrix, itemBased=False):
    "Declaration for kRange = range(50,120,10)"
    #similarity = train(similarityMatrix)
    MAP10 = {}
    #Loop through the kvalues 
    for kValue in kRange:
        if(itemBased==False):
            user_item_prediction_score = predict(predictionMatrix, kValue, similarityMatrix, item_similarity_en= False)
        else:
            user_item_prediction_score = predict(predictionMatrix, kValue, similarityMatrix, item_similarity_en= True)
        user_item_predict = prediction(user_item_prediction_score, 50, predictionMatrix)
        user_item_res = evaluate(user_item_predict, validOrTestMatrix)
        
        MAP10[kValue] = user_item_res.get('MAP@10')
        
    return MAP10

#Passing in the trained similarity matrx
def KNNPrediction (similarityMatrix, predictionMatrix, kValue, validOrTestMatrix, itemBased=False):

    if(itemBased==False):
        user_item_prediction_score = predict(predictionMatrix, kValue, similarityMatrix, item_similarity_en= False)
    else:
        user_item_prediction_score = predict(predictionMatrix, kValue, similarityMatrix, item_similarity_en= True)
    user_item_predict = prediction(user_item_prediction_score, 50, predictionMatrix)
    user_item_res = evaluate(user_item_predict, validOrTestMatrix)

        
    return user_item_res.get('MAP@10')

