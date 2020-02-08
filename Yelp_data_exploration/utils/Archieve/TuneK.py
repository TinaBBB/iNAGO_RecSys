#!/usr/bin/env python
# coding: utf-8

# ## This file produces algorithm performance for different approaches - Tune hyperparameter K


import numpy as np
import pandas as pd
import sys
import os
#from prediction.models import train, predict, prediction
#from evaluation.metrics import evaluate
from prediction.models import train, predict, prediction
from evaluation.metrics import evaluate


# In[3]:


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


# In[ ]:




