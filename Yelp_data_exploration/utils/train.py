# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:48:50 2020

@author: hexiaoni
"""

from sklearn.metrics.pairwise import cosine_similarity

def train(matrix_train):
    similarity = cosine_similarity(X=matrix_train, Y=None, dense_output=True)
    return similarity