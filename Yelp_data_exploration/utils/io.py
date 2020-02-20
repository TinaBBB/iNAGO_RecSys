# -*- coding: utf-8 -*-
"""
io

"""

import datetime
import json
import pandas as pd
import time
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, save_npz, load_npz
import numpy as np

def save_dataframe_csv(df, path, name):
    csv_filename = "{:s}.csv".format(name)
    df.to_csv(path+csv_filename, index=False)
    print('Dataframe Saved Successfully')

def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)

def save_numpy_csr(matrix, path, model):
    save_npz('{}{}'.format(path, model), matrix)

def load_numpy_csr(path, name):
    return load_npz(path+name).tocsr()

def save_numpy(matrix, path, model):
    np.save('{}{}'.format(path, model), matrix)

def load_numpy(path, name):
    return np.load(path+name)

def saveDictToJson(dictionary, path, fileName, trainOrTest='train'):
    json_fileName = "{:s}.json".format(fileName)
    if(trainOrTest == 'train'):
        json.dump(dictionary, open(path+json_fileName, 'w') )
    else:
        json.dump(dictionary, open(path+json_fileName, 'w') )

def loadDict(path, fileName, trainOrTest='train'):
    # json_fileName = "{:s}.json".format(fileName)
    # Read data from file:
    if(trainOrTest == 'train'):
        dataDict = json.load(open(path+fileName))
    else:
        dataDict = json.load(open(path+fileName))
    return dataDict

def get_yelp_df(path = 'data/', filename = 'Export_CleanedReview.json', sampling=False, top_user_num=7000, top_item_num=5000):
    """
    Get the pandas dataframe
    Sampling only the top users/items by density 
    Implicit representation applies
    """
    with open(filename,'r') as f:
        data = f.readlines()
        data = list(map(json.loads, data))
    
    data = data[0]
    #Get all the data from the dggeata file
    df = pd.DataFrame(data)
    
    df.rename(columns={'stars': 'review_stars', 'text': 'review_text', 'cool': 'review_cool',
                       'funny': 'review_funny', 'useful': 'review_useful'},
              inplace=True)

    df['business_num_id'] = df.business_id.astype('category').\
        cat.rename_categories(range(0, df.business_id.nunique()))
    df['business_num_id'] = df['business_num_id'].astype('int')

    df['user_num_id'] = df.user_id.astype('category').\
    cat.rename_categories(range(0, df.user_id.nunique()))
    df['user_num_id'] = df['user_num_id'].astype('int')

    df['timestamp'] = df['date'].apply(date_to_timestamp)

    if sampling:
        df = filter_yelp_df(df, top_user_num=top_user_num, top_item_num=top_item_num)
        # Refresh num id
        df['business_num_id'] = df.business_id.astype('category').\
        cat.rename_categories(range(0, df.business_id.nunique()))
        df['business_num_id'] = df['business_num_id'].astype('int')
        
        df['user_num_id'] = df.user_id.astype('category').\
        cat.rename_categories(range(0, df.user_id.nunique()))
        df['user_num_id'] = df['user_num_id'].astype('int')
#     drop_list = ['date','review_id','review_funny','review_cool','review_useful']
#     df = df.drop(drop_list, axis=1)

    df = df.reset_index(drop = True)

    return df 

def filter_yelp_df(df, top_user_num=7000, top_item_num=5000):
    #Getting the reviews where starts are above 3
    df_implicit = df[df['review_stars']>3]
    frequent_user_id = df_implicit['user_num_id'].value_counts().head(top_user_num).index.values
    frequent_item_id = df_implicit['business_num_id'].value_counts().head(top_item_num).index.values
    return df.loc[(df['user_num_id'].isin(frequent_user_id)) & (df['business_num_id'].isin(frequent_item_id))]

def date_to_timestamp(date):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return time.mktime(dt.timetuple())

def df_to_sparse(df, row_name='userId', col_name='movieId', value_name='rating',
                 shape=None):
    rows = df[row_name]
    cols = df[col_name]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    return csr_matrix((values, (rows, cols)), shape=shape)

"""

Matrix Generation

"""

def get_rating_timestamp_matrix(df, sampling=False, top_user_num=7000, top_item_num=5000):
    """
    """
    #make the df implicit with top frenquent users and 
    #no need to sample anymore if df was sampled before 
    if sampling:
        df = filter_yelp_df(df, top_user_num=top_user_num, top_item_num=top_item_num)

    rating_matrix = df_to_sparse(df, row_name='user_num_id',
                                 col_name='business_num_id',
                                 value_name='review_stars',
                                 shape=None)
    
    #Have same dimension and data entries with rating_matrix, except that the review stars are - user avg
#     ratingWuserAvg_matrix = df_to_sparse(df, row_name='user_num_id',
#                                  col_name='business_num_id',
#                                  value_name='reviewStars_userAvg',
#                                  shape=None)
    
    timestamp_matrix = df_to_sparse(df, row_name='user_num_id',
                                    col_name='business_num_id',
                                    value_name='timestamp',
                                    shape=None)
    
    
    IC_matrix, IC_dictionary = get_I_C(df)
#     ratingWuserAvg_matrix
    return rating_matrix, timestamp_matrix, IC_matrix, IC_dictionary

def get_I_C(df):
    lst = df.categories.values.tolist()
    cat = []
    for i in range(len(lst)):
        if lst[i] is None:
            print(i)
        cat.extend(lst[i].split(', '))
        
    unique_cat = set(cat)
    #     set categories id
    df_cat = pd.DataFrame(list(unique_cat),columns=["Categories"])
    df_cat['cat_id'] = df_cat.Categories.astype('category').cat.rename_categories(range(0, df_cat.Categories.nunique()))
    dict_cat = df_cat.set_index('Categories')['cat_id'].to_dict()
    
    df_I_C = pd.DataFrame(columns=['business_num_id', 'cat_id'])
    
    for i in range((df['business_num_id'].unique().shape)[0]):
        df_temp = df[df['business_num_id'] == i].iloc[:1]
        temp_lst = df_temp['categories'].to_list()[0].split(",")
        for j in range(len(temp_lst)):
            df_I_C = df_I_C.append({'business_num_id' : i  , 'cat_id' : dict_cat[temp_lst[j].strip()]} , ignore_index=True)
    
    IC_Matrix = df_to_sparse(df_I_C, row_name='business_num_id',
                                 col_name='cat_id',
                                 value_name=None,
                                 shape=None)    
    return IC_Matrix, dict_cat

def getImplicitMatrix(sparseMatrix, threashold=0):
    temp_matrix = sparse.csr_matrix(sparseMatrix.shape)
    temp_matrix[(sparseMatrix > threashold).nonzero()] = 1
    return temp_matrix

def get_UC_Matrix(IC_Matrix,rtrain_implicit):
    U_C_matrix_explicit = rtrain_implicit*IC_Matrix
    U_C_matrix_implicit = getImplicitMatrix(U_C_matrix_explicit,3)
    return U_C_matrix_explicit,U_C_matrix_implicit

