# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:01:01 2020

@author: hexiaoni
"""

from tqdm import tqdm
from geopy import Point, distance

import yaml

def get_IP_matrix_dictionary(df,item_sim):
    # get an initial item price dataframe(without onehot encoding)
    # drop duplicates
    df_temp = df[['business_num_id', 'price']].drop_duplicates()
    # with nontype with string "NaN"
    df_temp.fillna(value = "NaN", inplace = True)
    
    for i in tqdm(range(df_temp.shape[0])):
    # find all the items with no price
        if df_temp[df_temp['business_num_id'] == i]['price'].values[0] == "NaN":
            # get the index of the second large number in the similarity matrix
            temp_l = list(item_sim[i])
            index = [temp_l.index(x) for x in sorted(temp_l, reverse=True)[:2]][1]
            # get the dollar sign of the similar item
            dollar_of_sim_item = df_temp[df_temp['business_num_id'] == index]['price'].values[0]
            # replace the Nan
            df_temp.loc[df_temp['business_num_id'] == i,"price"] = dollar_of_sim_item

    # assign single dollar sign($) to the ones still with no price tag(since there is no items that are similar to this item)
    df_temp.loc[df_temp["price"] == "NaN","price"] = "$"

    # One hot encoding
    #note that the last column is price__$$$$
    #cat_columns = ["price"]
    #df_processed = pd.get_dummies(df_temp, prefix_sep="_",
    #                          columns=cat_columns)
    
    df_processed = df_temp.copy()
    df_processed['Price'] = df_processed.apply (lambda row: len(row.price), axis=1)
    
    #drop the $ column
    df_processed = df_processed.drop('price', 1)
    
    #Adding additional column of price label, range 1-4
    #df_preprocessed['Price_label'] = df.apply (lambda row: label_price(row), axis=1)
    df_processed.set_index("business_num_id", drop=True, inplace=True)
    I_P_dictionary = df_processed.to_dict()['Price']
    df_processed.reset_index(level=0, inplace=True)
    
    return df_processed, I_P_dictionary

def get_IS_dictionary(df):
    df_IS = df[['business_num_id', 'business_stars']].drop_duplicates()
    df_IS.set_index("business_num_id", drop=True, inplace=True)
    IS_dictionary = df_IS.to_dict()['business_stars']
    df_IS.reset_index(level=0, inplace=True)
    
    return IS_dictionary

# Input a list of prediction matrix
def get_ID_dictionary(df,prediction_matrix,intersection):
    ID_dictionary = dict()
    length = len(prediction_matrix)
    
    for j in tqdm(range(length)):
        #Save the coordinates of the business id to a dictionary 
        coordinateDict = yaml.safe_load(df[df["business_num_id"] == prediction_matrix[j]].iloc[0].coordinates)
    
        #Load the business latitude and longitude
        test_point = Point(coordinateDict['latitude'],coordinateDict['longitude'])

        #Get the distance with the test point, unit in km 
        result = round(distance.distance(intersection,test_point).kilometers,1)

        ID_dictionary[prediction_matrix[j]] = result
        
    return ID_dictionary

def get_intersection():
    
    yonge_and_finch = Point("43.779824, -79.415665")
    bloor_and_bathurst = Point("43.665194,-79.411208")
    queen_and_spadina = Point("43.648772,-79.396259")
    bloor_and_yonge = Point("43.670409,-79.386814")
    dundas_and_yonge = Point("43.6561,-79.3802")
    spadina_and_dundas = Point("43.653004,-79.398082")
    
    intersection_yonge_and_finch = yonge_and_finch
    intersection_bloor_and_bathurst = bloor_and_bathurst
    intersection_queen_and_spadina = queen_and_spadina
    intersection_bloor_and_yonge = bloor_and_yonge
    intersection_dundas_and_yonge = dundas_and_yonge
    intersection_spadina_and_dundas = spadina_and_dundas
    
    return intersection_yonge_and_finch, intersection_bloor_and_bathurst, intersection_spadina_and_dundas,\
           intersection_queen_and_spadina, intersection_bloor_and_yonge, intersection_dundas_and_yonge