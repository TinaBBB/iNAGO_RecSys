# -*- coding: utf-8 -*-
"""timeSplit

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KtCKIPeD9asdwa7pdXp4XMcHFEbx90XD
"""

from tqdm import tqdm

import numpy as np
import scipy.sparse as sparse

def time_ordered_split(rating_matrix, ratingWuserAvg_matrix, timestamp_matrix, ratio=[0.5, 0.2, 0.3],
                       implicit=True, remove_empty=False, threshold=3,
                       sampling=False, sampling_ratio=0.1):
    """
    Split the data to train,valid,test by time
    ratio:  train:valid:test
    threshold: for implicit representation
    """
    
    if implicit:
        temp_rating_matrix = sparse.csr_matrix(rating_matrix.shape)
        temp_rating_matrix[(rating_matrix > threshold).nonzero()] = 1
        rating_matrix = temp_rating_matrix
        timestamp_matrix = timestamp_matrix.multiply(rating_matrix)
        #ratingWuserAvg_matrix = ratingWuserAvg_matrix.multiply(rating_matrix)

    nonzero_index = None

    #Default false, not removing empty columns and rows
    #Should not have this case, since users should have at least 1 record of 4,5 
    #And restuarant should have at least 1 record of 4,5 
    if remove_empty:
        # Remove empty columns. record original item index
        nonzero_index = np.unique(rating_matrix.nonzero()[1])
        rating_matrix = rating_matrix[:, nonzero_index]
        timestamp_matrix = timestamp_matrix[:, nonzero_index]
        ratingWuserAvg_matrix = ratingWuserAvg_matrix[:, nonzero_index]

        # Remove empty rows. record original user index
        nonzero_rows = np.unique(rating_matrix.nonzero()[0])
        rating_matrix = rating_matrix[nonzero_rows]
        timestamp_matrix = timestamp_matrix[nonzero_rows]
        ratingWuserAvg_matrix = ratingWuserAvg_matrix[nonzero_rows]

    user_num, item_num = rating_matrix.shape

    rtrain = []
    rtrain_userAvg = []
    rtime = []
    rvalid = []
    rvalid_userAvg = []
    rtest = []
    rtest_userAvg = []
    # Get the index list corresponding to item for train,valid,test
    item_idx_train = []
    item_idx_valid = []
    item_idx_test = []
    
    for i in tqdm(range(user_num)):
        #Get the non_zero indexs, restuarants where the user visited/liked if implicit 
        item_indexes = rating_matrix[i].nonzero()[1]
        
        #Get the data for the user
        data = rating_matrix[i].data
        
        #Get time stamp value 
        timestamp = timestamp_matrix[i].data
        
        #Get review stars with user avg data 
        if implicit == False:
            dataWuserAvg = ratingWuserAvg_matrix[i].data
        
        #Non zero reviews for this user
        num_nonzeros = len(item_indexes)
        
        #If the user has at least one review
        if num_nonzeros >= 1:
            #Get number of test and valid data 
            #train is 30%
            num_test = int(num_nonzeros * ratio[2])
            #validate is 50%
            num_valid = int(num_nonzeros * (ratio[1] + ratio[2]))

            valid_offset = num_nonzeros - num_valid
            test_offset = num_nonzeros - num_test

            #Sort the timestamp for each review for the user
            argsort = np.argsort(timestamp)
            
            #Sort the reviews for the user according to the time stamp 
            data = data[argsort]
            
            #Sort the review with user avg accoridng to the time stamp
            if implicit == False:
                dataWuserAvg = dataWuserAvg[argsort]
            
            #Non-zero review index sort according to time
            item_indexes = item_indexes[argsort]
            
            #list of ratings, num of valid_offset index, index where there's non-zeros
            rtrain.append([data[:valid_offset], np.full(valid_offset, i), item_indexes[:valid_offset]])
            
            if implicit == False:
                #Changing valid set to binary
                count=valid_offset
                for eachData in data[valid_offset:test_offset]:
                    #if rating-avgRating > 0 then like
                    if eachData >= 4:
                        data[count] = 1
                    else:
                        data[count] = 0
                    count += 1
                
            #50%-70%
            rvalid.append([data[valid_offset:test_offset], np.full(test_offset - valid_offset, i),
                           item_indexes[valid_offset:test_offset]])
            #remaining 30%
            rtest.append([data[test_offset:], np.full(num_test, i), item_indexes[test_offset:]])
            
            if implicit == False:
                #Now for the rating matrix that considers user average rating
                #list of ratings, num of valid_offset index, index where there's non-zeros
                rtrain_userAvg.append([dataWuserAvg[:valid_offset], np.full(valid_offset, i), item_indexes[:valid_offset]])
                #50%-70%

                #Changing valid set to binary
                count=valid_offset
                for eachData in dataWuserAvg[valid_offset:test_offset]:
                    #if rating-avgRating > 0 then like
                    if eachData > 0:
                        dataWuserAvg[count] = 1
                    else:
                        dataWuserAvg[count] = 0
                    count += 1

                rvalid_userAvg.append([dataWuserAvg[valid_offset:test_offset], np.full(test_offset - valid_offset, i),
                               item_indexes[valid_offset:test_offset]])

                #Change test set to binary even we don't use it
                countTest = test_offset
                for eachData in dataWuserAvg[test_offset:]:
                    #if rating-avgRating > 0 then like
                    if eachData > 0:
                        dataWuserAvg[count] = 1
                    else:
                        dataWuserAvg[count] = 0
                    count += 1


                #remaining 30%
                rtest_userAvg.append([dataWuserAvg[test_offset:], np.full(num_test, i), item_indexes[test_offset:]])
                
            item_idx_train.append(item_indexes[:valid_offset])
            
#             item_idx_valid.append(item_indexes[valid_offset:test_offset])
#             item_idx_test.append(item_indexes[test_offset:])
        else:
            item_idx_train.append([])
#             item_idx_valid.append([])
#             item_idx_test.append([])
    
    rtrain = np.array(rtrain)
    rvalid = np.array(rvalid)
    rtest = np.array(rtest)
    if implicit == False:
        rtrain_userAvg = np.array(rtrain_userAvg)
        rvalid_userAvg = np.array(rvalid_userAvg)
        rtest_userAvg = np.array(rtest_userAvg)

    #take non-zeros values, row index, and column (non-zero) index and store into sparse matrix
    rtrain = sparse.csr_matrix((np.hstack(rtrain[:, 0]), (np.hstack(rtrain[:, 1]), np.hstack(rtrain[:, 2]))),
                               shape=rating_matrix.shape, dtype=np.float32)
    rvalid = sparse.csr_matrix((np.hstack(rvalid[:, 0]), (np.hstack(rvalid[:, 1]), np.hstack(rvalid[:, 2]))),
                               shape=rating_matrix.shape, dtype=np.float32)
    rtest = sparse.csr_matrix((np.hstack(rtest[:, 0]), (np.hstack(rtest[:, 1]), np.hstack(rtest[:, 2]))),
                              shape=rating_matrix.shape, dtype=np.float32)
    
    if implicit == False:
        rtrain_userAvg = sparse.csr_matrix((np.hstack(rtrain_userAvg[:, 0]), (np.hstack(rtrain_userAvg[:, 1]), np.hstack(rtrain_userAvg[:, 2]))),
                                   shape=rating_matrix.shape, dtype=np.float32)
        rvalid_userAvg = sparse.csr_matrix((np.hstack(rvalid_userAvg[:, 0]), (np.hstack(rvalid_userAvg[:, 1]), np.hstack(rvalid_userAvg[:, 2]))),
                                   shape=rating_matrix.shape, dtype=np.float32)
        rtest_userAvg = sparse.csr_matrix((np.hstack(rtest_userAvg[:, 0]), (np.hstack(rtest_userAvg[:, 1]), np.hstack(rtest_userAvg[:, 2]))),
                                  shape=rating_matrix.shape, dtype=np.float32)

    return rtrain, rvalid, rtest,rtrain_userAvg, rvalid_userAvg, rtest_userAvg, nonzero_index, timestamp_matrix, item_idx_train, item_idx_valid, item_idx_test

#Item idex matrix stores the reivews starts
#This function returns a list of index for the reviews included in training set 
def get_corpus_idx_list(df, item_idx_matrix):
    """
    Input: 
    df: total dataframe
    item_idx_matrix: train index list got from time_split 
    Output: row index in original dataframe for training data by time split
    """
    lst = []
    #For all the users: 5791
    for i in tqdm(range(len(item_idx_matrix))):
        
        #find row index where user_num_id is i
        a = df.index[df['user_num_id'] == i].tolist()
        
        #loop through the busienss id that the user i reviewed for in offvalid set 
        for item_idx in  item_idx_matrix[i]:
            
            #get the row index for reviews for business that the user liked in the train set
            b = df.index[df['business_num_id'] == item_idx].tolist()
            
            #Find the index for which this user liked, one user only rate a business once
            idx_to_add = list(set(a).intersection(b))
            
            if idx_to_add not in lst:
                lst.extend(idx_to_add)
    return lst