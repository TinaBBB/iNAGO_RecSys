'import classes and functions'
from utils.io import get_yelp_df, get_rating_timestamp_matrix, get_UC_Matrix
from utils.io import save_dataframe_csv, load_dataframe_csv, save_numpy, load_numpy 
from utils.io import saveDictToJson, loadDict, save_numpy_csr, load_numpy_csr
from utils.split import time_ordered_splitModified
from utils.train import train
from utils.ikGeneration import ikGeneration
from utils.getDictionary import get_IP_matrix_dictionary, get_IS_dictionary, get_ID_dictionary, get_intersection
from utils.progress import WorkSplitter
from utils.predict import predict, prediction

import argparse
import numpy as np
import scipy.sparse as sparse

def main(args):
    progress = WorkSplitter()


    progress.section("Parameter Setting")
    print("Data Path: {}".format(args.data_dir))
    reviewJsonToronto = args.data_dir+args.data_name
    
    
    progress.section("Load data")
    df = get_yelp_df(path ='', filename=reviewJsonToronto, sampling= True)
    print('Data loaded sucessfully')
    
    
    progress.section("Matrix Generation")
    rating_matrix,timestamp_matrix,I_C_matrix,IC_dictionary = get_rating_timestamp_matrix(df)
    # get ratingWuserAvg_matrix
    rating_array = rating_matrix.toarray()
    user_average_array = rating_array.sum(axis = 1)/np.count_nonzero(rating_array,axis = 1)
    init_UI = np.zeros(rating_array.shape)
    init_UI[rating_array.nonzero()] = 1
    
    #Creating rating with user average array array
    for i in range(user_average_array.shape[0]):
        init_UI[i] = init_UI[i] * (user_average_array[i]-0.001) 
    user_average_array = init_UI
    ratingWuserAvg_array = rating_array - user_average_array
    ratingWuserAvg_matrix=sparse.csr_matrix(ratingWuserAvg_array)
    
    
    progress.section("Split for training")
    rtrain_implicit, rvalid_implicit, rtest_implicit, rtrain_userAvg_implicit, rvalid_userAvg_implicit, \
    rtest_userAvg_implicit, nonzero_index, rtime, item_idx_matrix_train_implicit,item_idx_matrix_valid_implicit, item_idx_matrix_test_implicit \
    = time_ordered_splitModified(rating_matrix=rating_matrix, ratingWuserAvg_matrix=ratingWuserAvg_matrix, timestamp_matrix=timestamp_matrix,
                                                                     ratio=[0.5,0.2,0.3],
                                                                     implicit=True,
                                                                     remove_empty=False, threshold=3,sampling=False, 
                                                                     sampling_ratio=0.1, trainSampling=0.95)
    
    rtrain, rvalid, rtest, rtrain_userAvg, rvalid_userAvg, rtest_userAvg, nonzero_index, rtime, \
    item_idx_matrix_train,item_idx_matrix_valid, item_idx_matrix_test = time_ordered_splitModified(rating_matrix=rating_matrix,
                                                                     ratingWuserAvg_matrix=ratingWuserAvg_matrix, timestamp_matrix=timestamp_matrix,
                                                                     ratio=[0.5,0.2,0.3],
                                                                     implicit=False,
                                                                     remove_empty=False, threshold=3,
                                                                     sampling=False, sampling_ratio=0.1, 
                                                                     trainSampling=0.95)  
    
    rtrain = rtrain + rvalid + rtest
    rtrain_implicit = rtrain_implicit + rvalid_implicit + rtest_implicit
    
    
    progress.section("Get UC Matrix")
    #Get UC matrices
    U_C_matrix_explicit,U_C_matrix_implicit = get_UC_Matrix(I_C_matrix,rtrain_implicit)
    
    
    progress.section("Get IK Similarity")
    IK_MATRIX = ikGeneration(df)
    IK_similarity = train(IK_MATRIX)
    
    '''
    progress.section("Get IC Similarity")
    IC_similarity = train(I_C_matrix)
    '''
    
    progress.section("Get IP, IS, ID Dictionary")
    #intersection = get_intersection()
    intersection_yonge_and_finch, intersection_bloor_and_bathurst, intersection_spadina_and_dundas,\
    intersection_queen_and_spadina, intersection_bloor_and_yonge, intersection_dundas_and_yonge = get_intersection()
    IP_df, IP_dictionary = get_IP_matrix_dictionary(df, IK_similarity)
    IS_dictionary = get_IS_dictionary(df)
    #ID_dictionary = get_ID_dictionary(df,list(set(df['business_num_id'])),intersection)
    ID_dictionary_yonge_and_finch = get_ID_dictionary(df,list(set(df['business_num_id'])),intersection_yonge_and_finch)
    ID_dictionary_bloor_and_bathurst = get_ID_dictionary(df,list(set(df['business_num_id'])),intersection_bloor_and_bathurst)
    ID_dictionary_spadina_and_dundas = get_ID_dictionary(df,list(set(df['business_num_id'])),intersection_spadina_and_dundas)
    ID_dictionary_queen_and_spadina = get_ID_dictionary(df,list(set(df['business_num_id'])),intersection_queen_and_spadina)
    ID_dictionary_bloor_and_yonge = get_ID_dictionary(df,list(set(df['business_num_id'])),intersection_bloor_and_yonge)
    ID_dictionary_dundas_and_yonge = get_ID_dictionary(df,list(set(df['business_num_id'])),intersection_dundas_and_yonge)
    
    
    progress.section("user item predict")
    user_item_prediction_score = predict(rtrain, 110, IK_similarity, item_similarity_en= True)
    UI_Prediction_Matrix = prediction(user_item_prediction_score, rtrain)
    

    progress.section("Save datafiles csv")
    save_dataframe_csv(df,args.data_dir,"Dataframe")
    
    
    progress.section("Save datafiles JSON")
    saveDictToJson(IC_dictionary, args.data_dir, 'icDictionary', trainOrTest='train')
    saveDictToJson(IP_dictionary, args.data_dir, 'ipDictionary', trainOrTest='train')
    saveDictToJson(IS_dictionary, args.data_dir, 'isDictionary', trainOrTest='train')
    #saveDictToJson(ID_dictionary, args.data_dir, 'idDictionary', trainOrTest='train')
    saveDictToJson(ID_dictionary_yonge_and_finch, args.data_dir, 'idDictionary_yongefinch', trainOrTest='train')
    saveDictToJson(ID_dictionary_bloor_and_bathurst, args.data_dir, 'idDictionary_bloorbathurst', trainOrTest='train')
    saveDictToJson(ID_dictionary_spadina_and_dundas, args.data_dir, 'idDictionary_spadinadundas', trainOrTest='train')
    saveDictToJson(ID_dictionary_queen_and_spadina, args.data_dir, 'idDictionary_queenspadina', trainOrTest='train')
    saveDictToJson(ID_dictionary_bloor_and_yonge, args.data_dir, 'idDictionary_blooryonge', trainOrTest='train')
    saveDictToJson(ID_dictionary_dundas_and_yonge, args.data_dir, 'idDictionary_dundasyonge', trainOrTest='train')
    
    progress.section("Save datafiles Numpy")
    save_numpy_csr(rtrain, args.data_dir, "rtrain")
    save_numpy_csr(I_C_matrix, args.data_dir, "icmatrix")
    #save_numpy(user_item_prediction_score, args.data_dir, "predictionScore")
    save_numpy(IK_similarity, args.data_dir, "IKbased_II_similarity") #Tina requested for this name
    save_numpy(UI_Prediction_Matrix, args.data_dir, "UI_prediction_matrix")
    
    '''
    # Testing
    progress.section("Load datafiles")
    df = load_dataframe_csv(args.data_dir, "Dataframe.csv")
    print(type(df))
    print(df.head(5))
    
    rtrain = load_numpy_csr(args.data_dir, "rtrain.npz")
    print(type(rtrain))
    
    user_item_prediction_score = load_numpy(args.data_dir, "predictionScore.npz")
    print(type(user_item_prediction_score))
    
    user_item_prediction_score = load_numpy(args.data_dir, "predictionScore.npy")
    print(type(user_item_prediction_score))
    
    IC_Dictionary = loadDict(args.data_dir, "icDictionary.json")
    print(type(IC_Dictionary))
    
    ID_Dictionary = loadDict(args.data_dir, "idDictionary.json")
    print(type(ID_Dictionary))
    '''
    
if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Project_Initialization")
    
    parser.add_argument('--data_dir', dest='data_dir', default="../data/",
                        help='Directory path to the dataset. (default: %(default)s)')
    
    parser.add_argument('--data_name', dest='data_name', default='Cleaned_Toronto_Reviews.json',
                        help='File name of the raw dataset. (default: %(default)s)')
    
    args = parser.parse_args()

    main(args)