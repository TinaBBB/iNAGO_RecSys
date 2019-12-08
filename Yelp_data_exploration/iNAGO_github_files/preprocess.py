from models.knn import KNN
from utils.util import get_POI_df, get_POI_df_toronto, get_review_df_toronto, filter_by_city, select_restaurants, \
    return_POI_lat_long_df, save_POI_lat_long_df, get_review_df, \
    binarized_star, merge_df, pandas_to_dict, save_user_id_dict_pickle, \
    train_test_split, df_to_sparse, save_npz_data
import argparse


def main(args):

    # Import business dataset
    #business_df = get_POI_df(args.path+'yelp_dataset_business.json')
    
    #Import Cleaned_Toronto_Business
    business_df = get_POI_df_toronto(args.path+'Cleaned_Toronto_Business.json')
    
    # Filter business dataset by city
    #business_df = filter_by_city(business_df, city=args.city)    comment out since all toronto restaurants

    # Filter business dataset by restaurants
    #business_df = select_restaurants(business_df)     comment out since already cleaned

    # Import review dataset
    #review_df = get_review_df(args.path+'yelp_dataset_review.json')

    #Import Toronto review dataset
    review_df = get_review_df_toronto(args.path+'Cleaned_Toronto_Reviews.json')
    
    # Binarize review stars, adding a new column called review_stars_binarized
    review_df = binarized_star(review_df)

    print('review df columns',  review_df.columns)
    print('business_df columns', business_df.columns)
    
    # Merge business df and review df
    rating_df = merge_df(review_df, business_df, on_column='business_id', how_merge='inner', columns=["user_id", "business_id", "date", "review_stars", "review_text", "review_stars_binary", "categories",  "latitude", "longitude"], sort_column='date')
    
    num_cols = rating_df.business_id.nunique()
    num_rows = rating_df.user_id.nunique()
    print('unique businesses:', num_cols, 'unique users', num_rows)
    print('unique user id:', rating_df.user_id.nunique())
    
    # Assign numbers to user_id -> user_num_id
    rating_df['user_num_id'] = rating_df.user_id.astype('category').\
    cat.rename_categories(range(0, rating_df.user_id.nunique()))
    rating_df['user_num_id'] = rating_df['user_num_id'].astype('int')
    
    #Encode business_num_id
    rating_df['business_num_id'] = rating_df.business_id.astype('category').\
       cat.rename_categories(range(0, rating_df.business_id.nunique()))
    rating_df['business_num_id'] = rating_df['business_num_id'].astype('int')
    rating_df = rating_df.reset_index()
    # Get all restaurants latitude and longitude df
    POI_lat_long_df = return_POI_lat_long_df(rating_df)
    
    # Export all restaurants latitude and longitude df
    save_POI_lat_long_df(args.path+"POI_lat_long_df", POI_lat_long_df)
    
    # Get pandas user_id and business_id dict
    user_id_dict = pandas_to_dict(rating_df, "user_id", "user_num_id")

    # Export dict to disk as json
    save_user_id_dict_pickle(args.path, user_id_dict, 'user_id_dict')
    
    # Split into train and test dataset
    train_df, test_df = train_test_split(rating_df)

    # Form train set and test set
    #train_set generate UI matrix 
    '''
        Computing binary rating UI for train and test
        Computing raw rating UI for train and test 
        Combine both for entire dataset 
    '''
    
    #Getting both thresholded and raw user item review (UI) matrix 
    train_set_binary = df_to_sparse(train_df, num_rows, num_cols)
    test_set_binary = df_to_sparse(test_df, num_rows, num_cols)
    train_set_rawRating  = df_to_sparse(train_df, num_rows, num_cols, binarySetting=False)
    train_set_rawRating = df_to_sparse(train_df, num_rows, num_cols, binarySetting=False)
    entire_set_binary = train_set_binary + test_set_binary
    entire_set_raw = train_set_rawRating + train_set_rawRating
    
    #Sorting both binary, rawRating UI matrix, and entire UI matrix
    save_npz_data(args.path+"toronto_train_set_binary.npz", train_set_binary)
    save_npz_data(args.path+"toronto_test_set_binary.npz", test_set_binary)
    save_npz_data(args.path+"toronto_train_set_rawRating.npz", train_set_rawRating)
    save_npz_data(args.path+"toronto_test_set_rawRating.npz", train_set_rawRating)
    save_npz_data(args.path+"toronto_entire_set_binary.npz", entire_set_binary)
    save_npz_data(args.path+"toronto_entire_set_rawRating.npz", entire_set_raw)
    
    
    #To compute item similarity using IK 
    IK_matrix_train = get_I_K(train_df)
    IK_matrix_entire = get_I_K(rating_df)
    
    # Get item similarity
    item_IK_model_train = KNN()
    item_IK_model_train.fit(X=IK_matrix_train.T)
    sparse_item_similarity_train = item_IK_model_train.get_weights()
    save_npz_data(args.path+"item_similarity_train.npz", sparse_item_similarity_train)
    
    item_IK_model_entire = KNN()
    item_IK_model_entire.fit(X=IK_matrix_entire.T)
    sparse_item_similarity_entire = item_IK_model_entire.get_weights()
    save_npz_data(args.path+"item_similarity_entire.npz", sparse_item_similarity_entire)
    
    
    # Get user similarity for train set 
    user_model_trainBinary = KNN()
    user_model_trainBinary.fit(X=train_set_binary)
    
    sparse_user_similarity_train = user_model_trainBinary.get_weights()
    save_npz_data(args.path+"user_similarity_trainSet.npz", sparse_user_similarity_train)
    
    user_model_entireBinary = KNN()
    user_model_entireBinary.fit(X=entire_set_binary)
    
    sparse_user_similarity_entire = user_model_entireBinary.get_weights()
    save_npz_data(args.path+"user_similarity_entireSet.npz", sparse_user_similarity_entire)
    
    
if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('-d', dest='path', default="data/")
    parser.add_argument('-c', dest='city', default="Toronto")
    args = parser.parse_args()

    main(args)
