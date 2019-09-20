# Yelp_data_exploration
Data Exploration on Yelp Open dataset

Contain files:
1. Data Exploration of Review Text: some data exploration
2. KNN final: pre-process, models and results

Original Review dataset is called: yelp_academic_dataset_review (4GB)
Separated into 10 different files for easier access: in data folder from 0-9

In folder Yelp dataset, I saved the User-Item matrices into train,validation and test matrices, each row was user_id and column was item_id, if the user give the item(restuarant) a star over 3, the value will be 1, otherwise 0 (implicit representation).

Time ordered split is how I separated the data into train,valid and test. More specifically, each user has n reviews with different timestamp, and first 50% are separated into train, last 20% into test etc. 

I utilized the review text a bit dirty with some functions e.g. get_corpus_idx_list, will edit this later.

In the notebook, I use 3 different models (including combinations of 2 of the models) to evaluate the knn performance on the dataset, leveraging the review texts. 


Yelp Dataset:
https://www.yelp.com/dataset 

Some code borrowing from:
KNN Code https://github.com/k9luo/K-Nearest-Neighbors-Recommendation-on-Yelp
