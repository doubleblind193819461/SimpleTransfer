import pandas as pd
import numpy as np
from tqdm import tqdm
from surprise import SVD, Dataset, Reader, SVDpp
from surprise import accuracy
from sklearn.preprocessing import LabelEncoder
import random

import warnings
warnings.filterwarnings("ignore")



# import data
# ML1M datasets

movies_df = pd.read_csv('../datasets/ml-1m/movies.dat', sep='::', names=['item_id', 'title', 'genres'], engine = "python", encoding = "ISO-8859-1")
movies_df['item_id'] = movies_df['item_id'].astype(np.int)

ratings_df = pd.read_csv('../datasets/ml-1m/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine = "python")
ratings_df['user_id'] = ratings_df['user_id'].astype(np.int)
ratings_df['item_id'] = ratings_df['item_id'].astype(np.int)
data_df_1m = pd.merge(ratings_df, movies_df, how = "inner", on = "item_id")


# ML100K

ratings_df_100k = pd.read_csv("../datasets/ml-100k/u.data", 
                         sep = "\t",
                         names = ['user_id', 'item_id', 'rating', 'timestamp'], 
                         engine = "python")

m_cols = ["item_id", "title"]
movies_df_100k = pd.read_csv('../datasets/ml-100k/u.item',
                        names = m_cols,
                        usecols = range(2),
                        sep='|', 
                        engine = "python",
                        encoding = "ISO-8859-1")

data_df_100 = pd.merge(ratings_df_100k, movies_df_100k, how = "inner", on = "item_id")


arr_100k = data_df_100["title"].values
arr_1m = data_df_1m["title"].values
all_titles = np.concatenate((arr_1m, arr_100k))
all_unique = np.unique(all_titles)


lbe = LabelEncoder()
lbe.fit(all_unique)

data_df_100["item_id"] = lbe.transform(data_df_100["title"])
data_df_1m["item_id"] = lbe.transform(data_df_1m["title"])

print(f"number of unique item in ml100k: {data_df_100['item_id'].nunique()}, number of unique item in ml1m: {data_df_1m['item_id'].nunique()}")

lbe = LabelEncoder()
data_df_100["user_id"] = lbe.fit_transform(data_df_100["user_id"])
print(f"number of unique user in ml100k: {data_df_100['user_id'].nunique()}")


lbe = LabelEncoder()
data_df_1m["user_id"] = lbe.fit_transform(data_df_1m["user_id"])
data_df_1m["user_id"] = data_df_1m["user_id"] + 943


from sklearn.model_selection import train_test_split
# 1m
train_df_1m, rest_df_1m = train_test_split(data_df_1m, test_size = 0.2, random_state = 133)
test_df_1m, valid_df_1m = train_test_split(rest_df_1m , test_size = 0.5, random_state = 133)
# 100k
train_df_100, rest_df_100 = train_test_split(data_df_100, test_size = 0.2, random_state = 133)
test_df_100, valid_df_100 = train_test_split(rest_df_100, test_size = 0.5, random_state = 133)

train_df_100.drop(columns= ["title"], inplace = True)
train_df_1m = train_df_1m.drop(columns = ["title", "genres"])
train_df = pd.concat([train_df_100, train_df_1m])


random.seed(133)
np.random.seed(133)

reader = Reader(rating_scale=(1, 5))
training_data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
testing_data = Dataset.load_from_df(test_df_100[['user_id', 'item_id', 'rating']], reader)
algo = SVD(verbose = True)
training_data = training_data.build_full_trainset()
testing_data = testing_data.build_full_trainset().build_testset()

algo.fit(training_data,)
training_eval = training_data.build_testset()
train_pre = algo.test(training_eval)
train_rmse = accuracy.rmse(train_pre, verbose=False)
test_pre = algo.test(testing_data)
test_rmse_concat = accuracy.rmse(test_pre, verbose=False)

print(f"the rmse on ml100k test data if we concatenate datasets together:{test_rmse_concat}")


random.seed(133)
np.random.seed(133)

reader = Reader(rating_scale=(1, 5))
training_data = Dataset.load_from_df(train_df_100[['user_id', 'item_id', 'rating']], reader)
testing_data = Dataset.load_from_df(test_df_100[['user_id', 'item_id', 'rating']], reader)
algo = SVD(verbose = True)
training_data = training_data.build_full_trainset()
testing_data = testing_data.build_full_trainset().build_testset()

algo.fit(training_data,)
training_eval = training_data.build_testset()
train_pre = algo.test(training_eval)
train_rmse = accuracy.rmse(train_pre, verbose=False)
test_pre = algo.test(testing_data)
test_rmse_100 = accuracy.rmse(test_pre, verbose=False)

print(f"the rmse on test if we train ml100k alone:{test_rmse_100}")


from tabulate import tabulate

headers = ["Model", "ML100K Alone", "ML100K with ML1M"]
tab = [["SVD", round(test_rmse_100, 3), round(test_rmse_concat, 3)]]

print(tabulate(tab, headers = headers, tablefmt="fancy_grid"))