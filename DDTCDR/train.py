import argparse
import pandas as pd
from engine import Engine
from data import SampleGenerator
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

parser = argparse.ArgumentParser('DDTCDR')
# Path Arguments
parser.add_argument('--num_epoch', type=int, default=100,help='number of epoches')
parser.add_argument('--batch_size', type=int, default=1024,help='batch size')
parser.add_argument('--lr', type=int, default=1e-2,help='learning rate')
parser.add_argument('--latent_dim', type=int, default=8,help='latent dimensions')
parser.add_argument('--alpha', type=int, default=0.03,help='dual learning rate')
parser.add_argument('--cuda', action='store_true',help='use of cuda')
args = parser.parse_args()

def dictionary(terms):
    term2idx = {}
    idx2term = {}
    for i in range(len(terms)):
        term2idx[terms[i]] = i
        idx2term[i] = terms[i]
    return term2idx, idx2term

mlp_config = {'num_epoch': args.num_epoch,
              'batch_size': args.batch_size,
              'optimizer': 'sgd',
              'lr': args.lr,
              'latent_dim': args.latent_dim,
              'nlayers':1,
              'alpha':args.alpha,
              'layers': [2*args.latent_dim,64,args.latent_dim],  # layers[0] is the concat of latent user vector & latent item vector
              'use_cuda': args.cuda,
              'pretrain': False,}

#Load Datasets
book = pd.read_csv('book.csv')
movie = pd.read_csv('movie.csv')
book['user_embedding'] = book['user_embedding'].map(eval)
book['item_embedding'] = book['item_embedding'].map(eval)
movie['user_embedding'] = movie['user_embedding'].map(eval)
movie['item_embedding'] = movie['item_embedding'].map(eval)

sample_book_generator = SampleGenerator(ratings=book)
evaluate_book_data = sample_book_generator.evaluate_data
sample_movie_generator = SampleGenerator(ratings=movie)
evaluate_movie_data = sample_movie_generator.evaluate_data

config = mlp_config
engine = Engine(config)
best_MSE = 1
book_hist_MAE = []
movie_hist_MAE = []
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_book_loader = sample_book_generator.instance_a_train_loader(config['batch_size'])
    train_movie_loader = sample_movie_generator.instance_a_train_loader(config['batch_size'])
    engine.train_an_epoch(train_book_loader, train_movie_loader, epoch_id=epoch)
    book_MSE, book_MAE, movie_MSE, movie_MAE = engine.evaluate(evaluate_book_data, evaluate_movie_data, epoch_id=epoch)
    book_hist_MAE.append(book_MAE)
    movie_hist_MAE.append(movie_MAE)
    
best_mae_book = min(book_hist_MAE)
best_mae_movie = min(movie_hist_MAE)

# The below part is using SVD on the same train test data with the same evaluation protocol

print('-' * 30 + "SVD Model Training" + '-' * 30) 
book = pd.read_csv('book.csv')
movie = pd.read_csv('movie.csv')

def normalize(ratings):
    """normalize into [0, 1] from [0, max_rating]"""
    ratings = deepcopy(ratings)
    max_rating = ratings.rating.max()
    ratings['rating'] = ratings.rating * 1.0 / max_rating
    return ratings

book = normalize(book)
movie = normalize(movie)

cut = 4 * len(book) // 5
train_book = book[:cut]
test_book = book[cut:]

cut_movie = 4 * len(movie) // 5
train_movie = movie[:cut_movie]
test_movie = movie[cut_movie:]

from surprise import SVD, Dataset, Reader
from surprise import accuracy
import random
import numpy as np

#  Movie data
my_seed = 13
random.seed(my_seed)
np.random.seed(my_seed)
reader = Reader(rating_scale=(0,1))
training_data = Dataset.load_from_df(train_movie[['userId', 'itemId', 'rating']], reader)
testing_data = Dataset.load_from_df(test_movie[['userId', 'itemId', 'rating']], reader)
algo = SVD(verbose = True, n_factors = 10)
training_data = training_data.build_full_trainset()
testing_data = testing_data.build_full_trainset().build_testset()

algo.fit(training_data,)
training_eval = training_data.build_testset()
train_pre = algo.test(training_eval)
train_mae = accuracy.mae(train_pre, verbose=False)
test_pre = algo.test(testing_data)
test_mae_movie = accuracy.mae(test_pre, verbose=False)

print(f"The only using in-domain movie data on SVD: {round(test_mae_movie, 3)}")

# Book data
my_seed = 13
random.seed(my_seed)
np.random.seed(my_seed)
reader = Reader(rating_scale=(0,1))
training_data = Dataset.load_from_df(train_book[['userId', 'itemId', 'rating']], reader)
testing_data = Dataset.load_from_df(test_book[['userId', 'itemId', 'rating']], reader)
algo = SVD(verbose = True, n_factors = 10)
training_data = training_data.build_full_trainset()
testing_data = testing_data.build_full_trainset().build_testset()

algo.fit(training_data,)
training_eval = training_data.build_testset()
train_pre = algo.test(training_eval)
train_mae = accuracy.mae(train_pre, verbose=False)
test_pre = algo.test(testing_data)
test_mae_book= accuracy.mae(test_pre, verbose=False)

print(f"The only using in-domain book data on SVD: {round(test_mae_book, 3)}")
    
    
from tabulate import tabulate

headers = ["Model", "Movie data", "Book data"]
tab = [
    ["DDTCDR", round(best_mae_movie, 3), round(best_mae_book, 3)],
    ["SVD", round(test_mae_movie, 3), round(test_mae_book, 3)],
    
]

print(tabulate(tab, headers = headers, tablefmt="fancy_grid"))