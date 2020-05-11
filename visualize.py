import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# argv[1] is a path to saved ALSModel
# argv[2] is a path to book_id_map.csv

spark = SparkSession.builder.getOrCreate()
model = ALSModel.load(sys.argv[1])

feat_dict = {}
for elem in model.itemFactors.collect():
    feat_dict[elem.id] = elem.features

books_genres = pd.read_json('goodreads_book_genres_initial.json', lines=True)
book_id_map = pd.read_csv(sys.argv[2], header=None, names=['id', 'book_id'])

# join genre data and id map, resultig columns are [book_id, id, map]
# book_id is id in Goodreads, id is id in model
df = maps.set_index('book_id').join(df.set_index('book_id'), how='left')
df.reset_index(inplace=True)

# drop books with no genres
indexNames = df[ df['genres'] == {} ].index
df.drop(indexNames, inplace=True)

# select most common genre for each book
df['genre'] = df.apply(lambda x: sorted(x['genres'].items(), key=lambda k:k[1], reverse=True)[0][0], axis=1)
df.drop(columns=['genres'], inplace=True)

# add latent factors to dataframe
df['features'] = df.apply(lambda x: feat_dict[x['id']], axis=1)

# Transform with t-SNE
X = np.stack(df['features'])
X_transformed = TSNE(n_components=2, n_jobs=6, n_iter_without_progress=150, perplexity=40, learning_rate=100).fit_transform(X)

# map each genre to color
genres = np.unique(df['genre'])
colors = ['b', 'g', 'r', 'c', 'gray', 'y', 'k', 'orange', 'magenta', 'pink', 'teal', 'purple', 'brown', 'lime']
genre_color_map = {}
for idx, g in enumerate(genres):
    genre_color_map[g] = colors[idx]

# plot for books
coloring = df.iloc[:n]['genre'].apply(lambda x: genre_color_map[x])
plt.figure(figsize=(20,20))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=coloring)
plt.savefig('books_t-SNE.png')
plt.show()



# plot for users
features = []
for elem in model.userFactors.collect():
    features.append(elem.features)

X = np.array(features)
X_transformed = TSNE(n_components=2, n_jobs=6, n_iter_without_progress=150, perplexity=70, learning_rate=300).fit_transform(X)
plt.figure(figsize=(20,20))
plt.scatter(X_transformed[:,0], X_transformed[:,1])
plt.savefig('users_t-SNE.png')
plt.show()