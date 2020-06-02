import gzip
import operator
import pandas as pd
import numpy as np
from scipy import spatial

def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance

def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv(gzip.open('../data/u.data.gz'), sep='\t', names=r_cols, usecols=range(3))
print('Ratings')
print(ratings.head())

movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
print('Movie Properties:')
print(movieProperties.head())

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print('Normalised Ratings')
print(movieNormalizedNumRatings.head())

movieDict = {}
with gzip.open('../data/u.item.gz') as f:
    for line in f:
        fields = line.decode("ISO-8859-1").rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))
print('Movie Meta:')
print(movieDict[1])

print('Compute Distance')
print(movieDict[2])
print(movieDict[4])
print(ComputeDistance(movieDict[2], movieDict[4]))

print('10 Nearest Neighbors:')
K = 10
avgRating = 0
neighbors = getNeighbors(1, K)
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print (movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))
    
avgRating /= K
print('Average Rating:')
print(avgRating)
