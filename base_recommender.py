import pandas as pd
import numpy as np
import gzip
import io

def baseSim(myRatings):
    simCandidates = pd.Series()
    for i in range(0, len(myRatings.index)):
        print ("Adding sims for " + myRatings.index[i] + "...")
        # Retrieve similar movies to this one that I rated
        sims = corrMatrix[myRatings.index[i]].dropna()
        # Now scale its similarity by how well I rated this movie
        sims = sims.map(lambda x: x * myRatings[i])
        # Add the score to the list of similarity candidates
        simCandidates = simCandidates.append(sims)
    return simCandidates

# Load the data
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv(gzip.open('u.data.gz'), sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv(gzip.open('u.item.gz'), sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

# Merge data sources
ratings = pd.merge(movies, ratings)
print('Ratings:')
print(ratings.head())

# Pivot table: user x movie
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
print('UserRatings:')
print(userRatings.head())

# Create correlation table
corrMatrix = userRatings.corr(method='pearson', min_periods=100)
print('UserRating Correlation:')
print(corrMatrix.head())

# A user from the dataset
myRatings = userRatings.loc[0].dropna()
print(myRatings)

simCandidates = baseSim(myRatings)
# Glance at our results so far:
print ("sorting...")
simCandidates.sort_values(inplace=True, ascending=False)
print('Similar Candidates:')
print(simCandidates.head(10))

# Group and sum candidates
simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace = True, ascending = False)
print('Group & Sum Candidates:')
print(simCandidates.head(10))

# Filter user ratings from results
filteredSims = simCandidates.drop(myRatings.index)
print('Final Similar Candidates:')
print(filteredSims.head(10))
