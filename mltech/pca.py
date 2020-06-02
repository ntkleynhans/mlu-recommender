from sklearn import datasets
from sklearn.decomposition import PCA
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()
iris = datasets.load_iris()

numSamples, numFeatures = iris.data.shape
print('Iris Data:')
print(numSamples)
print(numFeatures)
print(list(iris.target_names))

X = iris.data
pca = PCA(n_components=2, whiten=True).fit(X)
X_pca = pca.transform(X)

print('EigenVectors:')
print(pca.components_)

print('Norm EigenValues:')
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

data = np.hstack( (X_pca, iris.target.reshape(X_pca.shape[0], 1)) )
df = pd.DataFrame(data=data, columns=['X', 'Y', 'Hue'])
df['Hue'] = df['Hue'].apply(lambda x: int(x))
print(df.head())

g = sns.lmplot(x='X',y='Y',hue='Hue',data=df)
g._legend.set_title('Iris')
for t, l in zip(g._legend.texts, iris.target_names): t.set_text(l)

plt.show()
