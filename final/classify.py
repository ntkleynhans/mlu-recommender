import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import keras
#from keras.models import Model
#from keras.layers import Input
#from keras.layers import Dense

sns.set(color_codes=True)

# LOAD THE DATA
df = pd.read_csv('../data/mammographic_masses.data.txt', delimiter = ',')
df.columns = ['BR', 'age', 'shape', 'margin', 'density', 'severity']
df = df.drop(columns=['BR'])
df = df.replace('?',np.nan)
df = df.dropna()

print(df.describe(include='all'))

#sns.pairplot(df)
#plt.show()

X = df[df.columns[:-1]].to_numpy()
Y = df[df.columns[-1]].to_numpy()

print(X.shape, Y.shape)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# NORMALISE
scaler = StandardScaler()
scaler.fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)


# DECISION TREE CLASSIFIER
print('\nDECISION TREE')
dtc = DecisionTreeClassifier()
dtc.fit(Xs_train, y_train)
print('Accuracy: ', dtc.score(Xs_test, y_test))

# XGBOOST
print('\nXGBOOST')
xgb = XGBClassifier()
print(cross_val_score(xgb, Xs_train, y_train,verbose=2))
xgb.fit(Xs_train, y_train)
y_pred = xgb.predict(Xs_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#SVM
print('\nSVM:LINEAR')
svm = SVC(kernel='linear')
print(cross_val_score(svm, Xs_train, y_train,verbose=2))

print('\nSVM:POLY')
svm = SVC(kernel='poly',degree=3)
print(cross_val_score(svm, Xs_train, y_train,verbose=2))

print('\nSVM:RBF')
svm = SVC()
print(cross_val_score(svm, Xs_train, y_train,verbose=2))


# KNN
print('\nKNN')
for n in range(2,20):
    print(f'\n{n}')
    neigh = KNeighborsClassifier(n_neighbors=n)
    print(cross_val_score(neigh,Xs_train, y_train, verbose=2))


# NORMALISE
mmscaler = MinMaxScaler()
mmscaler.fit(X_train)
Xm_train = mmscaler.transform(X_train)
Xm_test = mmscaler.transform(X_test)

print('\nMNB')
mnb = MultinomialNB()
print(cross_val_score(mnb, Xm_train, y_train,verbose=2))


# LOGISTIC REGRESSION
print('\nLR')
lr = LogisticRegression()
print(cross_val_score(lr,Xs_train, y_train, verbose=2))


#DNN
print('\nDNN')
inputs = keras.layers.Input(shape=(4,))
hidden1 = keras.layers.Dense(8, activation='relu')(inputs)
hidden2 = keras.layers.Dense(16, activation='relu')(hidden1)
#hidden3 = keras.layers.Dense(8, activation='relu')(hidden2)
output = keras.layers.Dense(1, activation='sigmoid')(hidden2)
model = keras.models.Model(inputs=inputs, outputs=output)
# summarize layers
print(model.summary())

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(Xs_train, y_train, batch_size=64, epochs=15, validation_split=0.2)
test_scores = model.evaluate(Xs_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

