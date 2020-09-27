"""
This is a junk file for me to test functions
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]])
print(arr)
print(arr.shape)
reshaped = arr.reshape(-1, 3)
print(reshaped)
print(reshaped.shape)
rereshaped = reshaped.reshape(arr.shape)
print(rereshaped)
print(rereshaped.shape)

"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
print(df)

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# print(x)
# print(x.shape)
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
pca3 = PCA(n_components=3)
principalComponents3 = pca3.fit_transform(x)
principalDf3 = pd.DataFrame(data = principalComponents3, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
print(principalDf)
print(principalDf3)
"""

