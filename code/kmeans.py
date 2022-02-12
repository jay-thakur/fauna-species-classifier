'''
Name :- Jay Prakash Thakur
Id :- 1001861778
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange

import warnings
warnings.filterwarnings('ignore')

# np.random.seed(42)

iris_data = pd.read_csv('iris.data')
# print(iris_data.head())

# Column names
column_name = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Class']

iris_data.columns = column_name
# print(iris_data.head())

# print(iris_data.shape)
# print(iris_data.describe())
# print(iris_data.info())
# print(iris_data['Class'].value_counts())
unique_class = iris_data['Class'].unique()
# print("Unique Classes : ", unique_class)


def categorical_to_numerical(c):

  """ this method is to convert categorical value to numerical value
  :type c: string

  :rtype: a number denoting class
    """
  if c.lower() == 'iris-setosa':
    return 1
  elif c.lower() == 'iris-versicolor':
    return 2
  elif c.lower() == 'iris-virginica':
    return 3

iris_data['Class'] = iris_data['Class'].map(lambda c: categorical_to_numerical(c))
# print(iris_data.Class.value_counts())

print("iris_data shape :: ", iris_data.shape)

feature = iris_data.drop(['Class'], axis=1) # let's call X as feature
cluster = iris_data['Class'] # let's call y as cluster

print("feature Shape :", feature.shape)
print("cluster Shape :", cluster.shape)



class KMeansClustering():

    """ 
    This class containts all the required methods.	
    """
    
    def __init__(self, dataframe, K) -> None:
    
        """ this is init method
        :type self:
        :param self:
    
        :type dataframe:
        :param dataframe:
    
        :type K:
        :param K:
    
        :raises:
    
        :rtype: None
        """    
        self.centroids = None
        self.K = K
        self.dataframe = dataframe
        self.initialize_centroids(self.dataframe)

    def initialize_centroids(self, dataframe):
    
        """ this method is used to initialize random centroids
        :type self:
        :param self:
    
        :type dataframe:
        :param dataframe:
    
        :raises:
    
        :rtype: None
        """    
        self.centroids = {}

        random_index = np.random.choice(len(dataframe), self.K, replace=False)
        i = 0
        for k in range(1, self.K+1):
            idx = random_index[i]
            random_centroid = dataframe.iloc[idx]
            i += 1
            self.centroids[k] = random_centroid.tolist()
        print("Initialized Centroids are  :: ", self.centroids)

    # calculate euclidean distance
    def calculate_euclidean_distance(self, X, centroid_points):
    
        """ This is used to calculate euclidean distance when data & centroids are given
        :type self:
        :param self:
    
        :type X:
        :param X:
    
        :type centroid_points:
        :param centroid_points:
    
        :raises:
    
        :rtype: a number
        """    
        
        total_distance = 0
        total_distance = np.sqrt((X['Sepal_length'] - centroid_points[0]) ** 2 + (X['Sepal_width'] - centroid_points[1]) ** 2
                       + (X['Petal_length'] - centroid_points[2]) ** 2 + (X['Petal_width'] - centroid_points[3]) ** 2)
        # print("total_distance :: ", total_distance)
        return np.square(total_distance)

    def get_euclidean_norm_distance(self, point_array_1, point_array_2):
    
        """ this method is used to calcualte euclidean normal distance when two point list are given
        :type self:
        :param self:
    
        :type point_array_1:
        :param point_array_1:
    
        :type point_array_2:
        :param point_array_2:
    
        :raises:
    
        :rtype: a number
        """    
        return np.linalg.norm(point_array_1 - point_array_2)
    
    def assign_centroid(self, X):
    
        """ this method is used to assign centroids
        :type self:
        :param self:
    
        :type X:
        :param X:
    
        :raises:
    
        :rtype: update dataframe
        """    
        for centroid in self.centroids.keys():
            # print("cluster :: ", cluster)
            # print(self.centroids[cluster])
            X['Distance from {}'.format(centroid)] = self.calculate_euclidean_distance(X, self.centroids[centroid])
        # print(X.columns)
        centroid_distance_column = ['Distance from {}'.format(i) for i in self.centroids.keys()]
        # print(X.loc[:, centroid_distance_column].idxmin(axis=1))
        X['Closest Centroid'] = X.loc[:, centroid_distance_column].idxmin(axis=1)
        X['Closest Centroid'] = X['Closest Centroid'].map(lambda x: int(x.lstrip('Distance from ')))
        return X

    def calculate_new_centroids(self, X):
    
        """ this method is used to udpate centroids
        :type self:
        :param self:
    
        :type X:
        :param X:
    
        :raises:
    
        :rtype: new centroids
        """    
        # update centroids
        for i in range(1, self.K + 1):
            self.centroids[i][0] = np.mean(X[X['Closest Centroid'] == i]['Sepal_length'])
            self.centroids[i][1] = np.mean(X[X['Closest Centroid'] == i]['Sepal_width'])
            self.centroids[i][2] = np.mean(X[X['Closest Centroid'] == i]['Petal_length'])
            self.centroids[i][3] = np.mean(X[X['Closest Centroid'] == i]['Petal_width'])
        return self.centroids

    def fit_predict(self, X):

        """ this method is used to assign centroid & udpate 
        :type self:
        :param self:

        :type X:
        :param X:

        :raises:

        :rtype: new centroids
        """
        X = self.assign_centroid(X)

        new_centroids = self.calculate_new_centroids(X)
        # print("new_centroids :: ", new_centroids)
        return self.assign_centroid(X)

    def accuracy(self, actual, predicted):
    
        """ this method is used to calculate accuracy
        :type self:
        :param self:
    
        :type actual:
        :param actual:
    
        :type predicted:
        :param predicted:
    
        :raises:
    
        :rtype: accuracy
        """    
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct = correct + 1
        return correct / float(len(actual)) * 100.0

    def inertia(self):
    
        """ this method is used to calculate inertia
        :type self:
        :param self:
    
        :raises:
    
        :rtype: sse, a number
        """    
        sse = 0
        for centroid, centroid_points in self.centroids.items():
            for points in centroid_points:
                euclidean_norm_distance = self.get_euclidean_norm_distance(points, centroid)
                euclidean_norm_distance_squared = euclidean_norm_distance ** 2
                sse += euclidean_norm_distance_squared
        return sse

K = 3
model = KMeansClustering(feature, K)
# kmeans = model.fit_predict(feature)

result_centroid = {}
epochs = 5
for epoch in range(epochs):
    X = model.fit_predict(feature)
    print("Epoch %s, New Centroid ::\n%s\n" %(epoch, model.centroids))
    # closest_centroids = X['Closest Centroid'].copy(deep=True)
    # if closest_centroids.equals(X['Closest Centroid']):
    #     break
# centroids = model.fit(feature)
# print(centroids)
print("\n ************************ \n")
print("Final Centroids : ", model.centroids)

print("\n ************************ \n")

print("Datapoints belongs to cluster 1:\n")
print(X[X["Closest Centroid"]==1])

print("\n ************************ \n")

print ("\nCentroid 2:", model.centroids[1])
print("Datapoints belongs to cluster 2:\n")

print("\n ************************ \n")

print(X[X["Closest Centroid"]==2])
print ("\nCentroid 3:", model.centroids[2])

print("\n ************************ \n")

print("Datapoints belongs to cluster 3:\n")
print(X[X["Closest Centroid"]==3])

print("\n ************************ \n")
y_pred = X["Closest Centroid"]
print("\nACCURACY ::", model.accuracy(cluster, y_pred))


print("\n ************************ \n")
print("Elbow method to find optimal K")
print("\n ************************ \n")
K = range(1, 11)
inertia = [] 

for k in K:
    model = KMeansClustering(feature, k)

    epochs = 20
    
    for epoch in range(epochs):
        X = model.fit_predict(feature)
    
    sse = model.inertia()
    inertia.append(sse)
