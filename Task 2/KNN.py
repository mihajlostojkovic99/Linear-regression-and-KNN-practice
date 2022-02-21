import pandas as pd

import numpy as np


# returns the most frequent value from a list
def most_frequent(ret_list):
    counter = 0
    num = ret_list[0]
    for i in ret_list:
        curr_frequency = ret_list.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num


class KNN:
    def __init__(self):
        self.num_of_neighbors = None
        self.data = None

    # setting the number of neighbors to be used
    def KNeighborsClassifier(self, num_of_neighbors):
        self.num_of_neighbors = num_of_neighbors

    # data is a whole DataFrame, features is a DataFrame without target
    def fit(self, data):

        self.data = data.copy(deep=True)

        # inserting a new column 'distance' for calculating the distance between this point and a new point
        self.data.insert(0, 'distance', np.zeros((len(self.data), 1)))
        self.data = self.data.to_numpy()

    def predict(self, new_features):
        return_list = []
        new_features = new_features.to_numpy()
        new_rows = len(new_features)
        rows = len(self.data)
        cols = len(self.data[0])
        for new_row in range(new_rows):
            # For every new feature it iterates through available data and calculates euclidean distance
            # between the new feature and every node from data. Then sorts the data in ascending order and
            # cuts only the first k nodes. Then it finds the most common target value for those nodes and
            # appends it to the return list. Proceeds to do so with the rest of new features.
            for row in range(rows):
                self.data[row][0] = 0

                # calculating euclidean distance
                for col in range(1, cols - 1):
                    self.data[row][0] += pow(self.data[row][col] - new_features[new_row][col - 1], 2)
                self.data[row][0] = np.sqrt(self.data[row][0])

            # sorting and cutting first k closest neighbors
            return_value = self.data[self.data[:, 0].argsort()]
            return_value = return_value[0:self.num_of_neighbors, -1]

            # finding the most frequent target value and appending it to the return list
            return_value = most_frequent(return_value.tolist())
            return_list.append(return_value)
        return pd.Series(return_list)
