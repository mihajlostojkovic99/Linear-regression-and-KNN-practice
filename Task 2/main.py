import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from KNN import KNN

pd.set_option('display.max_columns', 7)
pd.set_option('display.width', None)
data = pd.read_csv('datasets/car_state.csv')

# DATA DESCRIPTION AND INFO
print("First 5 rows:")
print(data.head(), end='\n \n')
print("Last 5 rows:")
print(data.tail(), end='\n \n')
print("Info about the dataset:")
print(data.info(), end='\n \n')
print("General statistic information about attributes:")
print(data.describe())

for col in data:
    print('Column', col, 'unique values: ', data[col].unique())

# SHOWING DATA ON PLOTS, NO POINT
# ----------------------------------------------------------------------------------------------------------------------
# fig1 = plt.figure('Buying price')
# data['buying_price'].value_counts().plot(kind='bar')
# plt.xlabel('Buying Price', fontsize=13)
# plt.ylabel('Amount in the data', fontsize=13)
# plt.title('Number of cars in each price point', fontsize=16)
#
# fig2 = plt.figure('Maintenance')
# data['maintenance'].value_counts().plot(kind='bar')
# plt.xlabel('Maintenance bracket', fontsize=13)
# plt.ylabel('Amount in the data', fontsize=13)
# plt.title('Number of cars in each maintenance bracket', fontsize=16)
#
# fig3 = plt.figure('Doors')
# data['doors'].value_counts().plot(kind='bar')
# plt.xlabel('Number of doors', fontsize=13)
# plt.ylabel('Amount in the data', fontsize=13)
# plt.title('Number of cars with different door configurations', fontsize=16)
#
# fig4 = plt.figure('Seats')
# data['seats'].value_counts().plot(kind='bar')
# plt.xlabel('Number of seats', fontsize=13)
# plt.ylabel('Amount in the data', fontsize=13)
# plt.title('Number of cars with different seating configurations', fontsize=16)
#
# fig5 = plt.figure('Trunk size')
# data['trunk_size'].value_counts().plot(kind='bar')
# plt.xlabel('Trunk size', fontsize=13)
# plt.ylabel('Amount in the data', fontsize=13)
# plt.title('Number of cars per different trunk size bracket', fontsize=16)
#
# fig6 = plt.figure('Safety')
# data['safety'].value_counts().plot(kind='bar')
# plt.xlabel('Safety bracket', fontsize=13)
# plt.ylabel('Amount in the data', fontsize=13)
# plt.title('Number of cars in each safety bracket', fontsize=16)

# ----------------------------------------------------------------------------------------------------------------------

# DATA ENGINEERING
# ----------------------------------------------------------------------------------------------------------------------
le = LabelEncoder()
data["status"] = le.fit_transform(data["status"])

data["buying_price"] = data["buying_price"].map({"low": 1, "medium": 2, "high": 3, "very high": 4})
data["maintenance"] = data["maintenance"].map({"low": 1, "medium": 2, "high": 3, "very high": 4})
data["doors"] = data["doors"].map({"2": 2, "3": 3, "4": 4, "5 or more": 5})
data["seats"] = data["seats"].map({"2": 2, "4": 4, "5 or more": 5})
data["trunk_size"] = data["trunk_size"].map({"small": 1, "medium": 2, "big": 3})
data["safety"] = data["safety"].map({"low": 1, "medium": 2, "high": 3})
data_train = data
# ----------------------------------------------------------------------------------------------------------------------


print("\nStatistics with input converted to numerical values:")
print(data.describe())
print("Buying price (low-1, medium-2, high-3, very high-4)")
print("Maintenance (low-1, medium-2, high-3, very high-4)")
print("Doors (5 or more-5)")
print("Seats (5 or more-5)")
print("Trunk size (small-1, medium-2, big-3)")
print("Safety (low-1, medium-2, high-3)\n")

# Median statistics
print("\nMedian values: ")
print(data[['buying_price', 'maintenance', 'doors', 'seats', 'trunk_size', 'safety']].median())

# print('\nBuying price correlation with Max purchase amount is ', data_train.status.corr(data_train.buying_price),
#       end=' ')
# print('\nMaintenance correlation with Max purchase amount is ', data_train.status.corr(data_train.maintenance),
#       end=' ')
# print('\nDoors correlation with Max purchase amount is ', data_train.status.corr(data_train.doors),
#       end=' ')
# print('\nSeats correlation with Max purchase amount is ', data_train.status.corr(data_train.seats),
#       end=' ')
# print('\nTrunk size correlation with Max purchase amount is ', data_train.status.corr(data_train.trunk_size),
#       end=' ')
# print('\nSafety correlation with Max purchase amount is ', data_train.status.corr(data_train.safety),
#       end=' ')

# KNN FROM SCIKIT
# ----------------------------------------------------------------------------------------------------------------------
print('\n\nPerforming SciKit KNN...')

X_train, X_test, y_train, y_test = train_test_split(
    data_train[['buying_price', 'maintenance', 'doors', 'seats', 'trunk_size', 'safety']], data_train.status,
    test_size=0.2)

# num_neighbors = 5
num_neighbors = int(np.sqrt(len(data.index)))
if num_neighbors % 2 == 0:
    num_neighbors = num_neighbors - 1

acc_list = []
neigh_list = []
for i in range(1, num_neighbors, 2):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, y_train)
    acc_list.append(knn_model.score(X_test, y_test))
    neigh_list.append(i)

index = 0
for x in range(len(acc_list)):
    if acc_list[x] == max(acc_list):
        index = x
        break
print('Best achieved accuracy is: ', acc_list[index] * 100, '%')
print('Best number of neighbors is:', neigh_list[index])
print('Chosen number of neighbors (odd sqrt(n)) is: ', num_neighbors)

fig_acc = plt.figure('Accuracy per num of neighbors')
plt.plot(neigh_list, acc_list, color='red')

knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
knn_model.fit(X_train, y_train)

predictions = knn_model.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(y_test, predictions) * 100, '%')
target_names = le.inverse_transform([0, 1, 2, 3])
print('Classification report: \n', metrics.classification_report(y_test, predictions, target_names=target_names))

# MY IMPLEMENTATION OF KNN
# ----------------------------------------------------------------------------------------------------------------------

print('\nPerforming my KNN...')
knn_my_model = KNN()
knn_my_model.KNeighborsClassifier(num_neighbors)
data_for_model = pd.concat([X_train, y_train], axis=1)
knn_my_model.fit(data_for_model)
my_predictions = knn_my_model.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(y_test, my_predictions) * 100, '%')
print('Classification report: \n', metrics.classification_report(y_test, my_predictions, target_names=target_names))

plt.tight_layout()
plt.show()
