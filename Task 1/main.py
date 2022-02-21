import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from LinearRegressionGradientDescent import LinearRegressionGradientDescent
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 7)
pd.set_option('display.width', None)
data = pd.read_csv('datasets/car_purchase.csv')
data.drop("customer_id", axis=1, inplace=True)

# 1. DATA DESCRIPTION AND INFO
print("First 5 rows:")
print(data.head(), end='\n \n')
print("Last 5 rows:")
print(data.tail(), end='\n \n')
print("Info about the dataset:")
print(data.info(), end='\n \n')
print("General statistic information about attributes:")
print(data.describe())

# Some additional attributes statistics
print("\nMedian values: ")
print(data[['age', 'annual_salary', 'credit_card_debt', 'net_worth', 'max_purchase_amount']].median())
print("\nMean money statistics grouped by gender:")
print(
    data[['gender', 'annual_salary', 'credit_card_debt', 'net_worth', 'max_purchase_amount']].groupby('gender').mean())
print("\nPercentage of males and females:")
print(data["gender"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# DATA ENGINEERING
data_train = data
le = LabelEncoder()
data_train['gender'] = le.fit_transform(data_train['gender'])  # Gender string changed to 0 and 1

# Showing that gender and debt are not impacting max_purchase_amount
print('\nGender correlation with max_purchase_amount is ', data_train.max_purchase_amount.corr(data_train.gender),
      ' which is too low to take into account.')
print('Debt correlation with max_purchase_amount is ', data_train.max_purchase_amount.corr(data_train.credit_card_debt),
      ' which is too low to take into account.\n')

# SHOWING DATA ON PLOTS
# ----------------------------------------------------------------------------------------------------------------------
fig, plots = plt.subplots(2, 2, figsize=(14, 8), sharey='all')

plots[0, 0].scatter(data_train['annual_salary'], data_train['max_purchase_amount'], s=15, c='green', marker='o',
                    alpha=0.7, edgecolors='face', linewidths=2, label='person')
plots[0, 0].set_title('Car purchase power according to annual salary', fontsize="16")
plots[0, 0].set(xlabel='annual salary [US Dollars]')

plots[1, 0].scatter(data_train['credit_card_debt'], data_train['max_purchase_amount'], s=15, c='red', marker='o',
                    alpha=0.7, edgecolors='face', linewidths=2, label='person')
plots[1, 0].set_title('Car purchase power according to debt', fontsize="16")
plots[1, 0].set(xlabel='debt [US Dollars]')

plots[0, 1].scatter(data_train['net_worth'], data_train['max_purchase_amount'], s=15, c='blue', marker='o',
                    alpha=0.7, edgecolors='face', linewidths=2, label='person')
plots[0, 1].set_title('Car purchase power according to net worth', fontsize="16")
plots[0, 1].set(xlabel='net worth [US Dollars]')

plots[1, 1].scatter(data_train['age'], data_train['max_purchase_amount'], s=15, c='orange', marker='o',
                    alpha=0.7, edgecolors='face', linewidths=2, label='person')
plots[1, 1].set_title('Car purchase power according to age', fontsize="16")
plots[1, 1].set(xlabel='age [Years]')

for plot in plots.flat:
    plot.set(ylabel='max purchase amount')

for plot in plots.flat:
    plot.legend()
    plot.legend(loc='upper left')

plots[0, 1].ticklabel_format(style='plain', useOffset=False, axis='both')
plt.tight_layout()
plt.savefig('test.png', dpi=250)
# ----------------------------------------------------------------------------------------------------------------------

# ODABIR ATRIBUTA ZA ALGORITAM (VIDI SE IZ TABELA DA SE I credit_card_debt MOZE ZANEMARITI)
data_train.drop("credit_card_debt", axis=1, inplace=True)

# LINEAR REGRESSION IZ SCIKIT-A
# ----------------------------------------------------------------------------------------------------------------------
print('Performing SciKit Linear Regression...')

X_train, X_test, y_train, y_test = train_test_split(
    data_train[['annual_salary', 'age', 'net_worth']], data_train.max_purchase_amount, test_size=0.2
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

predictions = lr_model.predict(X_test)

print('Root mean squared error: ', mean_squared_error(np.array(y_test), predictions, squared=False))
print('Calculated linear regression function: y =', end=" ")
print(lr_model.coef_[0], '* annual_salary +', end=" ")
print(lr_model.coef_[1], '* age +', end=" ")
print(lr_model.coef_[2], '* net_worth +', end=" ")
print(lr_model.intercept_)
print('Model score [0..1]: ', lr_model.score(X_test, np.array(y_test)))

# MY LINEAR REGRESSION USING GRADIENT DESCENT
# ----------------------------------------------------------------------------------------------------------------------
print('\nPerforming my Linear Regression...')
lrgd = LinearRegressionGradientDescent()

# Scaling all data to a range from 0 to 1 for better performance

X_train, X_test, y_train, y_test = train_test_split(data_train[['annual_salary', 'age', 'net_worth']],
                                                    data_train.max_purchase_amount, test_size=0.3)

X = X_train.copy(deep=True)
y = y_train / y_train.max()

X[['annual_salary']] = X_train[['annual_salary']] / X_train['annual_salary'].max()
X[['net_worth']] = X_train[['net_worth']] / X_train['net_worth'].max()
X[['age']] = X_train[['age']] / X_train['age'].max()

X_test[['annual_salary']] = X_test[['annual_salary']] / X_train['annual_salary'].max()
X_test[['net_worth']] = X_test[['net_worth']] / X_train['net_worth'].max()
X_test[['age']] = X_test[['age']] / X_train['age'].max()


# Performing linear regression and adjusted learning rates
lrgd.fit(X, y)
learning_rates = np.array([[0.9], [0.9], [0.9], [0.9]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 2000)
myPredictions = lrgd.predict(X_test) * y_train.max()

# Plotting MSE history
fig2 = plt.figure('MSE History')
for i in range(len(mse_history)):
    mse_history[i] = mse_history[i] * y_train.max()
plt.plot(np.arange(0, len(mse_history), 1), mse_history)

# y/max(y) = w1 * x1 / max(x1) + w2 * x2 / max(x2) + w3
# y = x1 * w1 / max(x1) * max(y) + w2 * x2 / max(x2) * max(y) + w3 * max(y)

print('Root mean squared error: ', mean_squared_error(np.array(y_test), myPredictions, squared=False))
print('Calculated linear regression function: y =', end=" ")
print(res_coeff[1][0] * y_train.max() / X_train['annual_salary'].max(), '* annual_salary +', end=" ")
print(res_coeff[2][0] * y_train.max() / X_train['age'].max(), '* age +', end=" ")
print(res_coeff[3][0] * y_train.max() / X_train['net_worth'].max(), '* net_worth +', end=" ")
print(res_coeff[0][0] * y_train.max())
print('Original calculated linear regression function (everything scaled to [0..1]): y =', end=" ")
print(res_coeff[1][0], '* annual_salary +', end=" ")
print(res_coeff[2][0], '* age +', end=" ")
print(res_coeff[3][0], '* net_worth +', end=" ")
print(res_coeff[0][0])
print('Model score [0..1]: ', r2_score(np.array(y_test), myPredictions))

plt.tight_layout()
plt.show()
