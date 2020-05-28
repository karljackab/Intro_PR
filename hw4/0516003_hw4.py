import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import hw1_util as util

# ## Load data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# 550 data with 300 features
print(x_train.shape)

# It's a binary classification problem
print(np.unique(y_train))

# ## Question 1
# K-fold data partition: Implement the K-fold cross-validation function. Your function should take K as an argument and return a list of lists (len(list) should equal to K), which contains K elements. Each element is a list contains two parts, the first part contains the index of all training folds, e.g. Fold 2 to Fold 5 in split 1. The second part contains the index of validation fold, e.g. Fold 1 in  split 1

def cross_validation(x_train, y_train, k=5):
    base = []
    data_num = len(x_train)
    split_base = data_num//k
    cur_idx = 0
    grp_num = data_num % k

    idx_array = np.array(range(0, data_num))
    np.random.shuffle(idx_array)
    for _ in range(grp_num):
        next_idx = cur_idx+split_base+1
        base.append(idx_array[cur_idx:next_idx])
        cur_idx = next_idx
    for _ in range(grp_num, k):
        next_idx = cur_idx+split_base
        base.append(idx_array[cur_idx:next_idx])
        cur_idx = next_idx

    kfold_data = []
    for val_grp in range(k):
        kfold_data.append([[], []])
        for cur_grp in range(k):
            if cur_grp == val_grp:
                kfold_data[-1][1].extend(base[cur_grp])
            else:
                kfold_data[-1][0].extend(base[cur_grp])
        kfold_data[-1][1] = np.array(kfold_data[-1][1])
        kfold_data[-1][0] = np.array(kfold_data[-1][0])
    return kfold_data


kfold_data = cross_validation(x_train, y_train, k=10)
assert len(kfold_data) == 10  # should contain 10 fold of data
assert len(kfold_data[0]) == 2  # each element should contain train fold and validation fold
assert kfold_data[0][1].shape[0] == 55  # The number of data in each validation fold should equal to training data divieded by K

# ## Question 2
# Using sklearn.svm.SVC to train a classifier on the provided train set and conduct the grid search of “C”, “kernel” and “gamma” to find the best parameters by cross-validation.

def grid_search(x_train, y_train, kfold_idx, c_list, gamma_list, model=SVC):
    score_list = []
    k = len(kfold_idx)

    best_c, best_gamma, best_acc = None, None, 0
    for cur_c in c_list:
        score_list.append([])
        for cur_gamma in gamma_list:
            print(cur_c, cur_gamma)
            mean_train_acc, mean_val_acc = 0, 0
            for data in kfold_idx:
                clf = model(C=cur_c, kernel='rbf', gamma=cur_gamma)
                clf = clf.fit(x_train[data[0]], y_train[data[0]])
                mean_train_acc += clf.score(x_train[data[0]], y_train[data[0]])
                mean_val_acc += clf.score(x_train[data[1]], y_train[data[1]])
            mean_train_acc = mean_train_acc/k
            mean_val_acc = mean_val_acc/k
            score_list[-1].append(mean_val_acc)
            if best_acc < mean_val_acc:
                best_acc = mean_val_acc
                best_c, best_gamma = cur_c, cur_gamma

    score_list = np.array(score_list)
    return (best_c, best_gamma, best_acc), score_list

# prepare grid search value
c_list, gamma_list = [], []
base_c, base_gamma = 0.1, 0.0001
for i in range(1, 10):
    c_list.append(base_c*(2**i))
    gamma_list.append(base_gamma*(2**i))

best, score_list = grid_search(x_train, y_train, kfold_data, c_list, gamma_list)
c, gamma, acc = best
print(f'C: {c}, Gamma: {gamma}, Val acc: {acc}')


# ## Question 3
# Plot the grid search results of your SVM. The x, y represents the hyperparameters of “gamma” and “C”, respectively. And the color represents the average score of validation folds
# You reults should be look like the reference image ![image](https://miro.medium.com/max/1296/1*wGWTup9r4cVytB5MOnsjdQ.png)
fig, ax = plt.subplots()

im = ax.matshow(score_list, cmap='seismic')

for (i, j), value in np.ndenumerate(score_list):
    ax.text(j, i, f'{value:.2f}', ha='center', va='center')

ax.set_xticklabels(['']+gamma_list)
ax.set_yticklabels(['']+c_list)
plt.xlabel('Gamma Parameter')
plt.ylabel('C parameter')
plt.title('Hyperparameter Gridsearch')
fig.colorbar(im)
plt.show()

# ## Question 4
# Train your SVM model by the best parameters you found from question 2 on the whole training set and evaluate the performance on the test set. **You accuracy should over 0.85**

best_model = SVC(C=25.6, kernel='rbf', gamma=0.0004)
best_model = best_model.fit(x_train, y_train)

y_pred = best_model.predict(x_test)
print("Accuracy score: ", accuracy_score(y_pred, y_test))


# ## Question 5
# Compare the performance of each model you have implemented from HW1

# ### HW1
# ## SVM regression

train_df = pd.read_csv("../hw1/train_data.csv")
x_train = train_df['x_train'].to_numpy().reshape(-1, 1)
y_train = train_df['y_train'].to_numpy().reshape(-1)

test_df = pd.read_csv("../hw1/test_data.csv")
x_test = test_df['x_test'].to_numpy().reshape(-1, 1)
y_test = test_df['y_test'].to_numpy().reshape(-1)

# get CV data
kfold_data = cross_validation(x_train, y_train, k=10)

# prepare grid search value
c_list, gamma_list = [], []
base_c, base_gamma = 0.1, 0.0001
for i in range(1, 10):
    c_list.append(base_c*(2**i))
    gamma_list.append(base_gamma*(2**i))

# grid search
best, score_list = grid_search(x_train, y_train, kfold_data, c_list, gamma_list, model=SVR)
c, gamma, acc = best
print(f'C: {c}, Gamma: {gamma}, Val score: {acc}')

# plot matshow
fig, ax = plt.subplots()
im = ax.matshow(score_list, cmap='seismic')
for (i, j), value in np.ndenumerate(score_list):
    ax.text(j, i, f'{value:.2f}', ha='center', va='center')
ax.set_xticklabels(['']+gamma_list)
ax.set_yticklabels(['']+c_list)
plt.xlabel('Gamma Parameter')
plt.ylabel('C parameter')
plt.title('Hyperparameter Gridsearch')
fig.colorbar(im)
plt.show()

# use best model to do regression
best_model = SVR(C=c, kernel='rbf', gamma=gamma)
best_model = best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("Square error of SVM regresssion model: ", mean_squared_error(y_test, y_pred))

# ## Linear regression
linear_mode = util.LinearRegression()
_, test_loss_log = linear_mode.train()
print("Square error of Linear regression: ", test_loss_log[-1])
