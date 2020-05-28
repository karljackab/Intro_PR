import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from collections import Counter
import matplotlib.pyplot as plt
import random

data = load_breast_cancer()

x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

# Question 1


def gini(sequence):
    categories = set(sequence)
    # categories = [0, 1]
    res, num = 1., len(sequence)
    if num == 0:
        return 0
    for category in categories:
        res -= (len(sequence[sequence == category])/num) ** 2
    return res


def entropy(sequence):
    categories = set(sequence)
    # categories = [0, 1]
    res, num = 0, len(sequence)
    if num == 0:
        return 0
    for category in categories:
        prob = len(sequence[sequence == category])/num
        res += -prob*np.log2(prob)
    return res

data = np.array([1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2])
print("Gini of data is ", gini(data))
print("Entropy of data is ", entropy(data))

# Question 2


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None, max_features=None):
        if criterion != 'gini' and criterion != 'entropy':
            raise Exception('DecisionTree key error: criterion myst be "gini" \
                            or "entropy"')

        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = []  # each element would be
                        # [split_feature, split_value, larger_goto_idx, 
                        #  others_goto_idx, category]
        self.feature_importance = dict()
        self.max_features = max_features
        if self.max_features is not None:
            self.max_features = int(self.max_features)

    def check_terminate(self, y):
        first = None
        for i in y:
            if first is None:
                first = i
            elif first != i:
                return False
        return True

    def fit(self, x, y):
        if type(y) == pd.core.frame.DataFrame:
            y = y['0']

        self.tree.append([])
        for name in x.keys():
            self.feature_importance[name] = 0.0
        self.split(x, y, 1, 0)

    def pickup_feature_names(self, keys):
        if self.max_features is None or len(keys) < self.max_features:
            return keys
        else:
            return random.sample(list(keys.to_numpy()), self.max_features)

    def split(self, x, y, depth, tree_idx):
        if (self.max_depth is not None and depth > self.max_depth) or self.check_terminate(y.to_numpy()):
            cnt = Counter(y.to_numpy())
            self.tree[tree_idx] = \
                [None, None, None, None, cnt.most_common(1)[0][0]]
            return

        feature_names = self.pickup_feature_names(x.keys())

        best_split_feature = None
        best_split_value = None
        best_criteria = None

        for feature_name in feature_names:
            feature = x[feature_name].sort_values().to_numpy()
            for idx in range(1, len(feature)):
                split_value = (feature[idx-1]+feature[idx])/2
                larger_y, others_y = y[x[feature_name] > split_value], y[x[feature_name] <= split_value]
                larger_n, others_n = len(larger_y), len(others_y)
                if self.criterion == 'gini':
                    new_gini = larger_n*gini(larger_y.to_numpy()) + others_n*gini(others_y.to_numpy())
                    new_gini /= (larger_n+others_n)
                    if best_criteria is None or new_gini < best_criteria:
                        best_criteria = new_gini
                        best_split_feature = feature_name
                        best_split_value = split_value
                elif self.criterion == 'entropy':
                    after_entropy = larger_n*entropy(larger_y.to_numpy()) + others_n*entropy(others_y.to_numpy())
                    after_entropy /= (larger_n+others_n)
                    if best_criteria is None or after_entropy < best_criteria:
                        best_criteria = after_entropy
                        best_split_feature = feature_name
                        best_split_value = split_value

        larger_y, others_y = y[x[best_split_feature] > best_split_value], y[x[best_split_feature] <= best_split_value]
        larger_n, others_n = len(larger_y), len(others_y)
        if self.criterion == 'gini':
            init_criteria = gini(y.to_numpy())
            new_gini = larger_n*gini(larger_y.to_numpy()) + others_n*gini(others_y.to_numpy())
            new_gini /= (larger_n+others_n)
            self.feature_importance[best_split_feature] += init_criteria-new_gini
        elif self.criterion == 'entropy':
            init_criteria = entropy(y.to_numpy())
            after_entropy = larger_n*entropy(larger_y.to_numpy()) + others_n*entropy(others_y.to_numpy())
            after_entropy /= (larger_n+others_n)
            self.feature_importance[best_split_feature] += init_criteria-after_entropy

        greater_idx = len(self.tree)
        others_idx = greater_idx+1
        self.tree[tree_idx] = [best_split_feature, best_split_value, greater_idx, others_idx, None]
        self.tree.append([])
        self.tree.append([])
        self.split(x[x[best_split_feature] > best_split_value], larger_y, depth+1, greater_idx)
        self.split(x[x[best_split_feature] <= best_split_value], others_y, depth+1, others_idx)

    def print_tree(self, idx=0, spacing=''):
        """World's most elegant tree printing function."""

        # Base case: we've reached a leaf
        if self.tree[idx][4] is not None:
            print(f'{spacing} Predict {self.tree[idx][4]}')
            return

        # Print the question at this node
        print(spacing + f'{self.tree[idx][0]}>{self.tree[idx][1]}')

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.print_tree(self.tree[idx][2], spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.print_tree(self.tree[idx][3], spacing + "  ")

    def predict_single(self, x):
        # self.tree: [split_feature, split_value, larger_goto_idx, others_goto_idx, category]
        cur_tree_idx = 0
        while self.tree[cur_tree_idx][4] is None:
            if x[self.tree[cur_tree_idx][0]] > self.tree[cur_tree_idx][1]:
                cur_tree_idx = self.tree[cur_tree_idx][2]
            else:
                cur_tree_idx = self.tree[cur_tree_idx][3]
        return self.tree[cur_tree_idx][4]

    def predict(self, x):
        pred_y = []
        for _, row in x.iterrows():
            pred_y.append(self.predict_single(row))
        return pred_y

    def score(self, x, y, return_pred_y=False):
        pred_y = self.predict(x)
        tot_n, acc_n = len(pred_y), 0
        y = y.to_numpy()
        for i in range(tot_n):
            if y[i] == pred_y[i]:
                acc_n += 1
        if return_pred_y:
            return acc_n/tot_n, pred_y
        else:
            return acc_n/tot_n

    def gen_importance_feature(self, img_name='importance_feature.png'):
        import_feat_dict = dict()
        tot_cnt = 0
        for node in self.tree:
            if node[0] is None:
                continue
            tot_cnt += 1
            if node[0] not in import_feat_dict:
                import_feat_dict[node[0]] = 1
            else:
                import_feat_dict[node[0]] += 1
        
        import_feat_list = []
        for key in import_feat_dict:
            import_feat_list.append([key, import_feat_dict[key]/tot_cnt])
        import_feat_list = sorted(import_feat_list, key=lambda x: x[1], reverse=True)
        
        labels, values = [], []
        for pair in import_feat_list:
            labels.append(pair[0])
            values.append(pair[1])

        plt.bar(labels, values)
        plt.xlabel('Feature name')
        plt.ylabel('Importance ratio')
        plt.xticks(rotation=15)
        plt.savefig(img_name)

# Question 2.1
print('=============================')

clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth3.fit(x_train, y_train)
accuracy = clf_depth3.score(x_test, y_test)
print(f'gini accuracy depth 3: {accuracy}')

clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
clf_depth10.fit(x_train, y_train)
accuracy = clf_depth10.score(x_test, y_test)
print(f'gini accuracy depth 10: {accuracy}')

# Question 2.2
print('=============================')

clf_gini = DecisionTree(criterion='gini', max_depth=3)
clf_gini.fit(x_train, y_train)
accuracy = clf_gini.score(x_test, y_test)
print(f'gini accuracy depth 3: {accuracy}')

clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
clf_entropy.fit(x_train, y_train)
accuracy = clf_entropy.score(x_test, y_test)
print(f'entropy accuracy depth 3: {accuracy}')

# Question 3
clf_depth10.gen_importance_feature('depth10_feature.png')

# Question 4


class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        self.n_tree = n_estimators
        self.boostrap = boostrap
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(DecisionTree(criterion=criterion, max_depth=max_depth, max_features=max_features))

    def fit(self, x, y):
        base_x = x
        base_x['y'] = y
        for tree_idx in range(self.n_tree):
            if self.boostrap:
                new_x = base_x.sample(n=len(base_x), replace=True)
            else:
                new_x = base_x
            new_y = new_x['y']
            del new_x['y']

            self.trees[tree_idx].fit(new_x, new_y)

    def predict(self, x):
        res = None
        for i in range(self.n_tree):
            pred = self.trees[i].predict(x)
            pred = np.array(pred).reshape(-1, 1)
            if res is None:
                res = pred
            else:
                res = np.concatenate((res, pred), axis=1)
        return [Counter(res[i]).most_common(1)[0][0] for i in range(len(x))]

    def score(self, x, y, return_pred_y=False):
        pred_y = self.predict(x)
        tot_n, acc_n = len(pred_y), 0
        y = y.to_numpy()
        for i in range(tot_n):
            if y[i] == pred_y[i]:
                acc_n += 1
        if return_pred_y:
            return acc_n/tot_n, pred_y
        else:
            return acc_n/tot_n

# Question 4.1
print('=============================')

clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
clf_10tree.fit(x_train, y_train)
accuracy = clf_10tree.score(x_test, y_test)
print(f'clf_10tree accuracy: {accuracy}')

clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(x_train.shape[1]))
clf_100tree.fit(x_train, y_train)
accuracy = clf_100tree.score(x_test, y_test)
print(f'clf_100tree accuracy: {accuracy}')

# Question 4.2
print('=============================')

clf_random_features = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
clf_random_features.fit(x_train, y_train)
accuracy = clf_random_features.score(x_test, y_test)
print(f'clf_random_features accuracy: {accuracy}')

clf_all_features = RandomForest(n_estimators=10, max_features=x_train.shape[1])
clf_all_features.fit(x_train, y_train)
accuracy = clf_all_features.score(x_test, y_test)
print(f'clf_all_features accuracy: {accuracy}')
