
import numpy as np
import pandas as pd

# reproducible results
np.random.seed(42)

####################
### create data  ###
####################

data = pd.DataFrame(np.random.normal(2, 1, 200), # n = 100
                    columns = ["X1"])
np.random.randint(1, 10)
for i in range(2, 1001): # p = 1000
    # define an input (Xs)
    X = np.random.normal(np.random.randint(3, 10), np.random.randint(1, 3), 200)
    name = "X{}".format(i)
    data[name] = X

# create underlying truth with 20 varibales informative
# define betas coeficients
random_betas = np.random.randint(5, 20, 20)
# apply coefs to inputs
data_betas = data.loc[:, :"X20"] * random_betas
# define output based on the 20 varibales and their artificial coefficents
data["Y"] = data_betas.sum(axis=1)



##################################################################
### define specific dataset of p=20 with 20 important features ###
##################################################################
print("p = 20 with: 20 features are important")
X = data.loc[:, :"X20"].values
Y = data["Y"]

### Training and testing for each model ###

# apply a linear regression, a linear regression with univaritate feature selection,
# a ridge regression and a lasso
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import linear_model
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt

LinReg = linear_model.LinearRegression()
Ridge = linear_model.RidgeCV()
Lasso = linear_model.LassoCV()

# LINEAR REG
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    LinReg.fit(X[train_index, :], Y[train_index])
    acc.append(LinReg.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies = pd.DataFrame(acc,
                    columns = ["LinReg"])

# LINEAR REG + UNIVARIATE FEATURE SELECTION
selector = SelectPercentile(f_classif, percentile=20)
selector.fit(X, Y)
X_new = selector.transform(X)
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    LinReg.fit(X_new[train_index, :], Y[train_index])
    acc.append(LinReg.score(X_new[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["LinReg_feat_select"] = acc

# RIDGE
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    Ridge.fit(X[train_index, :], Y[train_index])
    acc.append(Ridge.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["Ridge"] = acc

# LASSO
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    Lasso.fit(X[train_index, :], Y[train_index])
    acc.append(Lasso.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["Lasso"] = acc

### Plot the resulting accuracies ###
acc_stack = accuracies.stack().reset_index()
del acc_stack["level_0"]
acc_stack = acc_stack.rename(columns={0: 'acc', 'level_1':'param'})
ax = sns.boxplot(x="param", y="acc", data=acc_stack)
ax.get_yaxis().set_visible(False)
plt.ylim(0.99993, 1)
plt.xlabel("")
plt.title("p = 20, 20 features associated with response")
plt.tight_layout()
plt.show()



##################################################################
### define specific dataset of p=50 with 20 important features ###
##################################################################
print("p = 50 with: 20 features are important")
X = data.loc[:, :"X50"].values
Y = data["Y"]

### Training and testing for each model ###

# LINEAR REG
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    LinReg.fit(X[train_index, :], Y[train_index])
    acc.append(LinReg.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies = pd.DataFrame(acc,
                    columns = ["LinReg"])

# LINEAR REG + UNIVARIATE FEATURE SELECTION
selector = SelectPercentile(f_classif, percentile=20)
selector.fit(X, Y)
X_new = selector.transform(X)
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    LinReg.fit(X_new[train_index, :], Y[train_index])
    acc.append(LinReg.score(X_new[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["LinReg_feat_select"] = acc

# RIDGE
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    Ridge.fit(X[train_index, :], Y[train_index])
    acc.append(Ridge.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["Ridge"] = acc

# LASSO
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    Lasso.fit(X[train_index, :], Y[train_index])
    acc.append(Lasso.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["Lasso"] = acc

### Plot the resulting accuracies ###
acc_stack = accuracies.stack().reset_index()
del acc_stack["level_0"]
acc_stack = acc_stack.rename(columns={0: 'acc', 'level_1':'param'})
ax = sns.boxplot(x="param", y="acc", data=acc_stack)
ax.get_yaxis().set_visible(False)
plt.ylim(0.9999, 1)
plt.xlabel("")
plt.title("p = 50, 20 features associated with response")
plt.tight_layout()
plt.show()

####################################################################
### define specific dataset of p=1000 with 20 important features ###
####################################################################
print("p = 1000 with: 20 features are important")
X = data.loc[:, :"X1000"].values
Y = data["Y"]

### Training and testing for each model ###

# LINEAR REGRESSION
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    LinReg.fit(X[train_index, :], Y[train_index])
    acc.append(LinReg.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies = pd.DataFrame(acc,
                    columns = ["LinReg"])

# LINEAR REG + UNIVARIATE FEATURE SELECTION
selector = SelectPercentile(f_classif, percentile=20)
selector.fit(X, Y)
X_new = selector.transform(X)
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    LinReg.fit(X_new[train_index, :], Y[train_index])
    acc.append(LinReg.score(X_new[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["LinReg_feat_select"] = acc

# RIDGE
kf = KFold(n_splits=5)
acc = []
for train_index, test_index in kf.split(X):
    Ridge.fit(X[train_index, :], Y[train_index])
    acc.append(Ridge.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["Ridge"] = acc
kf = KFold(n_splits=5)
acc = []

# LASSO
for train_index, test_index in kf.split(X):
    Lasso.fit(X[train_index, :], Y[train_index])
    acc.append(Lasso.score(X[test_index, :], Y[test_index]))
print(np.mean(acc))
accuracies["Lasso"] = acc

### Plot the resulting accuracies ###
acc_stack = accuracies.stack().reset_index()
del acc_stack["level_0"]
acc_stack = acc_stack.rename(columns={0: 'acc', 'level_1':'param'})
ax = sns.boxplot(x="param", y="acc", data=acc_stack)
ax.get_yaxis().set_visible(False)
plt.ylim(-0.1, 1)
plt.xlabel("")
plt.title("p = 1000, 20 features associated with response")
plt.tight_layout()
plt.show()
