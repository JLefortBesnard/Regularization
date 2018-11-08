import numpy as np
import pandas as pd


np.random.seed(42)

data = pd.DataFrame(np.random.normal(2, 1, 200), # n = 100
                    columns = ["X1"])
                    
np.random.randint(1, 10)
for i in range(2, 1001): # p = 1000
    X = np.random.normal(np.random.randint(3, 10), np.random.randint(1, 3), 200)
    name = "X{}".format(i)
    data[name] = X

random_betas = np.random.randint(5, 20, 20)  
data_betas = data.loc[:, :"X20"] * random_betas
Y_ = data_betas.sum(axis=1)  # create underlying truth with all variables informative
data["Y"] = Y_ # all variables are informative


# apply the ridge and lasso and return accuracy
from sklearn import linear_model
from sklearn import cross_validation
import seaborn as sns
import matplotlib.pyplot as plt
LinReg = linear_model.LinearRegression()
Ridge = linear_model.RidgeCV()
Lasso = linear_model.LassoCV()



print("p = 20, 20 features are important")

X = data.loc[:, :"X20"].values
Y = data["Y"]
# cross validation:
kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    LinReg.fit(X[train_index, :], Y[train_index])
    acc.append(LinReg.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies = pd.DataFrame(acc,
                    columns = ["LinReg"])

kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    Ridge.fit(X[train_index, :], Y[train_index])
    acc.append(Ridge.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies["Ridge"] = acc

kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    Lasso.fit(X[train_index, :], Y[train_index])
    acc.append(Lasso.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies["Lasso"] = acc
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



print("p = 50, 20 features are important")
X = data.loc[:, :"X50"].values
Y = data["Y"]
# cross validation:
kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    LinReg.fit(X[train_index, :], Y[train_index])
    acc.append(LinReg.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies = pd.DataFrame(acc,
                    columns = ["LinReg"])
                    
kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    Ridge.fit(X[train_index, :], Y[train_index])
    acc.append(Ridge.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies["Ridge"] = acc
kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    Lasso.fit(X[train_index, :], Y[train_index])
    acc.append(Lasso.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies["Lasso"] = acc
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


print("p = 1000, 20 features are important")
X = data.loc[:, :"X1000"].values
Y = data["Y"]
# cross validation:
kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    LinReg.fit(X[train_index, :], Y[train_index])
    acc.append(LinReg.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies = pd.DataFrame(acc,
                    columns = ["LinReg"])

kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    Ridge.fit(X[train_index, :], Y[train_index])
    acc.append(Ridge.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies["Ridge"] = acc
kf = cross_validation.KFold(len(Y), n_folds=5)
acc = []
for train_index, test_index in kf:
    Lasso.fit(X[train_index, :], Y[train_index])
    acc.append(Lasso.score(X[test_index, :], Y[test_index]))
print np.mean(acc)
accuracies["Lasso"] = acc
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









        
        