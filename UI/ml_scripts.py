#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:48:21 2019

@author: naoto
"""

from sklearn import datasets

data = datasets.load_diabetes()

X = data['data']
y = data['target']

from sklearn.linear_model import LinearRegression
regr = LinearRegression()

#X = taxi_200['trip_distance'].values.reshape(-1,1)
#y = taxi_200['tip_amount']

regr.fit(X,y)

print('beta_0 = {:0.3f}'.format(regr.intercept_))

print('beta_1 = {:0.3f}'.format(regr.coef_[0]))


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly = PolynomialFeatures(degree=2, include_bias=True)
poly.fit_transform(X[:3])

model = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=True)),
                  ('linear', LinearRegression(fit_intercept=False))])

model = model.fit(X, y)

for i in range(12):
    print('beta_{:} = {:0.3}'.format(i,model.named_steps['linear'].coef_[i]))

_ = sns.lmplot(x='trip_distance', y='tip_amount', data=taxi_200, order=2)


##############################################################################
# Pipeline
##############################################################################

from sklearn.preprocessing import FunctionTransformer

def get_categorical_cols(X):
    categorical_col_idxs = [6]
    return X[:,categorical_col_idxs]

def get_numerical_cols(X):
    real_col_idxs = [3]
    return X[:,real_col_idxs]

from sklearn.pipeline import Pipeline

# categorical (cannot be processed at the same time as numerical)
cat_pipeline = Pipeline([('selector',FunctionTransformer(get_categorical_cols, validate=False)),
                         ('imputer',Imputer(strategy='most_frequent')),
                         ('onehot',OneHotEncoder())])
    
# numerical
num_pipeline = Pipeline([('selector',FunctionTransformer(get_numerical_cols, validate=False)),
                         ('imputer',Imputer(strategy='median')),
                         ('robust_scaler',RobustScaler())])

from sklearn.pipeline import FeatureUnion

pipeline = FeatureUnion([('cat_pipeline',cat_pipeline),
                         ('num_pipeline',num_pipeline)])


##############################################################################
# FunctionTransformer
##############################################################################
    
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

def func_custom(X, arg1):
    return(X/arg1)

from sklearn.preprocessing import FunctionTransformer

ft = FunctionTransformer(func_custom, validate=True, kw_args={'arg1': 2})

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
pipe = Pipeline([('ft', ft),('svm', SVC(gamma='auto'))])

pipe.fit(X, y)
pipe.predict(X)

pipe.named_steps['ft']

X1 = X[:5,:]
print(X1)

print(pipe.named_steps['ft'].transform(X1))

pipe.named_steps['ft'].set_params(kw_args={'arg1': 4})

print(pipe.named_steps['ft'].transform(X1))



##############################################################################
# Poly grid search
##############################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(RobustScaler(),
                         PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.5,
                                                    random_state=0)

params = {'polynomialfeatures__degree':[1,2,3,4,5,6,7,8,9,10],
          'linearregression__normalize': [False, True]}

pr = PolynomialRegression()

gscv = GridSearchCV(pr,params,cv=5)

gscv.fit(X, y)

print(gscv.best_params_)


from sklearn.model_selection import validation_curve

degree = np.arange(0,6)
train_score, validation_score = validation_curve(PolynomialRegression(),
                                                 X_train, y_train,
                                                 'polynomialfeatures__degree',
                                                 degree, cv=3)


_ = plt.plot(degree, np.median(train_score,1), color='b',label='training score')
_ = plt.plot(degree, np.median(validation_score,1), color='r', label='validation score')
_ = plt.xlabel('degree'), plt.ylabel('score')
_ = plt.legend(loc='best')




 # feature selection

print(__doc__)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]

y_true = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
y_pred = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]

cm = confusion_matrix(y_true, y_pred)
cm = cm / len(y_true)

sns.heatmap(cm, cmap='Blues')
plt.savefig('data/dst/sklearn_confusion_matrix.png')




# Confusion Matrix

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

y_true = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
y_pred = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)
class_names = np.array(['True','False'])

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

