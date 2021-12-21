# 20211947_project
------------------
This repo contains machine learning algorithms
1. polynomial regression
2. Face recognition

I will write about the second algorithm.
This is a face recognition algorithm using olivetti_face.
I'll explain the code now.

```
import sklearn.datasets
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import numpy as np

import matplotlib.pyplot as plt 
%matplotlib inline
```
This part is the process of bringing up numpy, sklearn, and matplotlib packages.

```
olivetti_faces = sklearn.datasets.fetch_olivetti_faces(random_state=0,)
print(olivetti_faces['DESCR'])

example_indices = [0, 10, 62, 70]
for idx in example_indices:
    plt.title(olivetti_faces['target'][idx])
    plt.imshow(olivetti_faces['images'][idx])
    plt.gray()
    plt.show()
    
X = olivetti_faces['data']
y = olivetti_faces['target']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
```
It's the process of bringing up olivetti_face data.

Let me explain the variables in this code to find best model.
'olivetti_faces' is a variable containing a set of face images
'X_train' is feature vectors of training dataset
'y_train' is target labels of training dataset
'X_test' is feature vectors of test dataset
'y_test' is target labels of test dataset
'y_pred' was initialized as zero vectors and fill 'y_pred' with predicted labels

The following is the process of finding the best model using the scikitlearn package.
```
ppn = sklearn.linear_model.Perceptron(max_iter=100, tol=0.001, n_iter_no_change=9, random_state=0,alpha=0.5,eta0=1.0)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
```
I used perceptron and changed several hyperparameters such as random_state and max_iter.

```
print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```
As a result of checking the account through the code above, 0.86 was found.
