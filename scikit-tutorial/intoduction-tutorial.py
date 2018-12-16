from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
print(clf.predict(digits.data[-1:]))

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(digits.data[0:1]))

clf_iris =svm.SVC(gamma='scale')
clf_iris.fit(iris.data, iris.target_names[iris.target])
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
print(list(clf_iris.predict(iris.data[:3])))