"""
Created on Fri Dec 08 04:54:49 

@author: Bren Guzmán, Brenda García, María José Merino
"""

#%% LIBRERÍAS 

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from SVM import support_vector_machine as SVM


#%% CONJUNTO DE DATOS

X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0, cluster_std=0.45)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', edgecolor='k', marker='p')
plt.title('Conjunto de datos sintético')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()

#%% HOLDOUT PARA 75%-25%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

svm = SVM()
w,b = svm.fit(X_train,y_train)

print("Dataset (75%-25%), accuracy:" ,accuracy_score(svm.predict(X_test),y_test))

svm.plot_svm(X_train , y_train, X_test, y_test, 'SVM para conjunto de datos sintético (75%-25%)' )


