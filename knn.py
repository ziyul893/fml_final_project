# add the code after dataloader (inside main?)
# X_test, X_train, y_test, y_train can use enumerate to obtain. 
# svm.py has steps to do that, but it is slow
from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn_clf.fit(X_train,y_train)
ypred=knn_clf.predict(X_test) #These are the predicted output values