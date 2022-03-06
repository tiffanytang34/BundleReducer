
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np


def crossvalidation(X, dcom, num_fold):
    scaler = StandardScaler()
    imputer = SimpleImputer() 
    n = [2,4,8,16,32,64] # number of dimensions
    loss= np.zeros((len(n),X.shape[0]));
    kf = KFold(n_splits=num_fold);
    for i in range(len(n)): # loop through number of dimensions reduced

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            X_train =imputer.fit_transform(X_train)
            X_test = imputer.fit_transform(X_test)

            clf=dcom(n_components=n[i])

            model = clf.fit(X_train)
            transformed = clf.transform(X_test)
            reconstruct = clf.inverse_transform(transformed)

    
        
            loss[i, test_index] = np.sqrt(np.mean((reconstruct-X_test)**2, axis = -1));
    return loss