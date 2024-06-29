import numpy as np
import pandas as pd 

from sklearn.metrics import classification_report
from sklearn.svm import SVC

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()
        pass 

    def fit(self, X, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(X,y)
        pass

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        self.model.predict(x)
        pass

    def prediction_report(self, X, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.model.predict(X) 
        return classification_report(y,y_pred) 


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    svm_model = SVMClassifier()
    RL = ReviewLoader() 
    data = pd.read_csv('vectorized_IMDB_reviews.csv') 
    X_train, X_test, y_train, y_test = RL.split_data(data)
    #print(X_train.shape)
    #print(y_train.shape)
    svm_model.fit(X_train,y_train)
    print(svm_model.prediction_report(X_test,y_test)) 
# f1 score = 86%