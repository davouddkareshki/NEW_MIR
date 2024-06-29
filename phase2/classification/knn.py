import numpy as np
from sklearn.metrics import classification_report
import tqdm 

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader
import pandas as pd

class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.vectors = None 
        self.labels = None 
    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.vectors = X 
        self.labels = y 
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
        distances = np.linalg.norm(x - self.vectors, axis=1) 
        nearest_neighbors_idxs = distances.argsort()[:self.k]
        vote = {}
        for idx in nearest_neighbors_idxs :
            value = self.labels[idx]  
            if value not in vote.keys() : 
                vote[value] = 0
            vote[value] += 1
        mx_vote = -1 
        mx_key = None 
        for key in vote.keys() : 
            if vote[key] > mx_vote : 
                mx_vote = vote[key]
                mx_key = key  
        return mx_key

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
        y_pred = [self.predict(x) for x in tqdm.tqdm(X)] 
        return classification_report(y,y_pred) 

# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """    
    knn_model = KnnClassifier(5)
    RL = ReviewLoader() 
    data = pd.read_csv('vectorized_IMDB_reviews.csv') 
    X_train, X_test, y_train, y_test = RL.split_data(data)
    knn_model.fit(X_train,y_train)
    print(knn_model.prediction_report(X_test,y_test)) 
# F1 score was between 72% and 75%
