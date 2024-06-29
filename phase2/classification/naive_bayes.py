import numpy as np
import pandas as pd 
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.number_of_samples, self.number_of_features = X.shape
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        
        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        #print(y)
        for idx, c in enumerate(self.classes):
            X_class = [] 
            for i in range(len(y)) : 
                if y[i] == c : 
                    X_class.append(X[i])
            X_class = np.array(X_class)
            #print(X_class)
            self.prior[idx] = (X_class.shape[0] + self.alpha) / (self.number_of_samples + self.num_classes * self.alpha)
            self.feature_probabilities[idx, :] = (np.sum(X_class, axis=0).toarray() + self.alpha) / (np.sum(X_class).toarray() + self.number_of_features * self.alpha)
        
        self.log_probs = np.log(self.feature_probabilities)

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
        #print(x)
        #x = self.cv.fit_transform([x]).toarray()
        x = x.toarray()
        self.cv.get_feature_names_out()
        probs = np.dot(x, self.log_probs.T) + np.log(self.prior)    
        return self.classes[np.argmax(probs)]

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
        return classification_report(y, y_pred)

    def get_percent_of_positive_reviews(self):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        index_of_pos = 0 
        if self.classes[1] == 'positive' : 
            index_of_pos = 1
        return self.prior[index_of_pos] / (self.prior[0] + self.prior[1])



# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    #RL = ReviewLoader() 
    data = pd.read_csv('IMDB_reviews.csv')
    '''
    CV = CountVectorizer() 
    doc_term_mat = CV.fit_transform(data['review'], data['sentiment'])

    print(doc_term_mat.shape)
    print(doc_term_mat[0][0])
    '''
    CV = CountVectorizer()
    model = NaiveBayes(CV)
    X = np.array(data['review'])
    y = np.array(data['sentiment'])
   # print(y)
   # print('-------------------')
    test_size = int(len(X)/5) 
    train_size = len(X) - test_size 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, shuffle=True)  
    
    CV.fit(X)
    X_train = CV.transform(X_train)
    #print(len(CV.get_feature_names_out()))
    #print(CV.get_feature_names_out())
    X_test = CV.transform(X_test)
    #print(len(CV.get_feature_names_out()))
    #print(CV.get_feature_names_out())


    print('start naive bayes ...')
    model.fit(X_train, y_train) 
    #print('percentage of positives in train data :', model.get_percent_of_positive_reviews())
    #print('on diffrent shuffles it could be diffrent number')
    print(model.prediction_report(X_test, y_test))
# F1 score between 85% and 86%
    

