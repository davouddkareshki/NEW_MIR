import numpy as np
import pandas as pd
import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fasttext_model import FastText


class ReviewLoader:
    def __init__(self, file_path = None):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        ft_model = FastText()
        ft_model.prepare(None, mode = "load")
        self.fasttext_model = ft_model
        data = pd.read_csv('IMDB_reviews.csv')
        new_data = []
        for i in tqdm.tqdm(range(int(len(data['review'])))) : 
            text = data['review'][i] 
            vector = ft_model.get_query_embedding(text) 
            label = 1
            if data['sentiment'][i] == 'negative' : label = -1
            #print(vector)
            D = {'label' : label}
            for idx in range(len(vector)) : 
                D[idx] = vector[idx]
            #print(D)
            self.embeddings.append(vector)
            self.sentiments.append(label) 
            for token in text.split(' ') : 
                self.review_tokens.append(token) 
            new_data.append(D)
        new_data = pd.DataFrame(new_data)
        new_data.to_csv('vectorized_IMDB_reviews.csv', index=False)
        #with open('vectorized_IMDB_reviews.json','w') as f : 
        #    json.dump(new_data, f)
        pass

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        return self.embeddings 

    def split_data(self, data, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """

        X = np.array(data.drop('label', axis=1))
        for i in range(len(X)) : 
            X[i] = np.array(X[i]) 
       # print(X.shape)
        y = np.array(data['label']) 
        
        N = X.shape[0] 
        test_size = int(test_data_ratio * N) 
        train_size = N - test_size  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, shuffle=True)  
        return X_train, X_test, y_train, y_test
if __name__ == "__main__":
    
    #vectorize data with Fasttext 
    RL = ReviewLoader() 
    RL.load_data() 
    
    # test-train spliting 
    data = pd.read_csv('vectorized_IMDB_reviews.csv') 
    X_train, X_test, y_train, y_test = RL.split_data(data)

