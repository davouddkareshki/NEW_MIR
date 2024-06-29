import numpy as np
from tqdm import tqdm

from fasttext_model import FastText
import pandas as pd 

class BasicClassifier:
    def __init__(self):
        pass 
    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, data):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        num = 0
        num_of_positive = 0
        for val in data['label'] :
            if val == 1 : 
                num_of_positive += 1
            num += 1 
        #print(num_of_positive)
        #print(num)
        return num_of_positive / num

if __name__ == "__main__":
    BC = BasicClassifier() 
    data = pd.read_csv('vectorized_IMDB_reviews.csv') 
    print('percentage of positive reviews :', BC.get_percent_of_positive_reviews(data))
    # awnser was 0.5