import fasttext
import re
import numpy as np

import json
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import pandas as pd 
import string

def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True, punctuation_removal=True):
    """
    Preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        Text to be preprocessed
    minimum_length: int
        Minimum length of the token
    stopword_removal: bool
        Whether to remove stopwords
    stopwords_domain: list
        List of stopwords to be removed base on domain
    lower_case: bool
        Whether to convert to lowercase
    punctuation_removal: bool
        Whether to remove punctuations
    """

    stop_words = set(stopwords.words('english'))
    if lower_case:
        text = text.lower()
    if punctuation_removal :
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)
    if minimum_length > 0:
        text = ' '.join([word for word in text.split() if len(word) >= minimum_length])
    if stopword_removal : 
        tokens = text.split(' ') 
        text = ''
        for token in tokens : 
            if token not in stop_words : 
                text += token + ' '
        text = text.strip()
    text = text.strip()
    return text

class FastText:
    def __init__(self, method='skipgram', preprocessor=None):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        preprocessor : function, optional
            A function for text preprocessing.
        """
        self.method = method
        self.model = None
        self.preprocessor = preprocessor

    def train(self, path):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        #print(texts[1])
        #print(texts.shape)
        #preprocessed_texts = [self.preprocessor(text) for text in texts]
        #texts = [text for arr in texts for text in arr]
        self.model = fasttext.train_unsupervised(input='train_data.csv', model=self.method)

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        tokens = word_tokenize(query)

        embeddings = [self.model.get_word_vector(token) for token in tokens]
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return None

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """

        vector = self.model.get_word_vector(word2) - self.model.get_word_vector(word1) + self.model.get_word_vector(word3)

        best_word, min_distance = None, float('inf')
        for word in self.model.words:
            if word not in [word1, word2, word3]:
                word_vector = self.model.get_word_vector(word)
                dist = np.linalg.norm(vector - word_vector)
                if dist < min_distance:
                    best_word, min_distance = word, dist

        return best_word

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        save : bool, optional
            Whether to save the model after training.
        path : str, optional
            The path to save or load the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        elif mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)

if __name__ == "__main__":

    ft_model = FastText(preprocessor=preprocess_text, method='skipgram')

    path = 'train_data.csv'
    
    '''
    # train FastText model
    ft_model.train(path)
    ft_model.prepare(None, mode = "save", save=True)
    '''

    ft_model.prepare(None, mode = "load")
    
    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "woman"
    #print(ft_model.model.get_word_vector(word1 + ' ' + word2))
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
    