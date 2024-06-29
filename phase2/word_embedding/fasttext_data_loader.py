import fasttext
import re
import nltk
import numpy as np
#nltk.download('stopwords')
import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
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


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        pass

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open(self.file_path, 'r') as f:
            docs = json.load(f)
        #print(docs[0].keys())
        return pd.DataFrame(docs)

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        X = [[],[],[],[]]
        
        train_data_frame = pd.DataFrame() 
        for key in df.keys() : 
            if key in ['synposis', 'summaries', 'reviews', 'title'] :
                for lis in df[key] :
                    text = ''
                    if key == 'title' : 
                        text = lis
                    else : 
                        if key == 'reviews' : 
                            for lis2 in lis : 
                                for st in lis2 : 
                                    text = text + ' ' + st
                        else :
                            for st in lis :
                                text = text + ' ' + st  
                    text = preprocess_text(text)
                    # print(text)
                    # print(preprocess_text(text=text))
                    f = {'synposis' : 0, 'summaries' : 1, 'reviews' : 2, 'title' : 3}
                    X[f[key]].append(text)
        X = np.array(X) 
        X = np.transpose(X) 
        label_encoder = LabelEncoder()
        y = []
        for i,label in enumerate(df['genres']) :
            text = ''
            if len(label) > 0 :
                text = label[0]             
            y.append(text)
        y = label_encoder.fit_transform(y)

        return np.array(X), np.array(y)

if __name__ == "__main__":
    ft_data_loader = FastTextDataLoader('IMDB_crawled.json')

    X,y = ft_data_loader.create_train_data() 
    texts = [text for arr in X for text in arr]
    data = pd.DataFrame(texts) 
    data.to_csv('train_data.csv', index=False)

