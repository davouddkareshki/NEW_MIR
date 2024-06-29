import json 
import numpy as np
import string 
import nltk
from nltk.stem import WordNetLemmatizer
import re 
from typing import List
import tqdm 

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        
        self.documents = documents
        file =  open("stopwords.txt", "r")          
        self.stopwords = file.read().split('\n')
        arr = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
                'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
                'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
                'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
                'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        for word in arr : 
            self.stopwords.append(word)
        self.stopwords = set(self.stopwords)
        self.wnl = WordNetLemmatizer()

    def process_text(self, text: str) :
        flag = 0
       # if text == 'panda' : 
        #    print('this is here') 
         #   flag = 1

        text = self.remove_links(text) 
        text = self.remove_punctuations(text) 
     #   text = self.remove_stopwords(text) 
        text = self.remove_additional_space(text) 
        words = self.tokenize(text) 
        text = ''
        #if flag == 1 : 
         #   print(text)
        for word in words : 
          #  if flag == 1 :
           #     print('in preproccesor')
            #    print(word)
            word = self.normalize(word) 
            if word in self.stopwords : 
                continue
            #if flag == 1 :
             #   print(word)
              #  print('end')
            text = text + word + ' '
        text = text.strip()
      #  if flag == 1 : 
       #     print(text)
        return text

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        str
            The preprocessed documents.
        """
        preprocessed_documents = []
        for movie in tqdm.tqdm(self.documents) : 
            new_movie = {}
            for key in movie.keys() : 
                if movie[key] == None : 
                    if key == 'reviews' or 'summaries' or 'stars' or 'genres' : 
                        movie[key] = []
                    else : 
                        movie[key] = ''
                new_movie[key] = movie[key] 

                if type(movie[key]) == str : 
                 #   print('-----------------')
                #    print(movie[key])
                    new_movie[key] = self.process_text(movie[key])
             #       print('***')
              #      print(new_movie[key])
               #     print('-----------------')
                if type(movie[key]) == list and key != 'reviews':
                  #  print(key) 
                #    print('-----------------')
               #     print(movie[key])
                    new_list = [] 
                    for text in movie[key] : 
                        text = self.process_text(text) 
                        new_list.append(text) 
                    new_movie[key] = new_list 
            #        print('***')
             #       print(new_movie[key])
              #      print('-----------------')
                if key == 'reviews' :
                    new_reviews = []
                    #if movie[key] != None :  
                    for lis in movie[key] :
                        new_list = [] 
                        for text in lis : 
                            text = self.process_text(text) 
                            new_list.append(text) 
                        new_reviews.append(new_list)  
                    new_movie[key] = new_reviews   
            preprocessed_documents.append(new_movie)
        return preprocessed_documents 

    def normalize(self, word: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        word = word.lower()
        #print(word, word[-3:])
      #  if word[-3:] == 'ing' : 
       #     print(word,self.wnl.lemmatize(word))
        word = self.wnl.lemmatize(word)
        return word 
    
    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns : 
            text = re.sub(pattern,"",text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)
        return text  

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        tokens = self.tokenize(text) 
        text = ''
        for token in tokens : 
            if token not in self.stopwords : 
                text += token + ' '
        text = text.strip()
        return text 
    def remove_additional_space(self,text: str) : 
        text = re.sub(' +', ' ', text)
        return text 
    
    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        words = text.split(' ')
        return words 

def main():
    
    json_file_path = "../IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        documents = json.load(file)
    preprocessor = Preprocessor(documents)
    documents = preprocessor.preprocess() 

    with open('preprocessed_IMDB_crawled.json','w') as f : 
        json.dump(documents, f)

   # text = 'Davood, Kareshki!, https://www.geeksforgeeks.org/python-string-lower/ salam bar khar az man? for each man in the bar' 
   # print(preprocessor.process_text(text))
    '''
    print(text)
    text = preprocessor.remove_links(text) 
    print(text) 
    text = preprocessor.remove_punctuations(text) 
    print(text)
    text = preprocessor.remove_stopwords(text) 
    print(text)
    text = preprocessor.remove_additional_space(text) 
    print(text)
    '''
 #   text = preprocessor.normalize(text) 
  #  print(text)
    
   # wnl = WordNetLemmatizer()
   # text = 'drive the car by your hands'
   # text = 'by hands' 
   # print(wnl.lemmatize(text))
    
if __name__ == '__main__':
    main()
