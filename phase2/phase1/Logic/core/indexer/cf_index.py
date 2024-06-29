import time
import os
import json
import copy
from indexes_enum import Indexes
import sys

class cf_Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents


    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for movie in self.preprocessed_documents : 
            current_index[movie['id']] = movie
        return current_index

    def cf_index_function(self, where):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        cf = {}
        sum_lentgh = 0
        for movie in self.preprocessed_documents : 
            for st in movie[where] : 
                tokens = st.split(' ')
                for token in tokens :
                    if token not in cf.keys() : 
                        cf[token] = 0
                    cf[token] += 1  
                    sum_lentgh += 1
        for term in cf.keys() : 
            cf[term] /= sum_lentgh 
        return cf

def main():
    json_file_path = "../preprocessed_IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        preprocessed_documents = json.load(file)
    index = cf_Index(preprocessed_documents)
    for where in ['summaries', 'genres', 'stars'] :
        cf = index.cf_index_function(where) 
        with open(where+'_cf_indexer.json','w') as f : 
            json.dump(cf, f)

if __name__ == '__main__':
    main()
