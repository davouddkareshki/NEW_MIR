import time
import os
import json
import copy
from indexes_enum import Indexes
import numpy as np

class Tiered_index:
    def __init__(self, preprocessed_documents: list, number_of_doc_in_each_tier):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents
        self.N = len(preprocessed_documents) 
        self.number_of_doc_in_each_tier = number_of_doc_in_each_tier

        with open("documents indexer.json", 'r') as f : 
            self.documents_index = json.load(f)

        self.tiered_index = {
            Indexes.STARS.value: self.index(Indexes.STARS.value),
            Indexes.GENRES.value: self.index(Indexes.GENRES.value),
            Indexes.SUMMARIES.value: self.index(Indexes.SUMMARIES.value),
        }

        self.store(Indexes.STARS.value)
        self.store(Indexes.GENRES.value)
        self.store(Indexes.SUMMARIES.value)

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

    def tokenize(self,list_text) :
        try :  
            all_tokens = [] 
            for text in list_text : 
                tokens = text.split(' ')
                for token in tokens : 
                    all_tokens.append(token) 
            return all_tokens 
        except : 
            print('failed to tokenizing') 

    def index(self,field):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        tf = {}
        idf = {} 
        for movie in self.preprocessed_documents : 
            tokens = self.tokenize(movie[field])
            doc_id = movie['id']

            for term in tokens :
                if term not in tf :
                    tf[term] = {} 
                if doc_id not in tf[term] : 
                    tf[term][doc_id] = 0
                tf[term][doc_id] = tf[term][doc_id] + 1 
        
        for term in tf.keys() : 
            idf[term] = (self.N/len(tf[term].keys()))
        
        weight = {}
        for movie in self.preprocessed_documents : 
            tokens = self.tokenize(movie[field])
            doc_id = movie['id']

            for term in tokens :
                if term not in weight : 
                    weight[term] = {}
                weight[term][doc_id] = np.log(tf[term][doc_id]+1) * idf[term] 
    
        index = {}
        for term in weight.keys() :
            weight_doc_list = [] 

            for doc_id in weight[term].keys() : 
                weight_doc_list.append((weight[term][doc_id],doc_id))
            weight_doc_list = sorted(weight_doc_list,reverse=True)  
       #     if len(weight_doc_list) > 5 : 
        #        print(term)
         #       print(weight_doc_list)
            start_index = 0
            for num_tier in range(int(self.N/self.number_of_doc_in_each_tier)+1) : 
                if num_tier not in index : 
                    index[num_tier] = {}
                
                for _ in range(self.number_of_doc_in_each_tier) : 
                    if start_index >= len(weight_doc_list) : 
                        break

                    khar,doc_id = weight_doc_list[start_index]
                    start_index += 1
                    if term not in index[num_tier] : 
                        index[num_tier][term] = {}
                    if doc_id not in index[num_tier][term] : 
                        index[num_tier][term][doc_id] = {}
                    index[num_tier][term][doc_id] = tf[term][doc_id]
        
        return index 
    
    def store(self, index_name) : 
        with open('tiered ' + index_name + ' index.json', "w") as file:
            json.dump(self.tiered_index[index_name],file)

def main():
    
    json_file_path = "../preprocessed_IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        preprocessed_documents = json.load(file)
    index = Tiered_index(preprocessed_documents,10)

if __name__ == '__main__':
    main()
