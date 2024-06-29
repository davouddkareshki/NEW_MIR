import time
import os
import json
import copy
from indexes_enum import Indexes
import sys

class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

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

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for movie in self.preprocessed_documents : 
            tf = {}
            tokens = self.tokenize(movie['stars'])

            for term in tokens :
                if term not in tf : 
                    tf[term] = 0
                tf[term] = tf[term] + 1 

            for term in tf.keys() :
                if term not in index : 
                    index[term] = {}  
                index[term][movie['id']] = tf[term]
        return index 

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for movie in self.preprocessed_documents : 
            tf = {}
            tokens = self.tokenize(movie['genres'])

            for term in tokens :
                if term not in tf : 
                    tf[term] = 0
                tf[term] = tf[term] + 1 

            for term in tf.keys() :
                if term not in index : 
                    index[term] = {}  
                index[term][movie['id']] = tf[term]
        return index 
    
    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for movie in self.preprocessed_documents : 
            tf = {}
            tokens = self.tokenize(movie['summaries'])

            for term in tokens :
                if term not in tf : 
                    tf[term] = 0
                tf[term] = tf[term] + 1 

            for term in tf.keys() :
                if term not in index : 
                    index[term] = {}  
                index[term][movie['id']] = tf[term]
        return index
     
    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            if word in self.index[index_type] : 
                return self.index[index_type][word].keys()
            else : 
                return []
        except:
            print('failed to get posting list')
            return 
        
    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        
        # doc idx indexing 
        movie = document 
        idx = document['id']
        #self.preprocessed_documents.append(movie)
        self.index['documents'][idx] = movie 
        
        # stars indexing
        tf = {}
        tokens = self.tokenize(movie['stars'])
        for term in tokens :
            if term not in tf : 
                tf[term] = 0
            tf[term] = tf[term] + 1 
        for term in tf.keys() :
            if term not in self.index['stars'] : 
                self.index['stars'][term] = {}  
            self.index['stars'][term][idx] = tf[term]

        # genres indexing
        tf = {}
        tokens = self.tokenize(movie['genres'])
        for term in tokens :
            if term not in tf : 
                tf[term] = 0
            tf[term] = tf[term] + 1 
        for term in tf.keys() :
            if term not in self.index['genres'] : 
                self.index['genres'][term] = {}  
            self.index['genres'][term][idx] = tf[term]

        # summaries indexing 
        tf = {}
        tokens = self.tokenize(movie['summaries'])
        for term in tokens :
            if term not in tf : 
                tf[term] = 0
            tf[term] = tf[term] + 1 
        for term in tf.keys() :
            if term not in self.index['summaries'] : 
                self.index['summaries'][term] = {}  
            self.index['summaries'][term][idx] = tf[term]

        pass

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """
        idx = document_id
        movie = self.index[Indexes.DOCUMENTS.value][idx] 
        
        #doc idx indexing
        del self.index[Indexes.DOCUMENTS.value][idx] 

        # stars indexing
        tokens = self.tokenize(movie['stars'])
        for term in tokens :
            if term in self.index[Indexes.STARS.value] : 
                del self.index[Indexes.STARS.value][term][idx]  

        # genres indexing
        tokens = self.tokenize(movie['genres'])
        for term in tokens :
            if term in self.index[Indexes.GENRES.value] : 
                del self.index[Indexes.GENRES.value][term][idx] 

        # summaries indexing 
        tokens = self.tokenize(movie['summaries'])
        for term in tokens :
            if term in self.index[Indexes.SUMMARIES.value] : 
                del self.index[Indexes.SUMMARIES.value][term][idx] 

        pass

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)
       # print(index_after_add[Indexes.DOCUMENTS.value])
        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_type: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_type: str or None
            type of index we want to store (documents, stars, genres, summaries)
            if None store tiered index
        """

        if index_type is None:
            raise TypeError('index type is None')

        if index_type not in self.index:
            raise ValueError('Invalid index type')

        with open(index_type+' indexer.json','w') as f : 
            json.dump(self.index[index_type], f)
        pass

    def load_index(self, path: str, index_type: str = None):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        
        with open(index_type+' indexer.json','r') as f : 
            data = json.load(f)
        return data

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'jaw'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
       # print('size of preprocessed documents in byte :',sys.getsizeof(self.preprocessed_documents))
       # print(self.preprocessed_documents)
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue
            
            for field in document[index_type]:
                field = field.split(' ')
                if check_word in field:
               #     if index_type == 'genres' : 
                #        print(document['id'])
                    docs.append(document['id'])
                    break

            #     if index_type == 'genres' : 
                #        print(document['id'])
               # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)
     #   print(posting_list)
        end = time.time()
        implemented_time = end - start

        print()
        print(index_type, 'indexing :')
        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

   #     print(set(posting_list))
    #    print(set(docs))
      #  print(self.index['documents']['tt1453405'])
        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods
def main():
    
    json_file_path = "../preprocessed_IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        preprocessed_documents = json.load(file)
    index = Index(preprocessed_documents)
    index.check_add_remove_is_correct()
    
    index.check_if_indexing_is_good(Indexes.STARS.value)
    index.check_if_indexing_is_good(Indexes.GENRES.value)
    index.check_if_indexing_is_good(Indexes.SUMMARIES.value)
    
    index.store_index(None,Indexes.DOCUMENTS.value)
    index.store_index(None,Indexes.STARS.value)
    index.store_index(None,Indexes.GENRES.value)
    index.store_index(None,Indexes.SUMMARIES.value)
    
    print()
    print('correct loadings :')
    print(index.check_if_index_loaded_correctly(Indexes.DOCUMENTS.value,index.load_index(None,Indexes.DOCUMENTS.value)))
    print(index.check_if_index_loaded_correctly(Indexes.STARS.value,index.load_index(None,Indexes.STARS.value)))
    print(index.check_if_index_loaded_correctly(Indexes.GENRES.value,index.load_index(None,Indexes.GENRES.value)))
    print(index.check_if_index_loaded_correctly(Indexes.SUMMARIES.value,index.load_index(None,Indexes.SUMMARIES.value)))
if __name__ == '__main__':
    main()
