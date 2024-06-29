from indexes_enum import Indexes, Index_types

import json

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        self.documents = self.read_documents()    
        self.index = {
            Indexes.DOCUMENTS.value: None,
            Indexes.STARS.value: None,
            Indexes.GENRES.value: None,
            Indexes.SUMMARIES.value: None,
        }

        for index_type in ['documents', 'stars', 'genres', 'summaries']: 
            with open(index_type+' indexer.json','r') as f : 
                self.index[index_type] = json.load(f)

        self.metadata_index = self.create_metadata_index()

    def read_documents(self):
        """
        Reads the documents.
        
        """
        json_file_path = "../preprocessed_IMDB_crawled.json"
        with open(json_file_path, "r") as file:
            preprocessed_documents = json.load(file)

        return preprocessed_documents 

    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['averge_document_length'] = {
            'stars': self.get_average_document_field_length('stars'),
            'genres': self.get_average_document_field_length('genres'),
            'summaries': self.get_average_document_field_length('summaries')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        ans = 0
        for movie in self.documents : 
            S = movie[where] 
            text = S
            if type(S) == list : 
                text = '' 
                for s in S : 
                    text = s + ' ' 
            text = text.split(' ') 
            ans += len(text) 
        return ans / len(self.documents)

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        #path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open('metadata'+' indexer.json', 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


def main() : 
    meta_index = Metadata_index(None)
    meta_index.store_metadata_index(None)

if __name__ == "__main__":
    main()
