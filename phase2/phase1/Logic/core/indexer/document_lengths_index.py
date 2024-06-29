import json
from indexes_enum import Indexes,Index_types

class DocumentLengthsIndex:
    def __init__(self):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """
        with open("documents indexer.json", 'r') as f : 
            self.documents_index = json.load(f)
        self.document_length_index = {
            Indexes.STARS: self.get_documents_length(Indexes.STARS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.SUMMARIES: self.get_documents_length(Indexes.SUMMARIES.value)
        }
        self.store_document_lengths_index(Indexes.STARS)
        self.store_document_lengths_index(Indexes.GENRES)
        self.store_document_lengths_index(Indexes.SUMMARIES)

    def get_documents_length(self, where):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """
        ans = {}
        for idx in self.documents_index.keys() : 
            movie = self.documents_index[idx]
            
            lis = []
            for text in movie[where] :
                text = text.split(' ') 
                lis.append(len(text))
            ans[idx] = lis 
        return ans 
    
    def store_document_lengths_index(self, index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        name = index_name.value + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        with open(name, 'w') as file:
            json.dump(self.document_length_index[index_name], file, indent=4)
    

if __name__ == '__main__':
    document_lengths_index = DocumentLengthsIndex()
    print('Document lengths index stored successfully.')