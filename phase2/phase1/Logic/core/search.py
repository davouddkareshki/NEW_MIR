import json
import numpy as np
from preprocess import Preprocessor
from scorer import Scorer
from indexer.indexes_enum import Indexes, Index_types
from indexer.index_reader import Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = 'indexer/'
        self.document_indexes = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS).index,
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES).index,
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES).index
        }
        self.tiered_index = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS, Index_types.TIERED).index,
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES, Index_types.TIERED).index,
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED).index
        }
        self.document_lengths_index = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH).index,
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH).index,
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH).index
        }
        self.cf_index = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS, Index_types.CF).index,
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES, Index_types.CF).index,
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES, Index_types.CF).index
        }
        
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA)
        
      #  print(self.document_indexes[Indexes.STARS.value])
    
    def search(self, query, method, weights, safe_ranking=True, max_results=10, smoothing_method=None, alpha=0.5, lamda=0.5):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25 | Unigram
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results.
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """
        #print(query)
        preprocessor = Preprocessor([query])
        query = preprocessor.process_text(query)
        #print(query)

        scores = {}
        if method == 'unigram' : 
            scores = self.find_scores_with_unigram_model(query,smoothing_method,weights)
        else : 
            if safe_ranking:
                scores = self.find_scores_with_safe_ranking(query, method, weights)
            else:
                scores = self.find_scores_with_unsafe_ranking(query, method, weights, max_results)

        final_scores = self.aggregate_scores(weights, scores)
        
        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        final_scores = {} 
        for field in scores.keys() : 
            doc_scores = scores[field]
            for doc_id in doc_scores.keys() : 
                if doc_id not in final_scores : 
                    final_scores[doc_id] = 0
                final_scores[doc_id] += weights[field] * doc_scores[doc_id] 
        return final_scores

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        query_terms = query.split(' ')
       # print(query_terms)
        scores = {} 
        for field in weights:
            scores[field] = {}
            for tier in self.tiered_index[field].keys():
                #print(tier)
                doc_lentgh_index_tier = {} 
                for term in query_terms : 
                    #print(self.tiered_index[field][tier])
                    if term not in self.tiered_index[field][tier] : 
                        continue

                    for doc_id in self.tiered_index[field][tier][term] : 
                   #     print(doc_id)
                        doc_lentgh_index_tier[doc_id] = self.document_lengths_index[field][doc_id]

                scorer = Scorer(self.tiered_index[field][tier], 9950, doc_lentgh_index_tier) 
                if method == 'OkapiBM25' :
                    scores_of_tier = scorer.compute_socres_with_okapi_bm25(query)
                else :  
                    scores_of_tier = scorer.compute_scores_with_vector_space_model(query,method)
                    
                scores[field] = self.merge_scores(scores[field],scores_of_tier)
                if len(scores[field].keys()) > max_results :
                    break 
        return scores
    
    def find_scores_with_safe_ranking(self, query, method, weights):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """
        scores = {} 
        for field in weights:
            scorer = Scorer(self.document_indexes[field],9950,self.document_lengths_index[field]) 
            if method == 'OkapiBM25' :
                scores[field] = scorer.compute_socres_with_okapi_bm25(query)
            else : 
                scores[field] = scorer.compute_scores_with_vector_space_model(query,method)
        return scores 

    def find_scores_with_unigram_model(self, query, smoothing_method, weights, alpha=0.5, lamda=0.5):
            """
            Calculates the scores for each document based on the unigram model.

            Parameters
            ----------
            query : str
                The query to search for.
            smoothing_method : str (bayes | naive | mixture)
                The method used for smoothing the probabilities in the unigram model.
            weights : dict
                A dictionary mapping each field (e.g., 'stars', 'genres', 'summaries') to its weight in the final score. Fields with a weight of 0 are ignored.
            scores : dict
                The scores of the documents.
            alpha : float, optional
                The parameter used in bayesian smoothing method. Defaults to 0.5.
            lamda : float, optional
                The parameter used in some smoothing methods to balance between the document
                probability and the collection probability. Defaults to 0.5.
            """
            scores = {} 
            for field in weights:
                scorer = Scorer(self.document_indexes[field],1005,self.document_lengths_index[field], self.cf_index[field]) 
                scores[field] = scorer.compute_scores_with_unigram_model(query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5)
            return scores 
    
    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """
        scores = {} 
        for key in scores1.keys() : scores[key] = scores1[key] 
        for key in scores2.keys() : scores[key] = scores2[key] 
        return scores 
    

def main() :
    search_engine = SearchEngine()
    query = "fight hard club"
    method = "unigram"
    weights = {
        Indexes.STARS.value: 1,
        Indexes.GENRES.value: 1,
        Indexes.SUMMARIES.value: 1
    }
    result = search_engine.search(query, method, weights,smoothing_method='mixture')

    print(result)

if __name__ == '__main__':
    main()