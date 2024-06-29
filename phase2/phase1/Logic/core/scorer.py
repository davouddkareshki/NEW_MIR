import numpy as np
import json 
import math
 
class Scorer:    
    def __init__(self, index, number_of_documents, index_length, cf_index = None):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents
        self.index_length = index_length
        self.average_document_field_length = self.compute_average_document_field_length()
        self.cf_index = cf_index 

    def get_document_length(self,doc_id) : 
        return np.sum(np.array(self.index_length[doc_id]))
    
    def compute_average_document_field_length(self) : 
        sm = 0
        num = 0
        for doc_id in self.index_length.keys() : 
            sm += self.get_document_length(doc_id)
            num += 1 
        if num == 0 : 
            return 0
        return sm/num
        
    def get_list_of_documents(self,query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        tokens = query.split(' ')
        list_of_documents = []
        for term in tokens:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
     #       print(self.index[term])
      #  print(list_of_documents)
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        if term not in self.idf :
            if term not in self.index.keys() :
                self.idf[term] = 0
            else :  
                self.idf[term] = self.N/len(self.index[term].keys())    
        return self.idf[term]

    def cf_based_prob(self, term) : 
        '''
        #TODO it was better to do it in indexing phase 
        out = 0 
        sum_lentgh = 0
        for doc_id in self.index_length.keys() : 
            if term in self.index.keys() : 
                if doc_id in self.index[term].keys() : 
                    out += self.index[term][doc_id]
            sum_lentgh += self.get_document_length(doc_id)
        return out / sum_lentgh
        '''
        return self.cf_index[term] 

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        
        tokens = query.split(' ') 
        out = {}
        for term in tokens : 
            if term in self.index : 
                out[term] = self.index[term]
            else :
                out[term] = {}
        #    else : 
        #        print('we dont have term', term, 'in our data')
        return out # out[term][doc_id] = tf(term,doc)

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns 
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        doc_scores = {}
        query_tfs = self.get_query_tfs(query) 
        query_docs = self.get_list_of_documents(query)
        for doc_id in query_docs : 
            doc_scores[doc_id] = self.get_vector_space_model_score(query,query_tfs,doc_id,method[:3],method[4:])
        return doc_scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        query_terms = query.split(' ') 
        query_weights = []
        document_weights = []
        tf_q = {}
        for term in query_terms : 
            idf = 1 
            #print(query_tfs, term, )
            tf = 0
            if document_id in query_tfs[term] : 
                tf = query_tfs[term][document_id]
        
            if document_method[0] == 'l': 
                tf = np.log(tf+1) 
            if document_method[1] == 't' : 
                idf = self.get_idf(term)
                if idf > 0 : 
                    idf = np.log(idf)
                
      #      print(document_id, term, tf, idf)
            document_weights.append(idf*tf) 


            if term not in tf_q : 
                tf_q[term] = 0 
            tf_q[term] += 1

        for term in query_terms : 
            idf = 1 
            tf = tf_q[term]
        
            if query_method[0] == 'l': 
                tf = np.log(tf+1)
            if query_method[1] == 't' : 
                idf = self.get_idf(term)
                if idf > 0 : 
                    idf = np.log(idf)

            query_weights.append(idf*tf) 
        
      #  print(document_weights) 
       # print(query_weights)
       # print('----')
        document_weights = np.array(document_weights)
        query_weights = np.array(query_weights) 
        score = 0
        if np.linalg.norm(document_weights) * np.linalg.norm(query_weights) > 0 :
            score = np.dot(document_weights,query_weights) 
          #  print(query_weights[0:5]) 
           # print(document_weights[0:5])
            #print(score) 
            #print("---")
            if document_method[2] == 'c' : 
                score /= np.linalg.norm(document_weights) 
            if query_method[2] == 'c' :
                score /= np.linalg.norm(query_weights)
        return score  

    def compute_socres_with_okapi_bm25(self, query):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        doc_scores = {}
        query_docs = self.get_list_of_documents(query)
        query_tfs = self.get_query_tfs(query) 
        for doc_id in query_docs : 
            doc_scores[doc_id] = self.get_okapi_bm25_score(query,query_tfs,doc_id)
        return doc_scores

    def get_okapi_bm25_score(self, query, query_tfs, document_id):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        tokens = query.split(' ') 
        score = 0
        k1 = 1.2 
        b = 0.75 
        for term in tokens : 
            idf = self.get_idf(term) 
            idf = ((self.N - idf + 0.5) / (idf + 0.5)) + 1
            idf = math.log(idf) 

            tf = 0
            if document_id in query_tfs[term] :
                tf = query_tfs[term][document_id]
            
            up = tf * (k1 + 1)
        #    print('asdf',self.average_document_field_length)
         #   print(self.get_document_length(document_id))
            down = tf + k1 * (1-b + b*(self.get_document_length(document_id) / self.average_document_field_length))

            score += up / down * idf
        return score
     
    def get_unigram_model_score(self, query, query_tfs, document_id, smoothing_method, document_lengths, alpha, lamda):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """

        tokens = query.split(' ')
        
        final_score_based_on_prob = 0
        for term in tokens : 
            tf = 0 
            if document_id in query_tfs[term].keys() : 
                tf = query_tfs[term][document_id] 

            doc_lentgh = self.get_document_length(document_id)
            cf_prob = self.cf_based_prob(term)
        #    print(term, cf_prob)
            prob = 0
            if smoothing_method == 'naive' or smoothing_method == None: 
                prob = tf / doc_lentgh
            if smoothing_method == 'bayes' : 
                prob = (tf + alpha * cf_prob) / (doc_lentgh + alpha)
            if smoothing_method == 'mixture' : 
                prob = (lamda * (tf/doc_lentgh)) + ((1-lamda) * cf_prob)  
            
            if prob == 0 :
                final_score_based_on_prob = float('-inf')
                break
            else : 
                final_score_based_on_prob += np.log2(prob)
        return final_score_based_on_prob

    def compute_scores_with_unigram_model(self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.        
        """
        query_tfs = self.get_query_tfs(query) 
        doc_scores = {}
        query_tfs = self.get_query_tfs(query) 
        query_docs = self.get_list_of_documents(query)
        for doc_id in query_docs : 
            doc_scores[doc_id] = self.get_unigram_model_score(query,query_tfs,doc_id,smoothing_method,None,alpha,lamda)
        return doc_scores


def main() : 
    with open('indexer/summaries indexer.json','r') as f : 
        index = json.load(f)
    with open('indexer/summaries_document_length_index.json') as f : 
        index_length = json.load(f)
    with open('indexer/summaries_cf_indexer.json') as f : 
        cf_index = json.load(f)
    
    with open('IMDB_crawled.json') as f : 
        documents = json.load(f)
    
    scorer = Scorer(index,len(documents),index_length,cf_index) 
    scores = scorer.compute_scores_with_unigram_model('gotham','mixture')
    #scores = scorer.compute_socres_with_okapi_bm25('fight club')
    print('-----------------------------------------------------------')
    print('document scores :')
    print(scores)
    mx = float('-inf')
    max_key = None
    for key in scores.keys() :
        if scores[key] > mx : 
            mx = scores[key] 
            max_key = key 
    print('-----------------------------------------------------------')
    print('high score document is:',max_key,'with score:', scores[max_key])
   # print('tt0137523', scores['tt0137523'])
if __name__ == '__main__':
    main()