import preprocess
import json 
import numpy as np 

class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.preprocessor = preprocess.Preprocessor([]) 


    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        return self.preprocessor.remove_stopwords(query) 

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        #preprocessed_doc_tokens = self.preprocessor.tokenize(self.preprocessor.process_text(doc))
        doc_tokens = self.preprocessor.tokenize(doc)
        query = self.preprocessor.process_text(self.remove_stop_words_from_query(query))
        query_tokens = self.preprocessor.tokenize(query)

        query_positions = []
        '''
        for token in query_tokens:
            if token in preprocessed_doc_tokens:
                query_positions.append(preprocessed_doc_tokens.index(token))
            else:
                not_exist_words.append(token)
        ''' 
        is_in_doc = np.zeros(len(query_tokens))
        for idx, word in enumerate(doc_tokens) : 
         #   if word[0:4] == 'Bond' :
          #      print(word, self.preprocessor.process_text(word))
            #khar = 0
           # if word == 'partner' : 
            #    print(word)
            #    khar = 1
            word = self.preprocessor.process_text(word)
            #if khar : 
             #   print(word)
            flag = 0
            for i,token in enumerate(query_tokens) : 
                if token == word : 
                    flag = 1
                    is_in_doc[i] = 1
            if flag: 
            #    print(word) 
      #          print(doc_tokens[idx])
                query_positions.append(idx)   
               # is_in_doc[query_tokens.index(word)] = True  
        
        for i,token in enumerate(query_tokens) : 
            if is_in_doc[i] != True : 
                not_exist_words.append(token)
        #print(query_positions)
        last_start_index = 0 
        last_end_index = 0 

        # FIND CORNER CASE 
        for position in query_positions : 
            start_index = max(0, position - self.number_of_words_on_each_side)
            end_index = min(len(doc_tokens), position + self.number_of_words_on_each_side + 1)
            window_words = doc_tokens[start_index:end_index]
  #          print(window_words)
            set_window_words = set([self.preprocessor.process_text(word) for word in window_words]) 
            set_query_tokens = set(query_tokens)
   #         print(set_query_tokens) 
    #        print(set_window_words) 
     #       print('-------') 
            if set_query_tokens.issubset(set_window_words) :

                snippet = ""
                idx = start_index
                for token in window_words :
                    if idx in query_positions :
                        snippet += " ***" + token + "*** "
                    else:
                        snippet += token + ' '
                    idx += 1 
                final_snippet += self.preprocessor.remove_additional_space(snippet) + " ... "
                final_snippet = final_snippet.strip() 
                return final_snippet, not_exist_words

        for position in query_positions:
            start_index = max(0, position - self.number_of_words_on_each_side)
            end_index = min(len(doc_tokens), position + self.number_of_words_on_each_side + 1)
     #       print(start_index, end_index, last_start_index, last_end_index, position, doc_tokens[position])
            if last_end_index > position :
          #      print('khar', position) 
                continue 
            if last_end_index > start_index : 
                start_index = last_end_index 
                final_snippet = final_snippet[0:len(final_snippet)-5]
       #     print('here')
            last_start_index = start_index 
            last_end_index = end_index 

            snippet_tokens = doc_tokens[start_index:end_index]

            snippet = ""
            idx = start_index
            for token in snippet_tokens :
                if idx in query_positions :
                    snippet += " ***" + token + "*** "
                else:
                    snippet += token + ' '
                idx += 1 
            final_snippet += self.preprocessor.remove_additional_space(snippet) + " ... "
     #   print('final :', final_snippet)
        final_snippet = self.preprocessor.remove_additional_space(final_snippet)
     #   print('removed :', final_snippet)
     #   print(final_snippet)
    #    print('-----')
     #   print(doc)
      #  print('--')
       # print(final_snippet)
        #print('-----')
 #       print('data :', doc, query)
  #      print('in snippet :', final_snippet)
        return final_snippet, not_exist_words

def main() :
    json_file_path = "IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        documents = json.load(file)
    
    doc = documents[0]['first_page_summary'] 
    for movie in documents : 
        if movie['id'] == 'tt0381061' : 
            doc = movie['first_page_summary']
            break 

    print(doc) 
    snippet = Snippet()
  #  final_snippet, not_exist_words = snippet.find_snippet(doc,'007 Bond') 
    final_snippet, not_exist_words = snippet.find_snippet(doc,'earning terrorists mission') 
    print('final snippet :')
    print(final_snippet) 
    print('not exist words :') 
    print(not_exist_words)
    pass 

if __name__ == '__main__':
    main()
