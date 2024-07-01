import json 

class SpellCorrection:
    def __init__(self, dictionary):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        dictionary : list of str
            The input documents.
        """
        self.dictionary = dictionary
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(dictionary)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        
        shingles = [word[i:i+k] for i in range(len(word)-k+1)]
        return set(shingles)
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        if len(first_set) == 0 or len(second_set) == 0 : 
            return 0

        universal_set = first_set.union(second_set)
        A = sorted(list(first_set))
        B = sorted(list(second_set)) 
        
        iter_A = 0 
        iter_B = 0 
        sim = 0 
        while (iter_A < len(A)) and (iter_B < len(B)) :
            if A[iter_A] == B[iter_B] : 
                sim += 1 
                iter_A += 1 
                iter_B += 1 
                continue
            if A[iter_A] < B[iter_B] : 
                iter_A += 1
                continue 
            if A[iter_A] > B[iter_B] : 
                iter_B += 1
                continue
        '''
        if A == B :
            print('in spell correction')
            print(A)
            print(B)
            print(sim/len(universal_set))
            print('---')
        ''' 
        return sim / len(universal_set) 

    def shingling_and_counting(self, dictionary):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        dictionary : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        for word in dictionary : 
            all_shingled_words[word] = self.shingle_word(word)  
            if word not in word_counter : 
                word_counter[word] = 0 
            word_counter[word] = word_counter[word] + 1 
        
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = []
        top5_candidates_jac = []

        shingles = self.shingle_word(word) 
        for docs_word in self.all_shingled_words : 
            jac_score = self.jaccard_score(shingles,self.all_shingled_words[docs_word])
            if len(top5_candidates) < 5 :
                top5_candidates.append(docs_word) 
                top5_candidates_jac.append(jac_score)
            else : 
                min_idx = 0 
                for i in range(5) : 
                    if top5_candidates_jac[i] < top5_candidates_jac[min_idx] : 
                        min_idx = i 
                    if top5_candidates_jac[i] == top5_candidates_jac[min_idx] : 
                        if self.word_counter[top5_candidates[i]] < self.word_counter[top5_candidates[min_idx]] : 
                            min_idx = i
              #  if docs_word == 'while' : 
               #     print('***', top5_candidates[i], top5_candidates_jac[i], ':::' ,jac_score, '***')
                if jac_score > top5_candidates_jac[min_idx] : 
              #      if top5_candidates[min_idx] == 'while' : 
             #           print('***', top5_candidates[min_idx], top5_candidates_jac[min_idx], ':::' ,docs_word, jac_score, 'in','***')
                    top5_candidates[min_idx] = docs_word 
                    top5_candidates_jac[min_idx] = jac_score 
   #     print(top5_candidates)
    #    print(top5_candidates_jac)
        return top5_candidates,top5_candidates_jac
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        query_words = query.split(' ')
        
        final_result = ''
        for word in query_words : 
            if word in self.dictionary : 
                if len(final_result) == 0 : 
                    final_result = word 
                else : 
                    final_result = final_result + ' ' + word
                continue

            top5_candidates,top5_candidates_score = self.find_nearest_words(word)
     #       print(word) 
      #      print(top5_candidates)
            max_tf = 0
            for candidate in top5_candidates :
                tf = self.word_counter[candidate]
                if tf > max_tf :  
                    max_tf = tf 
                
            for i in range(5) : 
                top5_candidates_score[i] *= self.word_counter[top5_candidates[i]] / max_tf

            final_word = top5_candidates[0] 
            max_idx = 0
            for i,candidate in enumerate(top5_candidates) :
                if top5_candidates_score[i] > top5_candidates_score[max_idx] : 
                    final_word = candidate
                    max_idx = i 
            
            '''
            print("candidates for word ", word, "are :")
            print(top5_candidates) 
            print("with scores :")
            print(top5_candidates_score)
            print('---')
            '''
            if len(final_result) == 0 : 
                final_result = final_word 
            else : 
                final_result = final_result + ' ' + final_word
        return final_result
    
def main() :
    json_file_path = "dictionary.json"
    with open(json_file_path, "r") as file:
        dictionary = json.load(file)
   # dictionary = dictionary['dictionary']
    dictionary = dictionary[0]['dictionary']
    print('preprocess pending...')
    spell_correction = SpellCorrection(dictionary)
   # print(spell_correction.all_shingled_words['while'])
   # print(spell_correction.word_counter['while'])
   # print(spell_correction.jaccard_score(spell_correction.shingle_word('while'), spell_correction.shingle_word('whle')))
    print('query pending ...')
    query = 'whle'
    print(spell_correction.spell_check(query))
if __name__ == '__main__':
    main()
