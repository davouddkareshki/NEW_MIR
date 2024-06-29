import numpy as np
import itertools
import random
import json
import tqdm 

class MinHashLSH:
    def __init__(self, documents, num_hashes, threshhold, bands, rows_per_band):

        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents

        self.shingles = []
        self.num_hashes = num_hashes
        self.all_shingles = []
        self.characteristic_matrix = None 
        self.signature = None 
        self.threshhold = threshhold 
        self.bands = bands
        self.rows_per_band = rows_per_band
         
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = [document[i:i+k] for i in range(len(document)-k+1)]
        return set(shingles)

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        for doc in self.documents : 
            shingles_of_doc = self.shingle_document(doc)
            self.shingles.append(shingles_of_doc)
            self.all_shingles = set(self.all_shingles).union(shingles_of_doc) 
        self.all_shingles = list(self.all_shingles)
        random.shuffle(self.all_shingles)
       # print(self.all_shingles)

        self.characteristic_matrix = np.zeros((len(self.all_shingles),len(self.documents)))
        for i,shingle in enumerate(self.all_shingles) : 
            for j in range(len(self.documents)) :
                if shingle in self.shingles[j] : 
                    self.characteristic_matrix[i][j] = 1 
        return 

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        self.build_characteristic_matrix() 
        #TODO : permute better 
        all_permutation = [np.random.permutation(len(self.all_shingles)) for _ in range(self.num_hashes)] 
        signature_matrix = np.zeros(((self.num_hashes),len(self.documents)))
        for i, permutation in enumerate(all_permutation) : 
            for j in range(len(self.documents)) : 
                for k,pos in enumerate(permutation) : 
                    if self.all_shingles[pos] in self.shingles[j]:
                        signature_matrix[i][j] = k
                        break
        self.signature = signature_matrix
        return
    
    def arr_hash(self, arr) :
        hsh = [0,0,0,0,0,0,0]  
        prime = (3,7,13,31,43,97,137)  
        mod = (1000004249,1000000207,1000000181,1000004119,int(1e9+7),int(1e9+9),int((2**31)-1)) 
        for k in range(len(hsh)) :
            mode_pow_prime = 1 
            for val in arr : 
           #     print('khar', val, hsh[k], mode_pow_prime, mod[k])
         #       hsh[k] = 123
                hsh[k] = (int(hsh[k]) + (int(val) * int(mode_pow_prime)) % int(mod[k])) % int(mod[k])     
                mode_pow_prime = (mode_pow_prime * prime[k]) % mod[k]
          #  print(mode_pow_prime)
        #print(hsh)
        return tuple(hsh) 
    
    def lsh_buckets(self, signature):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        bands = self.bands 
        rows_per_band = self.rows_per_band 

        all_buckets = {}
        idx = 0
      #  candidate_pairs = set()
        for k in range(bands) : 
            hsh_col_pair = []
            for j in range(len(self.documents)) : 
                arr = []
                for i in range(rows_per_band) : 
                    arr.append(signature[i+k*rows_per_band][j]) 
                hsh = self.arr_hash(arr) 
                hsh_col_pair.append((hsh,j)) 
            hsh_col_pair = sorted(hsh_col_pair)

            prv = None 
            bucket = []
            for val in hsh_col_pair :
                do = 1
                if prv != None : 
                    if prv[0] == val[0] : 
                        #candidate_pairs.add((min(prv[1],val[1]), max(prv[1],val[1])))  
                        bucket.append(val[1])
                        do = 0 
                if do : 
                    if len(bucket) > 1 : 
                        idx += 1
                        all_buckets[idx] = bucket
                        '''
                        if len(bucket) > 1 :
                #            print('bucket :')
                            for ids in bucket : 
                                arr = []
                                for i in range(rows_per_band) : 
                                    arr.append(signature[i+k*rows_per_band][ids]) 
                        #        print(arr, self.arr_hash(arr))
                   #             print(signature[:,ids])
                        '''
                    bucket = []
                    bucket.append(val[1])
                prv = val 
            if len(bucket) > 1 :
                idx += 1
                all_buckets[idx] = bucket
            '''
            if len(bucket) > 1 :
                print('bucket :')
                for ids in bucket : 
                    arr = []
                    for i in range(rows_per_band) : 
                        arr.append(signature[i+k*rows_per_band][ids]) 
                    print(arr)
            '''
        return all_buckets

    def calculate_jaccard_score_with_signature_matrix(self, signature,  first_set_id, second_set_id):
        score = 0 
        for i in range(self.num_hashes) : 
            if signature[i][first_set_id] == signature[i][second_set_id] : 
                score += 1 
        score /= self.num_hashes 
        return score 

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        simillar_doc_ids = set()
        self.min_hash_signature()
        buckets = self.lsh_buckets(self.signature)
        
  #      jac = self.calculate_jaccard_score_with_signature_matrix(signature,2,3)
   #     print(jac) 
        for key in buckets.keys() :
            for i in buckets[key] : 
                for j in buckets[key] :
                    if j > i :         
                        if self.calculate_jaccard_score_with_signature_matrix(self.signature,i,j) >= self.threshhold : 
                            simillar_doc_ids.add((i,j))
                    #print('same bucjet :',i,j,self.jaccard_score(self.shingles[i],self.shingles[j]), self.calculate_jaccard_score_with_signature_matrix(signature,i,j)) 
        return simillar_doc_ids 

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
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
       # print(len(A))
       # print(len(B))
        while (iter_A < len(A)) and (iter_B < len(B)) :
       #     print(iter_A, iter_B)
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
        return sim / len(universal_set) 
    
    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

      #  print(len(buckets))
        for bucket_id in tqdm.tqdm(buckets.keys()):
         #   print(bucket_id)
            docs_in_this_bucket = buckets[bucket_id]
            #print(docs_in_this_bucket)
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    
                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0
       #             print(first_doc_id, second_doc_id, near_duplicated_jaccard_score, self.calculate_jaccard_score_with_signature_matrix(self.signature, first_doc_id, second_doc_id))
        #            print(self.signature[:,first_doc_id])
         #           print('---')
          #          print(self.signature[:,second_doc_id])

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

def main():
    json_file_path = "IMDB_crawled.json"
    json_file_path_fake = "LSHFakeData.json"
    with open(json_file_path, "r") as file:
        data = json.load(file)
    with open(json_file_path_fake, "r") as file:
        data_fake = json.load(file)
        
    documents = []
    
    sm = 0 
    mx = 0
    for movie in data_fake : 
        doc_str = movie['summaries'] 
        final_str = ''
        for s in doc_str : 
            final_str = final_str + s
        sm += len(final_str) 
        mx = max(mx,len(final_str))
        documents.append(final_str) 
    
    for movie in data : 
        doc_str = movie['summaries'] 
        final_str = ''
        for s in doc_str : 
            final_str = final_str + s
        sm += len(final_str) 
        mx = max(mx,len(final_str))
        documents.append(final_str) 
    
    print('num of data :', len(documents))
    print('max len :', mx) 
    print('sum of len:', sm)
    model = MinHashLSH(documents, 2000, 0.8, 100, 20)


    duplicates = model.perform_lsh() 
    
    '''
    print("******")    
    for j in range(10) :
        i = 2*j 
        print(model.jaccard_score(model.shingles[i],model.shingles[i+1]))
    print("******")
    ''' 

    print('first 20 data are fake') 
    print('other data are real data')
    print('duplicate detected :', duplicates)
    print('test :') 
    #print(model.min_hash_signature())
    model.jaccard_similarity_test(model.lsh_buckets(model.signature), model.documents)
if __name__ == '__main__':
    main()
