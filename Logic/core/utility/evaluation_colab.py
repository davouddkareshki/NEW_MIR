#RUN THIS CODE ON GOOGLE COLAB 
import numpy as np 
from typing import List
import wandb

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        n = len(actual)
        for i in range(n):
            actual_set = set(actual[i])
            predicted_set = set(predicted[i])
            intersection = actual_set.intersection(predicted_set)
            precision += len(intersection) / len(predicted_set) if len(predicted_set) > 0 else 0
        precision /= n
        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        for i in range(len(actual)):
            actual_set = set(actual[i])
            predicted_set = set(predicted[i])
            intersection = actual_set.intersection(predicted_set)
            recall += len(intersection) / len(actual_set) if len(actual_set) > 0 else 0
            
        recall /= len(actual)
        return recall

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1


    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """

        all_AP = [] 
        for i in range(len(actual)):
            actual_set = actual[i]
            predicted_set = predicted[i]
            num_correct = 0
            precision_sum = 0.0
          #  print('-----------------')
           # print(actual_set)
            #print(predicted_set)
            #print('answers :')

            number = 0
           # print('[-----]')
            for j, item in enumerate(predicted_set):
                if item in actual_set:
                    num_correct += 1
                    precision = num_correct / (j + 1)
                    precision_sum += precision
                    number += 1
            #        print(precision)
                  #  print(j, precision)
           # print('[---]')
            AP = precision_sum / number if number > 0 else 0
    #        print(precision_sum,number,AP) 
     #       print('----') 
            all_AP.append(AP)
      #  print(all_AP)
        return all_AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = np.mean(self.calculate_AP(actual, predicted)) 
        return MAP
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """

        all_DCG = []
        for i in range(len(actual)):
            actual_set = actual[i]
            predicted_set = predicted[i]
            #TODO: is there any better relevance score that we have or we can achieve? 
            relevance_scores = [1 if item in actual_set else 0 for item in predicted_set]
           
           # print(relevance_scores)
            DCG = 0.0
            for j in range(len(predicted_set)) : 
          #      print(2 ** relevance_scores[j] - 1, j+2, np.log2(j+2))
                DCG += (2 ** relevance_scores[j] - 1) / (np.log2(j + 2))
            all_DCG.append(DCG) 
        return all_DCG
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        all_DCG = self.cacluate_DCG(actual,predicted)
        all_NDCG = [] 
        for i in range(len(actual)):
            actual_set = actual[i]
            predicted_set = predicted[i]
            DCG = all_DCG[i]
          
            relevance_scores = [1 if item in actual_set else 0 for item in predicted_set]
            ideal_relevance_scores = sorted(relevance_scores, reverse=True)
            ideal_DCG = 0 
            for j in range(len(predicted_set)) : 
                ideal_DCG += (2 ** ideal_relevance_scores[j] - 1) / (np.log2(j + 2)) 
            NDCG = DCG / ideal_DCG if ideal_DCG > 0 else 0
            all_NDCG.append(NDCG) 
        average_NDCG = np.mean(all_NDCG)
        return average_NDCG 
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """

        all_RR = []
        for i in range(len(actual)):
            actual_set = actual[i]
            predicted_set = predicted[i]
            RR = 0
            for i,pred in enumerate(predicted_set) : 
                if pred in actual_set : 
                    RR = 1/(i+1) 
                    break 
            all_RR.append(RR) 
        return all_RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        return np.mean(self.cacluate_RR(actual,predicted))
    

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")
        print("Evaluation metrics:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")
        print(f"Average Precision: {ap}")
        print(f"Mean Average Precision: {map}")
        print(f"DCG: {dcg}")
        print(f"NDCG: {ndcg}")
        print(f"RR: {rr}")
        print(f"MRR: {mrr}")
      

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        wandb.init()
        wandb.log({
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
       #     'Average Precision': ap,
            'Mean Average Precision': map,
        #    'DCG': dcg,
            'NDCG': ndcg,
            'RR': rr,
            'MRR': mrr
        })

        return 


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)

if __name__ == '__main__':
    print('Run this code on google colab platform')
    evaluation = Evaluation('Davood Kareshki')
    '''
    query = 'fight club' 
    search_engine = SearchEngine()
    weights = {
        Indexes.STARS.value: 1,
        Indexes.GENRES.value: 1,
        Indexes.SUMMARIES.value: 1
    }
    predicted = search_engine.search(query, 'OkapiBM25', weights, safe_ranking = True, max_results=10)
    actual = search_engine.search(query, 'OkapiBM25', weights, safe_ranking = True, max_results=-1)
    '''
    
    '''
    #test 1 - slide 31 - AP|MAP test
    actual = [['1','2','3','4','5','6'],['1','2','3','4','5','6']]
    predicted = [['1','7','2','3','4','5','8','9','10','6'],['7','1','8','9','2','3','4','10','5','6']]
    '''
    '''
    #test 2 - slide 32 - AP|MAP test
    
    actual = [['1','2','3','4','5'],['1','2','3']]
    predicted = [['1','7','2','8','9','3','10','11','4','5'],['7','1','8','9','2','10','3','11','12','13']]
    '''
    
    '''
    #test 3 - khozabal
    actual = [['1','2','3']] 
    predicted = [['10','20','30','40','1','2','3']]
    '''
    '''
    #test 3 - DCG|NDCG test 
    actual = [['1'],['1']] 
    predicted = [['0','0','1','1','1'],['1','0','1','0','1']]
    ''' 

    # actual tests for 4 queries 
    # - fight club - kung fu - 007 - gotham
    # first 5 answer : 

    methods = ['ltn.lnn','ltc.lnc','OkapiBM25'] 
    for method in methods : 
        print('-----------------------------------------------------------')
        print(f'metrics of {method} method for test queries (fight club - kung fu - 007 - gotham) are :')
        print('note : actual data are first 5 movie that IMDB search engine retrieved for given query')
        print()

        if method == 'ltn.lnn' :      
            predicted_1 = ['tt0137523', 'tt1602613', 'tt3783958', 'tt0180052', 'tt0227445']
            predicted_2 = ['tt2267968', 'tt1302011', 'tt0070034', 'tt21692408']
            predicted_3 = ['tt0381061', 'tt1074638']
            predicted_4 = ['tt0103776', 'tt0118688', 'tt0372784', 'tt0112462', 'tt0096895']

        if method == 'ltc.lnc' :      
            predicted_1 = ['tt0137523', 'tt0039417', 'tt1020558', 'tt0082198', 'tt0111438']
            predicted_2 = ['tt2267968', 'tt1302011', 'tt0070034', 'tt21692408']
            predicted_3 = ['tt1074638', 'tt0381061']
            predicted_4 = ['tt7286456', 'tt0112462', 'tt0096895', 'tt1345836', 'tt0103776']

        if method == 'OkapiBM25' :      
            predicted_1,predicted_2,predicted_3,predicted_4 = (['tt0137523', 'tt3802576', 'tt0399102', 'tt0444182', 'tt0208092'], 
                                                               ['tt2267968', 'tt1302011', 'tt21692408', 'tt0070034'], 
                                                               ['tt1074638', 'tt0381061'], 
                                                               ['tt0103776', 'tt0118688', 'tt0372784', 'tt0096895', 'tt1345836'])
   
        actual = [['tt0137523', 'tt15561202', 'tt4622122', 'tt6107818', 'tt1196197'],
                  ['tt21692408', 'tt0441773', 'tt2267968', 'tt1302011', 'tt0373074'],
                  ['tt19049680', 'tt0830515', 'tt2382320', 'tt1074638', 'tt0381061'],
                  ['tt3749900', 'tt16418896', 'tt0095246', 'tt1345836', 'tt24223450']]
        predicted = [predicted_1,predicted_2,predicted_3,predicted_4]
        
        evaluation.calculate_evaluation(actual,predicted)
        print()
        
    pass 

