import json 
import numpy as np
import string 
import nltk
from nltk.stem import WordNetLemmatizer
import re 
from typing import List
import tqdm 

def main():
    
    json_file_path = "../IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        documents = json.load(file)
    
    new_documents = []
    for movie in documents : 
        for key in movie.keys() : 
            if movie[key] == None : 
                if key == 'reviews' or 'summaries' or 'stars' or 'genres' : 
                    movie[key] = []
                else : 
                    movie[key] = ''
        new_documents.append(movie)
    with open('IMDB_crawled.json','w') as f : 
        json.dump(new_documents, f)
    print(len(new_documents))
if __name__ == '__main__':
    main()
