from typing import Dict, List
from search import SearchEngine
from spell_correction import SpellCorrection
from snippet import Snippet
from indexer.indexes_enum import Indexes, Index_types
from preprocess import Preprocessor
import json

#print('preprocess pending...')

with open('dictionary.json','r') as f :
    dictionary = json.load(f)
dictionary = dictionary[0]['dictionary']
with open('IMDB_crawled.json','r') as f :
    movies_dataset = json.load(f) 

search_engine = SearchEngine()
snippet = Snippet() 
spell_correction_obj = SpellCorrection(dictionary)

#print('query pending...')

def correct_text(text: str) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    dictionary : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
   # print('wtnf',text)
    preprocessor = Preprocessor(None)
    text = preprocessor.process_text(text)
    #print('wtf',text)
    text = spell_correction_obj.spell_check(text)
    #print('this', text) 
    #print('---')
    return text

def concat_strings(list_str : List[str]) : 
    text = '' 
    for st in list_str : 
        text = text +' ' + st 
    text.strip()
    return text 


def search(query, max_result_count, method = 'OkapiBM25', weights: list = [0.3, 0.3, 0.4], unigram_smoothing='mixture', alpha=0.5, lamda=0.5, should_print=False, preferred_genre: str = None):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    preferred_genre: A list containing preference rates for each genre. If None, the preference rates are equal.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    actual_query = query
    query = correct_text(query)
#    print(query)
    if max_result_count == -1 : 
        max_result_count = None

    arr = weights
    if preferred_genre != None : 
        weights = []
        arr[1] *= 3 
        query = query + ' ' + preferred_genre

    weights = {
        Indexes.STARS.value: arr[0],
        Indexes.GENRES.value: arr[1],
        Indexes.SUMMARIES.value: arr[2] 
    }    

    scores = search_engine.search(query, method, weights, safe_ranking=False, max_results=max_result_count, smoothing_method=unigram_smoothing, alpha=alpha, lamda=lamda)

    relavent_oredered_doc_ids = []
    output = []
    for val in scores : 
        doc_id = val[0]
        relavent_oredered_doc_ids.append(doc_id)
        movie = get_movie_by_id(doc_id, movies_dataset)
       # print(movie['summaries']) 
        snippet_of_query_1, not_exist_words_of_query = snippet.find_snippet(concat_strings(movie['summaries']),actual_query)  
        snippet_of_query_2, not_exist_words_of_query = snippet.find_snippet(concat_strings(movie['genres']),actual_query) 
        snippet_of_query_3, not_exist_words_of_query = snippet.find_snippet(concat_strings(movie['stars']),actual_query) 
        snippet_of_query = snippet_of_query_1 + snippet_of_query_2 + snippet_of_query_3 
        output.append((doc_id,snippet_of_query))
    
    if should_print :
        print(output)
    return scores

def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    '''
    #print(movies_dataset[0]['id'])
    result = movies_dataset.get(
        id,
        {
            "Title": "This is movie's title",
            "Summary": "This is a summary",
            "URL": "https://www.imdb.com/title/tt0111161/",
            "Cast": ["Morgan Freeman", "Tim Robbins"],
            "Genres": ["Drama", "Crime"],
            "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
        },
    )
    '''
    
    #TODO: we can make it better by sorting and binary search
    result = {} 
    for movie in movies_dataset : 
        if movie['id'] == id : 
            result = movie 
            break 

    result["Image_URL"] = (
        "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"  # a default picture for selected movies
    )
    result["URL"] = (
        f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
    )
    return result

#query = 'fight club' 
#search(query,5,should_print=True)
