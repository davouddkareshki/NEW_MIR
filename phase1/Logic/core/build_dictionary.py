import json 
import preprocess
    
def main() :
    json_file_path = "IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        data = json.load(file)
    all_documents = [] 
    for movie in data : 
      #  for lis in movie['reviews'] : 
       #     for st in lis : 
        #        all_documents.append(st)
        for st in movie['summaries'] : 
            all_documents.append(st) 
        for st in movie['genres'] : 
            all_documents.append(st) 
        for st in movie['stars'] : 
            all_documents.append(st) 

    preprocesser = preprocess.Preprocessor(None) 
    new_data = []
    for text in all_documents : 
        text = preprocesser.remove_links(text) 
        text = preprocesser.remove_punctuations(text) 
        text = preprocesser.remove_additional_space(text)
        text = text.lower() 
        for word in preprocesser.tokenize(text) : 
            new_data.append(word) 
    print('pending...')
    json_file = [{'dictionary':new_data}]  
    with open('dictionary.json', "w") as file:
        json.dump(json_file,file)
if __name__ == '__main__':
    main()
