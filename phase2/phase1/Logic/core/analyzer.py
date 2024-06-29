from graph import LinkGraph
#from ..indexer.indexes_enum import Indexes
#from ..indexer.index_reader import Index_reader
import numpy as np 
import json

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.N = 0
        self.graph = LinkGraph()
        self.hubs = {}
        self.authorities = {}
        self.base_set = set()
        self.movie_nodes = set()
        self.star_nodes = set()
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            self.graph.add_node(movie['id'])    
            self.movie_nodes.add(movie['id'])

            self.hubs[movie['id']] = 1
            self.authorities[movie['id']] = 1

            for star in movie['stars'] : 
                self.graph.add_node(star)
                self.star_nodes.add(star)

                self.hubs[star] = 1
                self.authorities[star] = 1

                self.graph.add_edge(star,movie['id'])
                self.graph.add_edge(movie['id'],star)
        pass

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        base_set = set()
        for movie in corpus:
            movie_id = movie['id']
            for star in movie['stars'] :
                #print(star)
                if movie in self.root_set : 
                    base_set.add(star) 
                    base_set.add(movie_id)
                    self.movie_nodes.add(movie_id) 
                    self.star_nodes.add(star)

                    self.hubs[star] = 1
                    self.authorities[star] = 1
                    self.hubs[movie_id] = 1
                    self.authorities[movie_id] = 1

                    self.graph.add_node(star)
                    self.graph.add_node(movie_id)
                    self.graph.add_edge(star,movie_id)
                    self.graph.add_edge(movie_id,star) 
        self.base_set = base_set 
        self.N = len(self.base_set) 
        pass 

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
   #     print(self.root_set)
       # print('base set : ',self.base_set)
        A = np.zeros((self.N,self.N))
        map_nodes = {}
        remap_nodes = {}
        idx = 0
        for node in self.base_set : 
            map_nodes[node] = idx 
            remap_nodes[idx] = node 
            idx += 1
        
        for edge in self.graph.edges : 
            if edge[0] not in map_nodes.keys() or edge[1] not in map_nodes.keys() : 
                continue
            u = map_nodes[edge[0]]
            v = map_nodes[edge[1]] 
            A[u][v] = 1 
        
        P = A 
        alpha = 0.1
        yek = np.ones(self.N) 
        for i in range(self.N) : 
            P[i] = (1-alpha) * P[i] + alpha * yek 
        
        h = np.ones(self.N)
        a = np.ones(self.N)
       # print(P.shape) 
       # print(self.N)
        #print(P)
        for _ in range(num_iteration) : 
            new_h = np.matmul(P,a) 
            new_a = np.matmul(np.transpose(P) ,h) 
            a = new_a 
            h = new_h 
        
        movie_scores = []
        for movie_id in self.movie_nodes : 
            if movie_id in self.base_set : 
        #        print(movie_id, a[map_nodes[movie_id]])
                movie_scores.append((a[map_nodes[movie_id]],movie_id))
        star_scores = []
        for star in self.star_nodes : 
            if star in self.base_set : 
                star_scores.append((a[map_nodes[star]], star))

        movie_scores = sorted(movie_scores, key=lambda x: x[0], reverse=True)
        star_scores = sorted(star_scores, key=lambda x: x[0], reverse=True) 
     #   print(movie_scores)
     #   print(star_scores)
        movie_scores = movie_scores[:max_result]
        star_scores = star_scores[:max_result]

        out_movies = [x[1] for x in movie_scores]
        out_stars = [x[1] for x in star_scores]

        return out_stars, out_movies

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    json_file_path = "IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        corpus = json.load(file)

    root_set = [corpus[0], corpus[1], corpus[2], corpus[4], corpus[12]]
  #  print(corpus[0])
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
