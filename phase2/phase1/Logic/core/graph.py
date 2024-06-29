import networkx as nx

class LinkGraph:
    """
    Use this class to implement the required graph in link analysis.
    You are free to modify this class according to your needs.
    You can add or remove methods from it.
    """
    def __init__(self):
        self.nodes = set()
        self.edges = set() 
        self.adj = {}
        self.pre_adj = {}
        pass

    def add_node(self, node_to_add):
        if node_to_add in self.nodes :
            return
        self.nodes.add(node_to_add)
        self.adj[node_to_add] = set()
        self.pre_adj[node_to_add] = set()
        pass

    def add_edge(self, u_of_edge, v_of_edge):
        if (u_of_edge, v_of_edge) in self.edges : 
            return 

        self.edges.add((u_of_edge,v_of_edge))
        
        if u_of_edge not in self.adj : 
            self.adj[u_of_edge] = set()
        self.adj[u_of_edge].add(v_of_edge)
        
        if v_of_edge not in self.pre_adj :
            self.pre_adj[v_of_edge] = set() 
        self.pre_adj[v_of_edge].add(u_of_edge)
        pass


    def get_successors(self, node):
        return self.adj[node]
       
    def get_predecessors(self, node):
        return self.pre_adj[node] 