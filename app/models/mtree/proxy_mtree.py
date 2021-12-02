import mtree
from abc import ABC, abstractmethod

class IMtree(ABC):
    @abstractmethod
    def create_tree(self, distance_function, max_size):
        """ Function to implement """

    @abstractmethod
    def add_to_tree(self, node):
        """ Function to implement """

    @abstractmethod
    def add_all_to_tree(self, nodeList):
        """ Function to implement """

    @abstractmethod
    def knn_search(self, node, k):
        """ Function to implement """


class CMtree():
    def __init__(self):
        self.struct = None

    def create_tree(self, distance_function, max_size):
        self.struct = mtree.MTree(distance_function, max_node_size=max_size)

    def add_to_tree(self, node):
        self.struct.add(node)

    def add_all_to_tree(self, nodeList):
        # Multi dim
        size = len(nodeList)
        for i in range(size): self.struct.add(nodeList[i])
        
        # Uni dim
        # self.struct.add_all(nodeList)

    def knn_search(self, node, k):
        # found = None
        found = self.struct.search(node, k)
        return found

class Mtree():
    def __init__(self):
        self.tree = CMtree()
        self.node_size = 0
        # self.struct = None

    def create_tree(self, distance_function, max_size):
        if max_size < 1:
            print("Alert: Max node size need to be bigger than one.")
        else:
            self.tree.create_tree(distance_function, max_size)
            self.node_size = max_size

    def add_to_tree(self, node):
        self.tree.add_to_tree(node)
        print("Added", node, "to tree.")

    def add_all_to_tree(self, nodeList):
        self.tree.add_all_to_tree(nodeList)
        print("Added", nodeList, "to tree.")

    def knn_search(self, node, k):
        if (k < 1): 
            print("Alert: Search for", k, "elements is not possible.")
            return None
        found = self.tree.knn_search(node, k)
        return found