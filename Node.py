class Node:
    def __init__(self, node_id, node_type, content = None, source = None, edges = None):
        self.node_id = node_id
        self.node_type = node_type
        self.source = source
        self.edges = edges or {}
        self.content = content
        self.degree = self.getDegree()

    def link(self,node, weight = 1):
        if node.node_id not in self.edges:
            self.edges[node.node_id] = weight
        else:
            self.edges[node.node_id] += weight
        self.degree = self.getDegree()

    def print(self):
        print(f"ID: {self.node_id}")
        print(f"Type: {self.node_type}")
        print(f"Source: {self.source}")
        print(f"Edges: {self.edges}")
        print(f"Content: {self.content}")
        print(f"Degree: {self.degree}")
        
    def getDegree(self):
        return sum(self.edges.values())
        