from typing import List, Tuple
import random

from structure import Relationship

class GlobalGraph:
    """A helper to maintain global graphs of relationships and contrasts."""
    def __init__(self, relationships: List[Relationship]) -> None:
        """Initialize graphs from provided relationships."""
        self.relationships: List[Relationship] = relationships
        self.graph, self.nodes = self.initialize_graph(relationships)
        self.contrast_relations, self.other_relations = self.split_relationships_by_contrast(relationships)
        self.contrast_graph = self.initialize_graph(self.contrast_relations)[0]

    def initialize_graph(self, relationships: List[Relationship]) -> tuple[dict[str, set[str]], set[str]]:
        """Build adjacency map and node set from relationships."""
        graph: dict[str, set[str]] = {}
        nodes: set[str] = set()
        for rel in relationships or []:
            u = getattr(rel, "assertion1_id", None)
            v = getattr(rel, "assertion2_id", None)
            if u is None or v is None:
                continue
            nodes.add(u)
            nodes.add(v)
            graph.setdefault(u, set()).add(v)
            graph.setdefault(v, set())

        return graph, nodes


    def split_relationships_by_contrast(self, relationships: List[Relationship]) -> Tuple[list[Relationship], list[Relationship]]:
        """
        Split a list of Relationship objects into two lists:
        - contrast_relations: relationships where relationship_type == "contrast"
        - other_relations: relationships with any other type
        """
        if not relationships:
            return [], []

        contrast_relations: list[Relationship] = []
        other_relations: list[Relationship] = []

        for rel in relationships:
            rel_type = getattr(rel, "relationship_type", None)
            if rel_type == "contrast":
                contrast_relations.append(rel)
            else:
                other_relations.append(rel)

        return contrast_relations, other_relations

    def find_nodes_in_scc(self) -> list[list[str]]:
        """
        Given a list of relationships interpreted as a directed graph (assertion1_id -> assertion2_id),
        return all node IDs that belong to at least one Strongly Connected Component (SCC).

        Rules:
        - Nodes that form a component of size >= 2 are included.
        - A single node with a self-loop (edge u->u) also counts as an SCC and should be included.
        - If no SCCs exist, return an empty list.
        - The result is returned as a sorted list of unique node IDs for deterministic behavior.
        """
        # Tarjan's algorithm
        index = 0
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        components: List[List[str]] = []

        def strongconnect(v: str):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in self.graph.get(v, ()):  # neighbors
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            # If v is a root node, pop and form an SCC
            if lowlink[v] == indices[v]:
                component: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == v:
                        break
                components.append(component)  # collect every SCC

        # Run Tarjan
        for node in self.graph.keys():
            if node not in indices:
                strongconnect(node)

        return components

    def resolve_cycle_by_user(self, cycle :List[str]) -> str:
        #TODO should send cycle to user and get user's decision node
        pass
    
    def resolve_contrast_by_user(self, node: str) -> bool:
        #TODO should ask user if they want to delete it or its neighbors
        pass

    def automatic_or_manual(self):
        #TODO ask user if they want to resolve conflicts automatically or manually
        pass

    def remove_node(self, node):
        self.nodes.remove(node)
        self.graph.pop(node, None)
        for other_node in self.graph:
            self.graph[other_node].discard(node)
        self.contrast_graph.pop(node, None)
        for other_node in self.contrast_graph:
            self.contrast_graph[other_node].discard(node)
        for relation in self.contrast_relations:
            if relation.assertion1_id == node or relation.assertion2_id == node:
                self.contrast_relations.remove(relation)
        for relation in self.relationships:
            if relation.assertion1_id == node or relation.assertion2_id == node:
                self.relationships.remove(relation)


    def pick_worst_node(self, nodes_part_of_scc):
        min_score = 100000000
        min_node: str | None = None
        for node in self.nodes:
            if node not in nodes_part_of_scc:
                continue
            if node not in self.contrast_graph:
                continue
            score = len(self.graph[node]) - 2 * len(self.contrast_graph[node])
            if score < min_score:
                min_score = score
                min_node = node
        return min_node

    def resolve_cycles_and_conflicts(self) -> None:
        """Resolve cycles using contrast penalty heuristics (placeholder).
        
        Currently computes a simple score per node inside SCCs (out_degree - 2*contrast_out)
        and would select a candidate node to act upon. The concrete mutation of the
        graph/relations is not implemented yet.
        """
        # ask for user mode
        automatic = self.automatic_or_manual()

        list_of_scc = self.find_nodes_in_scc()
        nodes_part_of_scc = [node for sublist in list_of_scc for node in sublist]

        while len(nodes_part_of_scc) > 0 and len(self.contrast_relations) > 0:

            min_node = self.pick_worst_node(nodes_part_of_scc)

            if self.contrast_graph is None:
                if automatic:
                    remove_nodes = set(random.choice(nodes_part_of_scc))
                else:
                    remove_nodes = set(self.resolve_cycle_by_user(list_of_scc[0]))
            else:
                if automatic:
                    remove_nodes = set(min_node)
                else:
                    if self.resolve_contrast_by_user(min_node):
                        remove_nodes = set(min_node)
                    else:
                        remove_nodes = set(self.contrast_graph[min_node])

            for node in remove_nodes:
                self.remove_node(node)

            list_of_scc = self.find_nodes_in_scc()
            nodes_part_of_scc = self.find_nodes_in_scc()
