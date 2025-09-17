from functools import cmp_to_key
from typing import List, Tuple
import random

from structure import Relationship

class GlobalGraph:
    def __init__(self, relationships: List[Relationship]) -> None:
        """
        Initialize a graph-based data structure from a list of relationships.

        Args:
            relationships: A list of Relationship objects,
                    each representing a directed connection between two assertions
                    with fields `assertion1_id`, `assertion2_id`, and `relationship_type`.

        Description:
            Initializes the following internal structures:
                - relationship_for_pair (dict[Tuple[str, str], Relationship]): Maps assertion pairs to Relationship objects.
                - relationships (List[Relationship]): Original list of relationships.
                - good_graph_1 (dict[str, set[str]]): Adjacency list of the graph with edges going from first assertion to second assertion.
                                                      The graph is called good since it doesn't include any contradiction relationship_type edges.
                - good_graph_2 (dict[str, set[str]]): Adjacency list of the graph with edges going from second assertion to first assertion.
                                                      The graph is called good since it doesn't include any contradiction relationship_type edges.
                - bad_graph (dict[str, set[str]]): Adjacency list of the graph containing only edges with contradiction relationship_type.
                - nodes (set[str]): Set of all unique node IDs.
                - number_of_visited_parents (dict[str, int]): Tracks visited parent counts per assertion node. Used to determine if all parents have been visited.
                - ordered_graph (List[str]): Assertions in traversal or topological order and final desired output.
                - assertions_by_layers (dict[int, List[str]]): Assertions grouped by layers.
        """
        self.relationship_for_pair: dict[Tuple[str, str], Relationship] = {}
        self.relationships: List[Relationship] = relationships

        self.good_graph_1: dict[str, set[str]] = {}
        self.good_graph_2: dict[str, set[str]] = {}
        self.bad_graph: dict[str, set[str]] = {}
        self.nodes: set[str] = set()
        self.number_of_visited_parents: dict[str, int] = {}

        self.initialize_graph(relationships)
        self.ordered_graph: List[str] = []
        self.assertions_by_layers : dict[int, List[str]] = {}

    def initialize_graph(self, relationships: List[Relationship]):
        """
        Build adjacency maps and populate node sets from a list of relationships.

        Args:
            relationships (List[Relationship]): A list of Relationship objects.
                Each object should define:
                    - assertion1_id (str): The source node.
                    - assertion2_id (str): The target node.
                    - relationship_type (str): The type of relationship
                      (e.g., 'contradiction', 'cause', 'evidence', etc.).

        Description:
            Updates the following internal structures in place:
                - nodes (set[str]): Adds all unique assertion IDs encountered.
                - good_graph_1 (dict[str, set[str]]):
                    Builds forward adjacency lists for non-contradictory relationships.
                - good_graph_2 (dict[str, set[str]]):
                    Builds backward adjacency lists for non-contradictory relationships.
                - bad_graph (dict[str, set[str]]):
                    Adds symmetric edges for contradictory relationships.
                - number_of_visited_parents (dict[str, int]):
                    Initializes counters for each node with a default of 0.
                - relationship_for_pair (dict[Tuple[str, str], Relationship]):
                    Maps each (source, target) pair to the corresponding Relationship object.

        Notes:
            - If a relationship lacks `assertion1_id` or `assertion2_id`, it is skipped.
            - Contradiction relationships are treated as undirected edges
              (both directions added to bad_graph).
            - Other relationships are treated as directed edges.
            - Method modifies the graph in place and does not return anything.
        """
        for rel in relationships or []:
            u = getattr(rel, "assertion1_id", None)
            v = getattr(rel, "assertion2_id", None)
            if u is None or v is None:
                continue

            self.nodes.add(u)
            self.nodes.add(v)

            rel_type = getattr(rel, "relationship_type", None)

            if rel_type == "contradiction":
                self.bad_graph.setdefault(u, set()).add(v)
                self.bad_graph.setdefault(v, set()).add(u)
            else:
                self.good_graph_1.setdefault(u, set()).add(v)
                self.good_graph_1.setdefault(v, set())
                self.good_graph_2.setdefault(v, set()).add(u)
                self.good_graph_2.setdefault(u, set())

            self.number_of_visited_parents.setdefault(u, 0)
            self.number_of_visited_parents.setdefault(v, 0)
            self.relationship_for_pair[(u, v)] = rel

    def find_nodes_in_scc(self) -> list[list[str]]:
        """
        Given a list of relationships interpreted as a directed graph (assertion1_id -> assertion2_id),
        return all node IDs that belong to at least one Strongly Connected Component (SCC).

        Rules:
        - Nodes that form a component of size >= 2 are included.
        - A single node with a self-loop (edge u->u) also counts as an SCC and should be included.
        - If no SCCs exist, return an empty list.
        - The result is returned as a list of lists, where each inner list represents a single SCC.
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

            for w in self.good_graph_1.get(v, ()):  # neighbors
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
                if len(component) > 1:
                    components.append(component)  # collect every SCC

        # Run Tarjan
        for node in self.good_graph_1.keys():
            if node not in indices:
                strongconnect(node)

        return components

    def resolve_cycle_by_user(self, cycle: List[str]) -> str:
        """
        Ask the user to resolve a cycle in the graph by selecting an assertion to remove.

        Args:
            cycle (List[str]): A list of assertion IDs (nodes) that form a directed cycle.

        Returns:
            str: The assertion ID chosen by the user to remove, thereby breaking the cycle.

        Description:
            - Prompts the user (or external agent) to decide which node to remove.
        """
        #TODO should send cycle to user and get user's decision node
        pass

    def resolve_contradiction_by_user(self, node: str, other_nodes: List[str]) -> bool:
        """
        Ask the user to resolve a contradiction by choosing which nodes to remove.

        Args:
            node (str): The single assertion ID representing one side of the contradiction.
            other_nodes (List[str]): A list of assertion IDs representing the conflicting neighbors.

        Returns:
            bool:
                - True if the user decides to remove the single `node`.
                - False if the user decides to remove all nodes in `other_nodes`.

        Description:
            - Prompts the user to decide:
                * Keep `other_nodes` and delete `node`, or
                * Keep `node` and delete all `other_nodes`.
        """
        #TODO should ask user if they want to delete it or its neighbors
        pass

    def automatic_or_manual(self) -> bool:
        """
        Ask the user whether conflicts in the graph should be resolved automatically or manually.

        Args:
            None

        Returns:
            bool:
                - True if the user chooses automatic resolution.
                - False if the user chooses manual resolution.

        Description:
            - Determines how contradictions or cycles in the graph will be handled:
                * Manual: Each time a conflict arises, the user will be prompted to choose
                  which node(s) to remove.
                * Automatic: Conflicts will be resolved programmatically without further user input.
            - Provides a global setting that controls whether methods like resolve_cycle_by_user or resolve_contradiction_by_user will be used later.
        """
        #TODO ask user if they want to resolve conflicts automatically or manually
        return True

    def remove_node(self, node: str) -> None:
        """
        Remove a node and all its relationships from the graph.

        Args:
            node (str): The assertion ID of the node to remove.

        Returns:
            None

        Description:
            - Completely deletes the specified node from the graph and updates all
              associated structures to remain consistent.
            - Effects include:
                * Removes the node from the `nodes` set.
                * Deletes the node’s adjacency entries from `good_graph_1`, `good_graph_2`, and `bad_graph`.
                * Removes all relationships in `relationships` where the node is
                  either `assertion1_id` or `assertion2_id`.
                * Deletes all keys in `relationship_for_pair` involving the node.
                * Removes the node from other nodes’ adjacency sets in all graphs.

        Notes:
            - Safe to call even if the node does not exist; all removals use `discard`
              or `pop(..., None)` to avoid exceptions.
        """
        self.nodes.discard(node)
        self.good_graph_1.pop(node, None)
        self.good_graph_2.pop(node, None)
        self.bad_graph.pop(node, None)
        self.relationships = [
            rel for rel in self.relationships
            if rel.assertion1_id != node and rel.assertion2_id != node
        ]
        to_remove = [(a1, a2) for (a1, a2) in self.relationship_for_pair.keys() if a1 == node or a2 == node]
        for key in to_remove:
            self.relationship_for_pair.pop(key)
        for curr_node in self.good_graph_1:
            self.good_graph_1[curr_node].discard(node)
        for curr_node in self.good_graph_2:
            self.good_graph_2[curr_node].discard(node)
        for curr_node in self.bad_graph:
            self.bad_graph[curr_node].discard(node)
            if len(self.bad_graph[curr_node]) == 0:
                to_remove.append((curr_node, curr_node))
        for node, node in to_remove:
            self.bad_graph.pop(node, None)


    def pick_worst_node(self, nodes_part_of_scc) -> str:
        """
        Select the "worst" node from a set of nodes that are part of a strongly connected component (SCC).

        Args:
            nodes_part_of_scc (list[str]): A collection of assertion IDs that are part of some strongly connected component.

        Returns:
            str: The ID of the node considered the "worst" candidate to remove.

        Description:
            - Evaluates each node in the all SCC to determine which is "worst".
            - The scoring heuristic is:
                * score = out_degree + in_degree − contradiction_degree
                  where:
                    - out_degree = number of outgoing edges in `good_graph_1`
                    - in_degree = number of incoming edges in `good_graph_2`
                    - contradiction_degree = number of contradictory edges in `bad_graph`
                * A lower score indicates a worse node.
            - If multiple nodes tie for the lowest score, the one with more contradictions (`bad_graph` degree) is chosen.
            - If no node is part of SCCs, the method falls back to scanning all graph nodes
              that appear in `bad_graph` and picks the lowest-score candidate.

        Notes:
            - Always returns a node ID (string) if any exist in the graph.
            - Intended to automate conflict resolution by identifying nodes most detrimental.
        """
        min_score = float('inf')
        min_node: str | None = None
        bad_nodes = 0
        for node in nodes_part_of_scc:
            score = len(self.good_graph_1[node]) + len(self.good_graph_2[node]) - len(self.bad_graph.get(node, set()))
            if score < min_score:
                min_score = score
                min_node = node
                bad_nodes = len(self.bad_graph.get(node, set()))
            elif score == min_score:
                if len(self.bad_graph.get(node, set())) > bad_nodes:
                    min_score = score
                    min_node = node
                    bad_nodes = len(self.bad_graph.get(node, set()))

        if min_node is None:
            for node in self.nodes:
                if node in self.bad_graph:
                    score = len(self.good_graph_1[node]) + len(self.good_graph_2[node]) - len(self.bad_graph.get(node, set()))
                    if score < min_score:
                        min_score = score
                        min_node = node
        return min_node

    def resolve_cycles_and_conflicts(self) -> None:
        """
        Resolve cycles and contradictions in the graph using either automatic or manual mode.

        Args:
            None

        Returns:
            None

        Description:
            - Ensures the graph becomes acyclic and contradiction-free by repeatedly removing nodes.
            - Workflow:
                1. Ask the user (or system) whether resolution should be automatic or manual
                   using `automatic_or_manual()`.
                2. Detect strongly connected components (SCCs) via `find_nodes_in_scc()`.
                3. While there are nodes in SCCs or contradictions in `bad_graph`:
                    * If handling a cycle:
                        - Automatic mode: remove a random node from the SCC.
                        - Manual mode: ask the user to choose a node with `resolve_cycle_by_user()`.
                    * If handling a contradiction:
                        - Automatic mode: pick the worst node with `pick_worst_node()` and remove it.
                        - Manual mode: ask the user with `resolve_contradiction_by_user()` whether to
                          remove the node or its neighbors.
                    * Remove chosen nodes via `remove_node()`.
                4. Recompute SCCs and repeat until no cycles or contradictions remain.

        Notes:
            - The method modifies the graph in place and does not return a value.
            - Combines all conflict-resolution helpers (`automatic_or_manual`,
              `resolve_cycle_by_user`, `resolve_contradiction_by_user`, `pick_worst_node`, `remove_node`).
        """
        # ask for user mode
        automatic = self.automatic_or_manual()

        list_of_scc = self.find_nodes_in_scc()
        nodes_part_of_scc = [node for sublist in list_of_scc for node in sublist]
        while len(nodes_part_of_scc) > 0 or len(self.bad_graph) > 0:
            min_node = self.pick_worst_node(nodes_part_of_scc)
            if self.bad_graph is None:
                if automatic:
                    remove_nodes = [random.choice(nodes_part_of_scc)]
                else:
                    remove_nodes = [self.resolve_cycle_by_user(list_of_scc[0])]
            else:
                if automatic:
                    remove_nodes = [min_node]
                else:
                    if self.resolve_contradiction_by_user(min_node, list(self.bad_graph[min_node])):
                        remove_nodes = [min_node]
                    else:
                        remove_nodes = list(self.bad_graph[min_node])

            for node in remove_nodes:
                self.remove_node(node)

            list_of_scc = self.find_nodes_in_scc()
            nodes_part_of_scc = [node for sublist in list_of_scc for node in sublist]

    def find_all_starting_nodes(self) -> List[str]:
        """
        Find all starting nodes in the graph (nodes with no incoming edges).

        Args:
            None

        Returns:
            List[str]: A list of assertion IDs that have no parents (no incoming edges in `good_graph_2`).

        Description:
            - Iterates over all nodes in the graph.
            - For each node, checks its entry in `good_graph_2` (reverse adjacency map).
            - If the node has an empty parent set, it is considered a "starting node".
            - Starting nodes serve as entry points for later DAG traversal and
              topological ordering of assertions.
        """
        starter_nodes = []
        for node in self.nodes:
            if len(self.good_graph_2[node]) == 0:
                starter_nodes.append(node)
        return starter_nodes

    def check_used_all_parents(self, node) -> bool:
        """
        Check whether all parent nodes of a given node have been visited.

        Args:
            node (str): The assertion ID of the node to check.

        Returns:
            bool:
                - True if all parent nodes of `node` have been visited.
                - False if at least one parent node has not yet been visited.

        Description:
            - Uses `number_of_visited_parents[node]` to track how many parent nodes have been processed.
            - Compares it with `len(self.good_graph_2[node])`, which is the total number of parent nodes.
            - Essential for DAG traversal or topological sorting: a node can only be processed
              once all its parents are already visited.
        """
        return self.number_of_visited_parents[node] == len(self.good_graph_2[node])

    def compare_two_relations(self, relation1: Relationship, relation2: Relationship) -> bool:
        """
        Compare two Relationship objects to determine which has higher priority.

        Args:
            relation1 (Relationship): The first Relationship object to compare.
            relation2 (Relationship): The second Relationship object to compare.

        Returns:
            bool:
                - True if `relation1` has higher priority than `relation2`.
                - False if `relation2` has higher priority than `relation1`.

        Description:
            - Extracts the `relationship_type` from each Relationship object.
            - Uses a predefined weight hierarchy to determine priority:
                * "cause" < "condition" < "evidence" < "contrast" < "background"
                  (lower weight means higher priority)
            - If the weights differ, the method returns True if `relation1` has the lower weight.
            - If the weights are equal, the comparison is deferred to LLM.
            - Designed to assist in ordering or selecting relationships when constructing or analyzing the graph.
        """
        relation_type1 = relation1.gettattr("relationship_type", None)
        relation_type2 = relation2.gettattr("relationship_type", None)
        weights = {"cause": 0, "condition": 1, "evidence": 2, "contrast": 3, "background": 4}
        if weights[relation_type1] != weights[relation_type2]:
            return weights[relation_type1] < weights[relation_type2]
        else:
            #TODO ask which one is better from Amirali the LLM
            return True

    def sort_all_children(self, parent_id: str, children : List[str]) -> List[str]:
        """
        Sort the children of a given parent node according to relationship priority.

        Args:
            parent_id (str): The assertion ID of the parent node.
            children (List[str]): A list of assertion IDs representing the child nodes to sort.

        Returns:
            List[str]: A new list of child nodes sorted by the decreasing priority of their relationship
                       with the parent node.

        Description:
            - Defines a comparison function for two child nodes based on the relationships
              they have with the specified parent:
                * Retrieves the Relationship objects for (parent_id, child1) and (parent_id, child2).
                * If either relationship is missing, defaults to keeping the order as-is.
                * Otherwise, uses `compare_two_relations` to determine which child has a higher-priority relationship.
            - Uses `cmp_to_key` to convert the comparison function for use with Python's `sorted`.
            - If `parent_id` is empty, the comparison is deferred to LLM.
            - Helps establish an ordered structure among children for DAG traversal, visualization, or analysis.
        """
        def comparison(node1: str, node2: str) -> bool:
            if parent_id == "":
                # TODO ask which one is better from Amirali the LLM
                return
            relation1 = self.relationship_for_pair.get((parent_id, node1))
            relation2 = self.relationship_for_pair.get((parent_id, node2))
            if relation1 is None or relation2 is None:
                return True
            return self.compare_two_relations(relation1, relation2)

        return sorted(children, key=cmp_to_key(comparison))

    def get_valid_children(self, node: str) -> List[str]:
        """
        Get all children of a node that are ready to be processed in traversal.

        Args:
            node (str): The assertion ID of the parent node.

        Returns:
            List[str]: A list of child nodes whose all parents have been visited
                       and are now ready for processing.

        Description:
            - Iterates over all children of the given `node` in `good_graph_1`.
            - For each child, increments its counter in `number_of_visited_parents`.
            - Uses `check_used_all_parents` to determine if the child is now ready
              (all its parents have been visited).
            - Only adds children that are ready to the returned list.
            - Essential for DAG traversal and topological sorting to ensure
              nodes are processed in dependency order.
        """
        children = []
        for child in self.good_graph_1[node]:
            self.number_of_visited_parents[child] += 1
            if self.check_used_all_parents(child):
                children.append(child)
        return children

    def ordering_assertions(self, lst: List[str], deep: int):
        """
        Recursively order assertions in the graph to produce a linear topological structure.

        Args:
            lst (List[str]): A list of assertion IDs to process at the current layer.
            deep (int): The current layer depth (0 for the top layer).

        Returns:
            None

        Description:
            - For each node in the input list `lst`:
                * Appends the node to `ordered_graph` (the global traversal order).
                * Adds the node to `assertions_by_layers[deep]`, grouping nodes by layer.
                * Retrieves children that are ready for processing using `get_valid_children`.
                * Sorts the children based on relationship priority with `sort_all_children`.
                * Recursively calls `ordering_assertions` on the sorted children, incrementing `deep`.
            - Produces both a flattened ordering (`ordered_graph`) and a layered representation (`assertions_by_layers`) of the DAG.
            - Ensures nodes are processed only after all parent nodes have been visited.
            - Useful for topological sorting, visualization, or downstream graph analysis.
        """
        for node in lst:
            """
            Produce a fully ordered graph of assertions using topological sorting and layered traversal.
            
            Args:
                None
            
            Returns:
                None
            
            Description:
                - Finds all starting nodes (nodes with no incoming edges) using `find_all_starting_nodes`.
                - Sorts the starting nodes according to relationship priority via `sort_all_children` (parent_id is empty for starting nodes).
                - Calls `ordering_assertions` recursively to process the entire DAG:
                    * Updates `ordered_graph` with the traversal order.
                    * Updates `assertions_by_layers` with nodes grouped by layer depth.
                - Returns the final flattened `ordered_graph`.
                - Ensures that all dependencies (parents) are respected, producing a valid DAG ordering.
            """
            self.ordered_graph.append(node)
            self.assertions_by_layers.setdefault(deep, []).append(node)
            children = self.get_valid_children(node)
            sorted_children = self.sort_all_children(node, children)
            self.ordering_assertions(sorted_children, deep + 1)

    def order_the_graph(self) -> None:
        starting_nodes = self.sort_all_children("", self.find_all_starting_nodes())
        self.ordering_assertions(starting_nodes, 0)

test_1 = [
    # Cycle 1: A1 ↔ A2 ↔ A3 ↔ A1 (all contradictions)
    Relationship(assertion1_id="A1", assertion2_id="A2",
                 relationship_type="contradiction", confidence=0.91,
                 explanation="A1 claims X holds; A2 asserts not-X."),
    Relationship(assertion1_id="A2", assertion2_id="A3",
                 relationship_type="contradiction", confidence=0.88,
                 explanation="A2 negates the outcome proposed by A3."),
    Relationship(assertion1_id="A3", assertion2_id="A1",
                 relationship_type="contradiction", confidence=0.90,
                 explanation="A3 rejects A1’s premise, closing the cycle."),

    # Cycle 2: A4 → A5 → A6 → A4 (mixed types)
    Relationship(assertion1_id="A4", assertion2_id="A5",
                 relationship_type="evidence", confidence=0.84,
                 explanation="A4 cites data that supports A5."),
    Relationship(assertion1_id="A5", assertion2_id="A6",
                 relationship_type="cause", confidence=0.79,
                 explanation="A5 describes a mechanism that produces A6."),
    Relationship(assertion1_id="A6", assertion2_id="A4",
                 relationship_type="background", confidence=0.73,
                 explanation="A6 provides context assumed by A4."),

    # Cycle 3: A7 → A8 → A9 → A10 → A7 (with contradictions)
    Relationship(assertion1_id="A7", assertion2_id="A8",
                 relationship_type="contradiction", confidence=0.86,
                 explanation="A7 states a limit exists; A8 says no such limit."),
    Relationship(assertion1_id="A8", assertion2_id="A9",
                 relationship_type="evidence", confidence=0.76,
                 explanation="A8 references measurements backing A9."),
    Relationship(assertion1_id="A9", assertion2_id="A10",
                 relationship_type="cause", confidence=0.80,
                 explanation="A9 implies A10 via a causal link."),
    Relationship(assertion1_id="A10", assertion2_id="A7",
                 relationship_type="contradiction", confidence=0.82,
                 explanation="A10’s conclusion conflicts with A7."),

    # Additional contradiction cycles to reach 20 nodes
    Relationship(assertion1_id="A11", assertion2_id="A12",
                 relationship_type="contradiction", confidence=0.85,
                 explanation="A11 and A12 claim opposing outcomes."),
    Relationship(assertion1_id="A12", assertion2_id="A13",
                 relationship_type="contradiction", confidence=0.83,
                 explanation="A12 and A13 disagree fundamentally."),
    Relationship(assertion1_id="A13", assertion2_id="A11",
                 relationship_type="contradiction", confidence=0.84,
                 explanation="A13 refutes A11, completing a contradiction cycle."),

    Relationship(assertion1_id="A14", assertion2_id="A15",
                 relationship_type="contradiction", confidence=0.80,
                 explanation="A14 predicts growth; A15 predicts decline."),
    Relationship(assertion1_id="A15", assertion2_id="A16",
                 relationship_type="contradiction", confidence=0.82,
                 explanation="A15 and A16 provide incompatible claims."),
    Relationship(assertion1_id="A16", assertion2_id="A14",
                 relationship_type="contradiction", confidence=0.81,
                 explanation="A16 undermines A14, closing the cycle."),

    # Larger contradiction cycle: A17 → A18 → A19 → A20 → A17
    Relationship(assertion1_id="A17", assertion2_id="A18",
                 relationship_type="contradiction", confidence=0.87,
                 explanation="A17 and A18 cannot both be true."),
    Relationship(assertion1_id="A18", assertion2_id="A19",
                 relationship_type="contradiction", confidence=0.89,
                 explanation="A18 asserts the opposite of A19."),
    Relationship(assertion1_id="A19", assertion2_id="A20",
                 relationship_type="contradiction", confidence=0.88,
                 explanation="A19 and A20 contradict each other."),
    Relationship(assertion1_id="A20", assertion2_id="A17",
                 relationship_type="contradiction", confidence=0.90,
                 explanation="A20 invalidates A17, completing the cycle.")
]

test_2 = [
    Relationship(assertion1_id="A1", assertion2_id="A3", relationship_type="cause", confidence=0.91, explanation="A1 causes A3."),
    Relationship(assertion1_id="A1", assertion2_id="A5", relationship_type="evidence", confidence=0.82, explanation="A1 supports A5."),
    Relationship(assertion1_id="A2", assertion2_id="A4", relationship_type="contradiction", confidence=0.76, explanation="A2 contrasts A4."),
    Relationship(assertion1_id="A2", assertion2_id="A6", relationship_type="background", confidence=0.68, explanation="A2 provides context for A6."),
    Relationship(assertion1_id="A3", assertion2_id="A7", relationship_type="cause", confidence=0.94, explanation="A3 leads to A7."),
    Relationship(assertion1_id="A4", assertion2_id="A8", relationship_type="condition", confidence=0.80, explanation="A4 is valid if A8 holds."),
    Relationship(assertion1_id="A5", assertion2_id="A9", relationship_type="evidence", confidence=0.85, explanation="A5 gives evidence for A9."),
    Relationship(assertion1_id="A6", assertion2_id="A10", relationship_type="contradiction", confidence=0.73, explanation="A6 contrasts with A10."),
    Relationship(assertion1_id="A7", assertion2_id="A11", relationship_type="background", confidence=0.67, explanation="A7 provides background for A11."),
    Relationship(assertion1_id="A8", assertion2_id="A12", relationship_type="cause", confidence=0.92, explanation="A8 directly causes A12."),
    Relationship(assertion1_id="A9", assertion2_id="A13", relationship_type="condition", confidence=0.79, explanation="A9 holds under the condition of A13."),
    Relationship(assertion1_id="A10", assertion2_id="A14", relationship_type="evidence", confidence=0.81, explanation="A10 supports A14."),
    Relationship(assertion1_id="A15", assertion2_id="A11", relationship_type="contradiction", confidence=0.74, explanation="A11 is contrasted with A15."),
    Relationship(assertion1_id="A12", assertion2_id="A16", relationship_type="background", confidence=0.65, explanation="A12 provides context for A16."),
    Relationship(assertion1_id="A13", assertion2_id="A17", relationship_type="cause", confidence=0.90, explanation="A13 results in A17."),
    Relationship(assertion1_id="A14", assertion2_id="A18", relationship_type="condition", confidence=0.78, explanation="A14 is valid if A18 holds."),
    Relationship(assertion1_id="A15", assertion2_id="A19", relationship_type="evidence", confidence=0.87, explanation="A15 gives evidence for A19."),
    Relationship(assertion1_id="A16", assertion2_id="A20", relationship_type="contradiction", confidence=0.71, explanation="A16 contrasts with A20."),
    Relationship(assertion1_id="A17", assertion2_id="A1", relationship_type="background", confidence=0.66, explanation="A17 gives background for A1."),
    Relationship(assertion1_id="A18", assertion2_id="A2", relationship_type="cause", confidence=0.93, explanation="A18 leads to A2."),
    Relationship(assertion1_id="A19", assertion2_id="A5", relationship_type="condition", confidence=0.77, explanation="A19 holds only if A5 is true."),
    Relationship(assertion1_id="A20", assertion2_id="A8", relationship_type="evidence", confidence=0.84, explanation="A20 supports A8."),
    Relationship(assertion1_id="A3", assertion2_id="A12", relationship_type="contradiction", confidence=0.70, explanation="A3 contrasts with A12."),
    Relationship(assertion1_id="A6", assertion2_id="A17", relationship_type="cause", confidence=0.88, explanation="A6 leads to A17."),
    Relationship(assertion1_id="A10", assertion2_id="A19", relationship_type="background", confidence=0.63, explanation="A10 provides background for A19."),
]



if __name__ == "__main__":
    our_graph = GlobalGraph(test_2)
    our_graph.resolve_cycles_and_conflicts()
    # print(our_graph.ordered_graph)
    # print(our_graph.contradiction_graph)
    # print(our_graph.contradiction_relations)
    # print(our_graph.relationships)
    # print(our_graph.contradiction_nodes)
    # print(our_graph.nodes)
    # print(our_graph.reverse_graph)
    # print(our_graph.number_of_visited_parents)
    # print(our_graph.relationship_for_pair)