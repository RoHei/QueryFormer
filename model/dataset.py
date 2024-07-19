from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
from collections import deque

from model.database_util import Encoding, parse_filter, parse_join, TreeNode, pad_attn_bias_unsqueeze, \
    floyd_warshall_rewrite, pad_rel_pos_unsqueeze, pad_1d_unsqueeze, pad_2d_unsqueeze, filter_dict_to_histogram
from model.util import Normalizer


def calculate_node_heights(adjacency_matrix: List[Tuple[int, int]], tree_size: int):
    if tree_size == 1:
        return np.array([0])

    adjacency_matrix = np.array(adjacency_matrix)
    node_ids = np.arange(tree_size, dtype=int)
    node_order = np.zeros(tree_size, dtype=int)
    uneval_nodes = np.ones(tree_size, dtype=bool)

    parent_nodes = adjacency_matrix[:, 0]
    child_nodes = adjacency_matrix[:, 1]

    n = 0
    while uneval_nodes.any():
        uneval_mask = uneval_nodes[child_nodes]
        unready_parents = parent_nodes[uneval_mask]
        node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
        node_order[node2eval] = n
        uneval_nodes[node2eval] = False
        n += 1
    return node_order


def topological_sort(root_node: TreeNode) -> Tuple[List[Tuple[int, int]], List[int], List[np.ndarray]]:
    adjacency_matrix: List[Tuple[int, int]] = []  # from parent to children
    num_child: List[int] = []
    features: List[np.ndarray] = []

    to_visit = deque()
    to_visit.append((0, root_node))
    next_id = 1

    while to_visit:
        idx, node = to_visit.popleft()
        features.append(node.feature_vector)
        num_child.append(len(node.children))
        for child in node.children:
            to_visit.append((next_id, child))
            adjacency_matrix.append((idx, next_id))
            next_id += 1
    return adjacency_matrix, num_child, features


class PlanTreeDataset(Dataset):
    def __init__(self,
                 query_plans_df: pd.DataFrame,
                 encoding: Encoding,
                 hist_file: List[dict],
                 card_norm: Normalizer,
                 cost_norm: Normalizer,
                 table_sample: pd.DataFrame,
                 target_variable: str = "cost"):

        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        self.length = len(query_plans_df)
        query_plans = [json.loads(plan)['Plan'] for plan in query_plans_df['json']]

        self.cardinalities = [node['Actual Rows'] for node in query_plans]
        self.costs = [json.loads(plan)['Execution Time'] for plan in query_plans_df['json']]
        self.normalized_cardinalities = torch.from_numpy(card_norm.normalize_labels(self.cardinalities))
        self.normalized_costs = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        self.to_predict = target_variable
        if target_variable == 'cost':
            self.labels = self.normalized_costs
        elif target_variable == 'card':
            self.labels = self.normalized_cardinalities
        elif target_variable == 'both':  # try not to use, just in case
            self.labels = self.normalized_costs
        else:
            raise Exception('Unknown to_predict type')

        # Encode the query plans
        self.encoded_queries = []
        for query_index, query_plan in zip(list(query_plans_df['id']), query_plans):
            encoded_query = self.encode_query_plans(query_index=query_index, query_plan=query_plan)
            self.encoded_queries.append(encoded_query)

    def encode_query_plans(self, query_index: int, query_plan: dict, max_node: int = 30, rel_pos_max: int = 20) -> dict:
        # Convert plan to tree node
        tree_node: TreeNode = self.convert_plan_to_tree_node(plan=query_plan, index=query_index, encoding=self.encoding)

        # Get adjacency matrix, num_child, features
        adjacency_matrix, number_of_children, features = topological_sort(root_node=tree_node)
        node_heights = calculate_node_heights(adjacency_matrix, len(features))

        # Do conversions
        features = torch.FloatTensor(features)
        node_heights = torch.LongTensor(node_heights)
        adjacency_matrix = torch.LongTensor(np.array(adjacency_matrix))

        # Initialize attention bias according to num_features plus extra entry
        attention_bias = torch.zeros([len(features) + 1, len(features) + 1], dtype=torch.float)

        # Transpose adjacency matrix to get edge index
        edge_index = adjacency_matrix.t()

        # Calculate the shortest path between all pairs of nodes in the graph
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
        else:
            boolean_adjacency = torch.zeros([len(features), len(features)], dtype=torch.bool)
            boolean_adjacency[edge_index[0, :], edge_index[1, :]] = True
            shortest_path_result = floyd_warshall_rewrite(boolean_adjacency.numpy())

        # Convert the shortest path result to a tensor
        rel_pos = torch.from_numpy(shortest_path_result).long()

        # Set elements of attention_bias to -inf if the shortest path is greater than rel_pos_max
        # This is to prevent the model from attending to nodes that are too far away
        attention_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')

        # Pad the attention bias tensor and extra dimension.
        attention_bias = pad_attn_bias_unsqueeze(attention_bias, max_node + 1)

        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)
        final_node_heights = pad_1d_unsqueeze(node_heights, max_node)
        features = pad_2d_unsqueeze(features, max_node)

        return {'features': features,
                'attention_bias': attention_bias,
                'rel_pos': rel_pos,
                'node_heights': final_node_heights}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.encoded_queries[idx], (self.normalized_costs[idx], self.normalized_cardinalities[idx])

    def convert_plan_to_tree_node(self, plan: dict, index: int, encoding: Encoding) -> TreeNode:
        """ bfs accumulate plan"""

        # Parse node information
        filters, alias = parse_filter(plan)
        join = parse_join(plan)
        join_id = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)

        # Create new TreeNode
        node = TreeNode(node_type=plan['Node Type'],
                        node_type_id=encoding.encode_type(plan['Node Type']),
                        card=plan['Actual Rows'],
                        filter=filters,
                        join_id=join_id,
                        join_str=join,
                        filter_dict=filters_encoded,
                        encoding=encoding)

        # Eventually add table information
        if 'Relation Name' in plan:
            node.table = plan['Relation Name']
            node.table_id = encoding.encode_table(table=plan['Relation Name'])
        node.query_id = index

        # Do featurization of existing features
        node.feature_vector = node.featurize_operator(hist_file=self.hist_file,
                                                      table_sample=self.table_sample)

        # Recursively convert children
        if 'Plans' in plan:
            for sub_plan in plan['Plans']:
                sub_plan['parent'] = plan  # Add parent plan to children as well.
                child = self.convert_plan_to_tree_node(plan=sub_plan, index=index, encoding=encoding)
                child.parent = node
                node.add_child(child)
        return node


