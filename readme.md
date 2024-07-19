# Explaining QueryFormer code (RoHei)

Input vector to the model.
A query plan is padded to 30 nodes and represented as the following.

- features [30 * 1165]: For each node, the features are specified:
  - node_type: Categorical encoding of the node type
  - join_id: Categorical encoding of the join id
  - filters: 3x3 vector (flattened to 9) that represents the node filters in the form of [column_id, operator, value]
  - filter_mask: 3 x 1 vector that represents the mask for existing filters
  - hist: 30 x 5 vector that represents the (adaptive) histogram of the node
  - table_id: Categorical encoding of the table id
  - sample: Base table sample according to potentially existing base table
- attention_bias [31 * 31]: Attention bias matrix, this is required to adjust the attention.
- rel_pos [30 * 30]: Relative position matrix, this encodes the graph (?)
- node_height [1 * 30]: Node height, this is the height of the node in the query plan.m