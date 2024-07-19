import torch
import torch.nn as nn
import torch.nn.functional as F

from model.database_util import Batch


class FinalPredictionLayer(nn.Module):
    def __init__(self,
                 in_feature: int = 69,
                 hid_units: int = 256,
                 contract: int = 1,
                 mid_layers: bool = True,
                 res_con: bool = True):

        super(FinalPredictionLayer, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        self.out_mlp1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp1 = nn.Linear(hid_units, hid_units // contract)
        self.mid_mlp2 = nn.Linear(hid_units // contract, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))
        return out


class FeatureEmbed(nn.Module):
    def __init__(self,
                 embedding_size: int = 32,
                 tables: int = 10,
                 types: int = 20,
                 joins: int = 40,
                 columns: int = 30,
                 ops: int = 4,
                 use_sample: bool = True,
                 use_hist: bool = True,
                 bin_number: int = 50):

        super(FeatureEmbed, self).__init__()
        self.use_sample = use_sample
        self.embed_size = embedding_size
        self.use_hist = use_hist
        self.bin_number = bin_number
        self.typeEmbed = nn.Embedding(types, embedding_size)
        self.tableEmbed = nn.Embedding(tables, embedding_size)
        self.columnEmbed = nn.Embedding(columns, embedding_size)
        self.opEmbed = nn.Embedding(ops, embedding_size // 8)
        self.linearFilter2 = nn.Linear(embedding_size + embedding_size // 8 + 1, embedding_size + embedding_size // 8 + 1)
        self.linearFilter = nn.Linear(embedding_size + embedding_size // 8 + 1, embedding_size + embedding_size // 8 + 1)
        self.linearType = nn.Linear(embedding_size, embedding_size)
        self.linearJoin = nn.Linear(embedding_size, embedding_size)
        self.linearSample = nn.Linear(1000, embedding_size)
        self.linearHist = nn.Linear(bin_number, embedding_size)
        self.joinEmbed = nn.Embedding(joins, embedding_size)
        if use_hist:
            self.project = nn.Linear(embedding_size * 5 + embedding_size // 8 + 1, embedding_size * 5 + embedding_size // 8 + 1)
        else:
            self.project = nn.Linear(embedding_size * 4 + embedding_size // 8 + 1, embedding_size * 4 + embedding_size // 8 + 1)

    def forward(self, feature):
        """Input: B by 14 (type, join, f1, f2, f3, mask1, mask2, mask3)"""
        typeId, joinId, filtersId, filtersMask, hists, table_sample = torch.split(feature, (1, 1, 9, 3, self.bin_number * 3, 1001), dim=-1)
        type_embedding = self.get_type(typeId)
        join_embedding = self.getJoin(joinId)
        filter_embedding = self.get_filter(filtersId, filtersMask)
        histogram_embedding = self.getHist(hists, filtersMask)
        table_embedding = self.get_table(table_sample)

        if self.use_hist:
            final = torch.cat((type_embedding, filter_embedding, join_embedding, table_embedding, histogram_embedding), dim=1)
        else:
            final = torch.cat((type_embedding, filter_embedding, join_embedding, table_embedding), dim=1)
        final = F.leaky_relu(self.project(final))
        return final

    def get_type(self, type_id):
        emb = self.typeEmbed(type_id.long())
        return emb.squeeze(1)

    def get_table(self, table_sample):
        table, sample = torch.split(table_sample, (1, 1000), dim=-1)
        emb = self.tableEmbed(table.long()).squeeze(1)

        if self.use_sample:
            emb += self.linearSample(sample)
        return emb

    def getJoin(self, join_id):
        emb = self.joinEmbed(join_id.long())
        return emb.squeeze(1)

    def getHist(self, hists, filters_mask):
        # batch * 50 * 3
        histExpand = hists.view(-1, self.bin_number, 3).transpose(1, 2)

        emb = self.linearHist(histExpand)
        emb[~filters_mask.bool()] = 0.  # mask out space holder

        # avg by # of filters
        num_filters = torch.sum(filters_mask, dim=1)
        total = torch.sum(emb, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg

    def get_filter(self, filters_id, filters_mask):
        # get filters, then apply mask
        filter_expand = filters_id.view(-1, 3, 3).transpose(1, 2)
        colsId = filter_expand[:, :, 0].long()
        opsId = filter_expand[:, :, 1].long()
        vals = filter_expand[:, :, 2].unsqueeze(-1)  # b by 3 by 1

        # b by 3 by embed_dim
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)

        concat = torch.cat((col, op, vals), dim=-1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))

        # apply mask
        concat[~filters_mask.bool()] = 0.

        # avg by # of filters
        num_filters = torch.sum(filters_mask, dim=1)
        total = torch.sum(concat, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg


class QueryFormer(nn.Module):
    def __init__(self,
                 embedding_size: int = 32,
                 ffn_dim: int = 32,
                 head_size: int = 8,
                 dropout: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 n_layers: int = 8,
                 use_sample: bool = True,
                 use_histogram: bool = True,
                 bin_number: int = 50,
                 hidden_dim_prediction: int = 256):

        super(QueryFormer, self).__init__()

        if use_histogram:
            hidden_dim = embedding_size * 5 + embedding_size // 8 + 1
        else:
            hidden_dim = embedding_size * 4 + embedding_size // 8 + 1

        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_histogram = use_histogram

        # Define feature embedding layer
        self.feature_embedding_layer = FeatureEmbed(embedding_size=embedding_size,
                                                    use_sample=use_sample,
                                                    use_hist=use_histogram,
                                                    bin_number=bin_number)

        # Define encoders for inputs
        self.relative_position_encoder = nn.Embedding(64, head_size, padding_idx=0)
        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        # Define intermediate transformer layers
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)

        # Define prediction layers
        self.prediction_layer = FinalPredictionLayer(hidden_dim, hidden_dim_prediction)
        self.prediction_layer_2 = FinalPredictionLayer(hidden_dim, hidden_dim_prediction)

    def forward(self, batched_data: Batch):
        # Read batch
        attention_bias = batched_data.attn_bias
        rel_pos = batched_data.rel_pos
        features = batched_data.x
        heights = batched_data.heights
        batch_size, number_nodes = features.size()[:2]

        # Shape the attention bias
        tree_attn_bias = attention_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)

        # Relative position encoding and rearranging dimensions
        # [n_batch, n_node, n_node, n_head] -> [# n_batch, n_head, n_node, n_node]
        rel_pos_bias = self.relative_position_encoder(rel_pos).permute(0, 3, 1, 2)

        # Adding relative position encoding to attention bias but exluding the first position in the last two dimensions
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        # Reset relative position encoding
        # Reshape the weights of super token virtual distance to the shape of (1, headsize, 1).
        # This distance is an embedding layer that represents the distance to the super token.
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)

        # Adding reshaped weigths to batches and heads ot of tree_attn_bias,
        # but only for the first position in the last dimension
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        # Embed features
        features_view = features.view(-1, 1165)
        embedded_features = self.feature_embedding_layer(features_view).view(batch_size, -1, self.hidden_dim)

        # Add height encoding to the embedded features
        embedded_features = embedded_features + self.height_encoder(heights)

        # Add super token to the embedded features
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        super_node_feature = torch.cat([super_token_feature, embedded_features], dim=1)

        # Pass through transformer layers
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)

        # Final layer normalization
        output = self.final_ln(output)
        return self.prediction_layer(output[:, 0, :]), self.prediction_layer_2(output[:, 0, :])


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
