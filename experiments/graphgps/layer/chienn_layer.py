import abc
from typing import Tuple, Optional, List

import torch
from math import sqrt
from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.graphgym import cfg
from torch_geometric.utils import degree


class ChiENN(nn.Module):
    """
    ChiENN model that encodes the messages in a chiral-aware clockwise and counterclockwise manner and aggregates them.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float, n_heads: int, return_batch: bool = False):
        super().__init__()
        self.double_aggregation = cfg.chienn.aggregate.double_aggregation
        self.message = ChiENNMessageKNeighbors(in_dim=in_dim,
                                               out_dim=out_dim,
                                               embedding_names=cfg.chienn.message.embedding_names,
                                               aggregation=cfg.chienn.message.aggregation,
                                               final_embedding_name=cfg.chienn.message.final_embedding_name,
                                               mask_by_in_degree=cfg.chienn.message.mask_by_in_degree,
                                               single_direction=cfg.chienn.message.single_direction,
                                               shared_weights=cfg.chienn.message.shared_weights,
                                               sanity_check_in_degree=cfg.chienn.message.sanity_check_in_degree,
                                               mask_non_chiral=cfg.chienn.mask_non_chiral,
                                               )
        if self.double_aggregation:
            self.ccw_aggregate = ChiENNAggregate(x_dim=in_dim,
                                                 msg_dim=out_dim,
                                                 self_embedding_name=cfg.chienn.aggregate.self_embedding_name,
                                                 aggregation=cfg.chienn.aggregate.aggregation,
                                                 embedding_name=cfg.chienn.aggregate.embedding_name,
                                                 final_embedding_name=cfg.chienn.aggregate.final_embedding_name,
                                                 parallel_embedding_name=cfg.chienn.aggregate.parallel_embedding_name,
                                                 mask_non_chiral=cfg.chienn.mask_non_chiral,
                                                 distinct_self=cfg.chienn.aggregate.distinct_self,
                                                 n_heads=n_heads)
            self.cw_aggregate = ChiENNAggregate(x_dim=in_dim,
                                                msg_dim=out_dim,
                                                self_embedding_name=cfg.chienn.aggregate.self_embedding_name,
                                                aggregation=cfg.chienn.aggregate.aggregation,
                                                embedding_name=cfg.chienn.aggregate.embedding_name,
                                                final_embedding_name=cfg.chienn.aggregate.final_embedding_name,
                                                parallel_embedding_name=cfg.chienn.aggregate.parallel_embedding_name,
                                                mask_non_chiral=cfg.chienn.mask_non_chiral,
                                                distinct_self=cfg.chienn.aggregate.distinct_self,
                                                n_heads=n_heads)
        else:
            self.aggregate = ChiENNAggregate(x_dim=in_dim,
                                             msg_dim=out_dim,
                                             self_embedding_name=cfg.chienn.aggregate.self_embedding_name,
                                             aggregation=cfg.chienn.aggregate.aggregation,
                                             embedding_name=cfg.chienn.aggregate.embedding_name,
                                             final_embedding_name=cfg.chienn.aggregate.final_embedding_name,
                                             parallel_embedding_name=cfg.chienn.aggregate.parallel_embedding_name,
                                             mask_non_chiral=cfg.chienn.mask_non_chiral,
                                             distinct_self=cfg.chienn.aggregate.distinct_self,
                                             n_heads=n_heads)
        self.after_aggregation_embedding = build_embedding_layer(out_dim,
                                                                 out_dim,
                                                                 cfg.chienn.aggregate.after_aggregation_embedding_name)
        self.dropout = nn.Dropout(p=dropout)
        self.return_batch = return_batch

    def forward(self, batch: Batch):
        ccw_msg, ccw_mask, cw_msg, cw_mask = self.message(batch)
        if self.double_aggregation:
            ccw_x = self.ccw_aggregate(batch, ccw_msg=ccw_msg, ccw_mask=ccw_mask, cw_msg=None, cw_mask=None)
            cw_x = self.cw_aggregate(batch, ccw_msg=cw_msg, ccw_mask=cw_mask, cw_msg=None, cw_mask=None)
            x = ccw_x + cw_x
        else:
            x = self.aggregate(batch, ccw_msg=ccw_msg, ccw_mask=ccw_mask, cw_msg=cw_msg, cw_mask=cw_mask)
        x = self.after_aggregation_embedding(x)
        x = self.dropout(x)
        if self.return_batch:
            batch.x = x
            return batch
        else:
            return x


class ChiENNAggregate(nn.Module):
    """
    Module for aggregation in ChiENN model.
    """

    def __init__(self,
                 x_dim: int,
                 msg_dim: int,
                 self_embedding_name: str,
                 aggregation: str,
                 embedding_name: str,
                 final_embedding_name: str,
                 parallel_embedding_name: str,
                 n_heads: Optional[int] = None,
                 mask_non_chiral: bool = False,
                 distinct_self: bool = False):
        super().__init__()
        self.mask_non_chiral = mask_non_chiral
        self.distinct_self = distinct_self
        if self_embedding_name == 'none' and distinct_self:
            raise AttributeError('Self embeddings must be set if `distinct_self = True`!')
        self.self_embedding = (
            build_embedding_layer(x_dim, msg_dim,
                                  self_embedding_name) if self_embedding_name != 'none' else None
        )
        self.parallel_embedding = (
            build_embedding_layer(x_dim, msg_dim,
                                  parallel_embedding_name) if parallel_embedding_name != 'none' else None
        )
        self.aggregation = aggregation
        if aggregation == 'scale_dot_attention':
            self.aggregate_fn = GraphScaleDotAttention(msg_dim,
                                                       n_heads=n_heads,
                                                       value_embedding_name=embedding_name,
                                                       final_embedding_name=final_embedding_name)
        elif aggregation == 'gat_attention':
            self.aggregate_fn = GraphConcatAttention(msg_dim,
                                                     n_heads=n_heads,
                                                     embedding_name=embedding_name,
                                                     final_embedding_name=final_embedding_name)
        elif aggregation == 'mean':
            self.aggregate_fn = GraphMeanAggregation(msg_dim,
                                                     embedding_name=embedding_name,
                                                     final_embedding_name=final_embedding_name)
        elif aggregation == 'add':
            self.aggregate_fn = GraphAddAggregation(msg_dim,
                                                    embedding_name=embedding_name,
                                                    final_embedding_name=final_embedding_name)
        else:
            raise NotImplementedError()

    def forward(self, batch: Batch, ccw_msg: Tensor, ccw_mask: Tensor, cw_msg: Tensor, cw_mask: Tensor) -> Tensor:
        """
        Every node needs its own embedding for the update. Unfortunately, there is no elegant way to create such
        embeddings in a message module, so the aggregation module creates `self_x` embeddings and concatenates them
        to the messages obtained with a message module.
        """
        msg_list = [ccw_msg]
        mask_list = [ccw_mask]
        if self.self_embedding is not None:
            self_msg = self.self_embedding(batch.x)
            if not self.distinct_self:
                self_mask = torch.ones((batch.x.shape[0], 1)).bool().to(batch.x.device)
                msg_list.append(self_msg.unsqueeze(1))
                mask_list.append(self_mask)
        if self.parallel_embedding is not None:
            parallel_msg = self.parallel_embedding(batch.x).unsqueeze(1)
            parallel_msg = parallel_msg[batch.parallel_node_index]
            parallel_mask = torch.ones((batch.x.shape[0], 1)).bool().to(batch.x.device)
            msg_list.append(parallel_msg)
            mask_list.append(parallel_mask)
        if cw_msg is not None:
            msg_list.append(cw_msg)
            mask_list.append(cw_mask)

        all_msg = torch.concat(msg_list, dim=1)
        all_mask = torch.concat(mask_list, dim=1)

        x = self.aggregate_fn(all_msg, all_mask)
        all_masked_mask = torch.eq(torch.sum(all_mask, dim=-1), 0)
        x = torch.masked_fill(x, all_masked_mask.unsqueeze(1), 0.0)

        if self.distinct_self:
            x = x + self_msg

        if self.mask_non_chiral:
            x = torch.masked_fill(x, ~batch.chiral_mask.unsqueeze(1), 0.0)

        return x


class GraphScaleDotAttention(nn.Module):
    """
    Standard multi-head scale dot product attention adapted for graphs such that single node attends only to
    its neighbors.
    """

    def __init__(self, in_dim: int, n_heads: int, value_embedding_name: str, final_embedding_name: str):
        super().__init__()
        assert in_dim % n_heads == 0
        # We don't necessarily need to embed the value as it has been embedded in message module:
        self.value_embedding = build_embedding_layer(in_dim, in_dim, value_embedding_name)
        self.query_linear = nn.Linear(in_dim, in_dim, bias=False)
        self.key_linear = nn.Linear(in_dim, in_dim, bias=False)
        self.final_embedding = build_embedding_layer(in_dim, in_dim, final_embedding_name)
        self.n_heads = n_heads
        self.head_dim = in_dim // n_heads

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        n_nodes, n_neighbors, dim = x.shape
        query = self.query_linear(x).view(n_nodes, n_neighbors, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(x[:, 0]).view(n_nodes, 1, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.value_embedding(x).view(n_nodes, n_neighbors, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(self.head_dim)
        scores = scores.squeeze(-1)
        scores = torch.masked_fill(scores, ~mask.unsqueeze(1), -float('inf'))

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = torch.masked_fill(p_attn, ~mask.unsqueeze(1), 0.0)

        x = torch.mul(p_attn.unsqueeze(-1), value)
        x = x.transpose(1, 2).contiguous().view(n_nodes, n_neighbors, dim)
        x = torch.sum(x, dim=1)
        return self.final_embedding(x)


class GraphConcatAttention(nn.Module):
    """
    Attention mechanism from GAT paper (https://arxiv.org/abs/1710.10903).
    """

    def __init__(self, in_dim: int, n_heads: int, embedding_name: str, final_embedding_name: str):
        super().__init__()
        assert in_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = in_dim // n_heads
        self.embedding_layer = build_embedding_layer(in_dim, in_dim, embedding_name)
        self.score_layer = nn.Sequential(nn.Linear(2 * in_dim, n_heads, bias=False), nn.LeakyReLU())
        self.final_embedding = build_embedding_layer(in_dim, in_dim, final_embedding_name)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        n_nodes, n_neighbors, dim = x.shape
        x = self.embedding_layer(x)

        self_x = x[:, 0].unsqueeze(1).repeat(1, n_neighbors, 1)
        concat_x = torch.cat([self_x, x], dim=-1)
        scores = self.score_layer(concat_x)

        x = x.view(n_nodes, n_neighbors, self.n_heads, self.head_dim).transpose(1, 2)
        scores = scores.transpose(1, 2)

        scores = torch.masked_fill(scores, ~mask.unsqueeze(1), -float('inf'))

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = torch.masked_fill(p_attn, ~mask.unsqueeze(1), 0.0)

        x = torch.mul(p_attn.unsqueeze(-1), x)
        x = x.transpose(1, 2).contiguous().view(n_nodes, n_neighbors, dim)
        x = torch.sum(x, dim=1)
        return self.final_embedding(x)


class GraphMeanAggregation(nn.Module):
    def __init__(self, in_dim: int, embedding_name: str, final_embedding_name: str):
        super().__init__()
        self.embedding_layer = build_embedding_layer(in_dim, in_dim, embedding_name)
        self.final_embedding = build_embedding_layer(in_dim, in_dim, final_embedding_name)

    def forward(self, all_msg: Tensor, all_mask: Tensor) -> Tensor:
        x = self.embedding_layer(all_msg)
        x = torch.nanmean(torch.masked_fill(x, ~all_mask.unsqueeze(-1), torch.nan), dim=1)
        return self.final_embedding(x)


class GraphAddAggregation(nn.Module):
    def __init__(self, in_dim: int, embedding_name: str, final_embedding_name: str):
        super().__init__()
        self.embedding_layer = build_embedding_layer(in_dim, in_dim, embedding_name)
        self.final_embedding = build_embedding_layer(in_dim, in_dim, final_embedding_name)

    def forward(self, all_msg: Tensor, all_mask: Tensor) -> Tensor:
        x = self.embedding_layer(all_msg)
        x = torch.sum(torch.masked_fill(x, ~all_mask.unsqueeze(-1), 0.0), dim=1)
        return self.final_embedding(x)


class ChiENNMessageBase(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, batch: Batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Embeds the messages for all nodes.

        Args:
            batch: batch of data with `N` nodes. Batch has to contain `cw_circle_index` and `ccw_circle_index`
                attributes described in `get_circle_index` function.

        Returns:
            Tuple containing:
                ccw_msg: padded tensor of shape `(N, L, D)`. Tensor `ccw_msg[i]` contains `L` messages of shape `(D,)`
                    obtained by a "counterclockwise walk" around the i-th node.
                ccw_mask: tensor of shape `(N, L)` where `False` values indicate padding messages in `ccw_msg`.
                cw_msg: padded tensor of shape `(N, L, D)`. Tensor `cw_msg[i]` contains `L` messages of shape `(D,)`
                    obtained by a "clockwise walk" around the i-th node.
                cw_mask: tensor of shape `(N, L)` where `False` values indicate padding messages in `cw_msg`.
        """
        ...


class ChiENNMessageKNeighbors(ChiENNMessageBase):
    """
    A message module for ChiENN model that embeds the neigbors around the node (which represents an edge) in a clockwise
    and counterclockwise manner.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 embedding_names: List[str],
                 aggregation: str,
                 final_embedding_name: str,
                 mask_by_in_degree: bool,
                 single_direction: bool,
                 shared_weights: bool,
                 sanity_check_in_degree: bool = False,
                 mask_non_chiral: bool = False):
        super().__init__()
        if shared_weights and single_direction:
            raise ValueError('When using shared weights, `single_direction` must be set to False!')
        final_in_dim = out_dim if aggregation != 'concat' else 2 * out_dim
        self.single_direction = single_direction
        self.ccw_layer = ChiENNMessageKNeighborsSingleDirection(
            embeddings=[build_embedding_layer(in_dim, out_dim, name) for name in embedding_names],
            aggregation=aggregation,
            final_embedding=build_embedding_layer(final_in_dim, out_dim, final_embedding_name),
            mask_by_in_degree=mask_by_in_degree,
            sanity_check_in_degree=sanity_check_in_degree,
            mask_non_chiral=mask_non_chiral)
        if not single_direction:
            if shared_weights:
                self.cw_layer = self.ccw_layer
            else:
                self.cw_layer = ChiENNMessageKNeighborsSingleDirection(
                    embeddings=[build_embedding_layer(in_dim, out_dim, name) for name in embedding_names],
                    aggregation=aggregation,
                    final_embedding=build_embedding_layer(final_in_dim, out_dim, final_embedding_name),
                    mask_by_in_degree=mask_by_in_degree,
                    sanity_check_in_degree=sanity_check_in_degree,
                    mask_non_chiral=mask_non_chiral)

        self.count = 0

    def forward(self, batch: Batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ccw_msg, ccw_mask = self.ccw_layer(batch, circle_index=batch.ccw_circle_index)
        if not self.single_direction:
            cw_msg, cw_mask = self.cw_layer(batch, circle_index=batch.cw_circle_index)
        else:
            cw_msg, cw_mask = None, None
        return ccw_msg, ccw_mask, cw_msg, cw_mask


class ChiENNMessageKNeighborsSingleDirection(nn.Module):
    """
    A message module for ChiENN model that embeds the neigbors around the node (which represents an edge) in a single
    direction (clockwise or counterclockwise).
    """

    def __init__(self,
                 embeddings: List[nn.Module],
                 aggregation: str,
                 final_embedding: nn.Module,
                 mask_by_in_degree: bool,
                 sanity_check_in_degree: bool = True,
                 mask_non_chiral: bool = False):
        super().__init__()
        self.embeddings_list = nn.ModuleList(embeddings)
        self.k_neighbors = len(embeddings)
        self.final_embedding = final_embedding
        self.mask_by_in_degree = mask_by_in_degree
        self.mask_non_chiral = mask_non_chiral
        self.sanity_check_in_degree = sanity_check_in_degree
        if self.k_neighbors != 2 and aggregation in ['minus', 'mul']:
            raise NotImplemented(
                f'Aggregation {aggregation} is not currently supported for `k_neighbors={self.k_neighbors}`!')
        if aggregation == 'add':
            self.aggregation_fn = lambda x: torch.sum(x, dim=-2)
            self.aggregation_id_element = 0
        elif aggregation == 'concat':
            self.aggregation_fn = lambda x: x.view(x.shape[0], -1)
            self.aggregation_id_element = 0
        elif aggregation == 'minus':
            self.aggregation_fn = lambda x: x[:, 0, :] - x[:, 1, :]
            self.aggregation_id_element = 0
        elif aggregation == 'mul':
            self.aggregation_fn = lambda x: torch.mul(x[:, 0, :], x[:, 1, :])
            self.aggregation_id_element = 1
        else:
            raise NotImplemented(f'Aggregation {aggregation} is not currently supported!')

    def forward(self, batch: Batch, circle_index: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, circle_size = circle_index.shape

        embeddings_list = [embed(batch.x) for embed in self.embeddings_list]
        dim = embeddings_list[0].shape[-1]
        zeros = torch.zeros((1, dim)).bool().to(batch.x.device)
        embeddings_list = [torch.cat([zeros, e]) for e in embeddings_list]

        circle_index_padding_length = self.k_neighbors - 1
        mask_sum = torch.sum(torch.eq(circle_index, -1), -1)
        in_degree = circle_size - mask_sum - circle_index_padding_length
        in_degree[mask_sum == circle_size] = 0
        if self.sanity_check_in_degree:
            real_in_degree = degree(batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.long)
            assert torch.equal(in_degree, real_in_degree)

        if self.mask_non_chiral:
            circle_index = torch.masked_fill(circle_index, ~batch.chiral_mask.unsqueeze(1), -1)
        circle_index = circle_index.view(-1)
        circle_index_original_length = len(circle_index)
        circle_index = circle_index + 1  # every -1 masking becomes 0 and matches appropriate index in `embeddings`
        circle_index = torch.cat([circle_index, zeros.view(-1)[: circle_index_padding_length]])

        shifted_embeddings_list = []
        for i, embedding_i in enumerate(embeddings_list):
            shifted_indexes = circle_index[i: i + circle_index_original_length]
            shifted_embedding = embedding_i[shifted_indexes]
            if self.mask_by_in_degree:
                mask = in_degree <= i
                mask = mask.view(-1, 1).repeat(1, circle_size).view(-1, 1)
                shifted_embedding = torch.masked_fill(shifted_embedding, mask, self.aggregation_id_element)
            shifted_embeddings_list.append(shifted_embedding)
        shifted_embeddings = torch.stack(shifted_embeddings_list, dim=-2)

        msg = self.aggregation_fn(shifted_embeddings)
        msg = msg.view(batch_size, circle_size, msg.size(-1))

        msg_mask = torch.arange(1, circle_size + 1).unsqueeze(0).to(batch.x.device) <= in_degree.unsqueeze(1)

        # Cut unnecessary messages
        msg = msg[:, :-circle_index_padding_length, :]
        msg_mask = msg_mask[:, :-circle_index_padding_length]

        msg = torch.masked_fill(msg, ~msg_mask.unsqueeze(-1), 0.0)
        msg = self.final_embedding(msg)
        return msg, msg_mask


def _build_single_embedding_layer(in_dim: int, out_dim: int, name: str):
    if name == 'linear':
        return nn.Linear(in_dim, out_dim, bias=False)
    elif name == 'identity':
        return nn.Identity()
    elif name == 'scalar':
        return nn.Linear(in_dim, 1, bias=True)
    elif name == 'self_concat':
        return lambda x: torch.cat([x, x], dim=-1)
    elif name == 'double':
        return lambda x: 2 * x
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)()
    else:
        raise NotImplementedError(f'Layer name {name} is not implemented.')


def build_embedding_layer(in_dim: int, out_dim: int, name: str):
    sub_names = name.split('+')
    if len(sub_names) == 1:
        return _build_single_embedding_layer(in_dim, out_dim, sub_names[0])
    else:
        return nn.Sequential(*[_build_single_embedding_layer(in_dim, out_dim, sub_name) for sub_name in sub_names])
