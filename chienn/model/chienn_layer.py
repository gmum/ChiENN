from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch_geometric.data import Batch

from chienn.model.utils import build_embedding_layer


class ChiENNLayer(nn.Module):
    """
    ChiENN layer that embeds the messages in a chiral-sensitive manner and aggregates them.
    The implementation of the eq. (4) from the paper.
    """

    def __init__(
            self,
            hidden_dim: int = 128,
            k_neighbors_embeddings_names: List[str] = ('linear', 'linear', 'linear'),
            message_final_embedding_name: str = 'ELU+linear',
            self_embedding_name: str = 'linear',
            parallel_embedding_name: str = 'linear',
            aggregation_name: str = 'sum',
            post_aggregation_embedding_name: str = 'ELU',
            dropout: float = 0.0
    ):
        """

        Args:
            hidden_dim: hidden dimension of the model.
            k_neighbors_embeddings_names: a description of the embedding layers that will be used to embed the
                k consecutive neighbors of each node. The length of this list determines the size of a window
                sliding over the neighbors in the pre-computed order. Note that setting this to ['linear'] * k
                is equivalent to concatenating the embeddings of the k neighbors and pplying a linear layer to the
                concatenated vector (as in the paper formulation). Default list describes the `W_4` matrix from the
                eq. (4), while the length of the list corresponds `k`.
            message_final_embedding_name: a description of the embedding layer that will applied to the message obtained
                from the k consecutive neighbors. Default `ELU+linear` embedding is denoted by `W_3\sigma` in the eq. (4).
            self_embedding_name: a description of the embedding layer that will be applied to the node itself. Default
                linear embedding is denoted by `W_1` matrix in the eq. (4).
            parallel_embedding_name: a description of the embedding layer that will be applied to the parallel node.
                Default linear embedding is denoted by `W_2` matrix in the eq. (4).
            aggregation_name: aggregation of incoming messages. For the moment, only 'sum' is supported.
            post_aggregation_embedding_name: a description of the embedding layer that will be applied to the
                aggregated messages. For simplicity, omitted in the paper.
            dropout: dropout probability.
        """
        super().__init__()
        self.message = ChiENNMessage(
            hidden_dim=hidden_dim,
            k_neighbors_embeddings_names=k_neighbors_embeddings_names,
            final_embedding_name=message_final_embedding_name,
        )

        self.aggregate = ChiENNAggregate(hidden_dim=hidden_dim,
                                         self_embedding_name=self_embedding_name,
                                         post_aggregation_embedding_name=post_aggregation_embedding_name,
                                         parallel_embedding_name=parallel_embedding_name,
                                         aggregation_name=aggregation_name)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, batch: Batch):
        msg, mask = self.message(batch)
        x = self.aggregate(batch, msg=msg, mask=mask)
        return self.dropout(x)


class ChiENNMessage(nn.Module):
    """
    A message module of ChiENN layer that embeds the messages in a chiral-sensitive manner.
    The implementation of the k-ary message function `\psi^k` from the eq. (4) from the paper.
    """

    def __init__(
            self,
            hidden_dim: int,
            k_neighbors_embeddings_names: List[str],
            final_embedding_name: str
    ):
        """

        Args:
            hidden_dim: hidden dimension of the model.
            k_neighbors_embeddings_names: a description of the embedding layers that will be used to embed the
                k consecutive neighbors of each node. The length of this list determines the size of a window
                sliding over the neighbors in the pre-computed order. Note that setting this to ['linear'] * k
                is equivalent to concatenating the embeddings of the k neighbors and pplying a linear layer to the
                concatenated vector (as in the paper formulation). Default list describes the `W_4` matrix from the
                eq. (4), while the length of the list corresponds `k`.
            final_embedding_name: a description of the embedding layer that will applied to the message obtained
                from the k consecutive neighbors. Default `ELU+linear` embedding is denoted by `W_3\sigma` in the eq. (4).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings_list = nn.ModuleList([
            build_embedding_layer(hidden_dim, hidden_dim, name) for name in k_neighbors_embeddings_names
        ])
        self.k = len(self.embeddings_list)
        self.final_embedding = build_embedding_layer(hidden_dim, hidden_dim, final_embedding_name)

    def forward(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Embeds the messages in a chiral-sensitive manner.

        Args:
            batch: a batch of data containing the following attributes:
                - `x` - node features of shape (num_nodes, hidden_dim).
                - `circle_index` - a tensor of shape (num_nodes, circle_size) containing the indices of the
                    (non-parallel) neighbors in the pre-computed order. To simplify the implementation, the first (k-1)
                    indices for every atom are repeated, e.g. for k=3, the circle_index[0] may
                    be (i_1, i_2, i_3, ..., i_n, i_1, i_2). Therefore, `circle_size` = `max_num_neighbors` + k-1.
                    The indices are padded with -1.

        Returns:
            A tuple of two tensors:
                - `msg` - a tensor of shape (num_nodes, max_num_neighbors, hidden_dim) containing the messages for each node.
                - `mask` - a tensor of shape (num_nodes, max_num_neighbors) containing the neighbors mask for every node.
        """
        circle_index = batch.circle_index
        num_nodes, circle_size = circle_index.shape

        # Embed all the nodes with all the embeddings form the embeddings list:
        embeddings_list = [embed(batch.x) for embed in self.embeddings_list]

        # Add a zero embedding to the beggining of each embedding list. It will be used for for retrieving padding
        # values (could be as well done with masking):
        zeros = torch.zeros((1, self.hidden_dim)).bool().to(batch.x.device)
        embeddings_list = [torch.cat([zeros, e]) for e in embeddings_list]

        # Retrieve the number of (non-parallel) neighbors for every node:
        wrapping_length = self.k - 1
        padding_size = torch.sum(torch.eq(circle_index, -1), -1)
        num_neighbors = circle_size - padding_size - wrapping_length
        num_neighbors[padding_size == circle_size] = 0

        # Flatten the circle index:
        flatten_circle_index = circle_index.view(-1)  # (num_nodes * circle_size)
        flatten_circle_length = len(flatten_circle_index)

        # Every -1 masking becomes 0 and matches appropriate index in `embeddings`:
        flatten_circle_index = flatten_circle_index + 1

        # Add k-1 zeros to the end of the flatten circle index to simplify some further calculations:
        padded_flatten_circle_index = torch.cat(
            [flatten_circle_index, zeros.view(-1)[: wrapping_length]])  # (flatten_circle_length + k-1)

        # Create `shifted_embeddings` of shape (flatten_circle_length + k-1, k, hidden_dim), so that
        # 'shifted_embeddings[i]' contains embeddings of k consecutive neighbors that will be used to compute a single
        # message with /psi^k function.
        shifted_embeddings_list = []
        for i, embedding_i in enumerate(embeddings_list):
            shifted_indexes = padded_flatten_circle_index[i: i + flatten_circle_length]
            shifted_embedding = embedding_i[shifted_indexes]
            mask = num_neighbors <= i
            mask = mask.view(-1, 1).repeat(1, circle_size).view(-1, 1)
            shifted_embedding = torch.masked_fill(shifted_embedding, mask, 0.0)
            shifted_embeddings_list.append(shifted_embedding)
        shifted_embeddings = torch.stack(shifted_embeddings_list,
                                         dim=-2)  # (flatten_circle_length + k-1, k, hidden_dim)

        # Sum which is a proxy for concatenation operator `|` in eq. (4):
        msg = torch.sum(shifted_embeddings, dim=-2)
        msg = msg.view(num_nodes, circle_size, msg.size(-1))  # (num_nodes, max_num_neighbors + k-1, hidden_dim)

        msg_mask = torch.arange(1, circle_size + 1).unsqueeze(0).to(batch.x.device) <= num_neighbors.unsqueeze(1)

        # Cut unnecessary messages
        msg = msg[:, :-wrapping_length, :]  # (num_nodes, max_num_neighbors, hidden_dim)
        msg_mask = msg_mask[:, :-wrapping_length]  # (num_nodes, max_num_neighbors)

        msg = torch.masked_fill(msg, ~msg_mask.unsqueeze(-1), 0.0)
        msg = self.final_embedding(msg)
        return msg, msg_mask


class ChiENNAggregate(nn.Module):
    """
    An aggregation module use in ChiENNLayer. It aggregates the messages from the incoming neighbors, the node itself
    and the parallel node.
    """

    def __init__(
            self,
            hidden_dim: int,
            self_embedding_name: str,
            parallel_embedding_name: str,
            aggregation_name: str,
            post_aggregation_embedding_name: str,
    ):
        """

        Args:
            hidden_dim: hidden dimension of the model.
            self_embedding_name: a description of the embedding layer that will be applied to the node itself. Default
                linear embedding is denoted by `W_1` matrix in the eq. (4).
            parallel_embedding_name: a description of the embedding layer that will be applied to the parallel node.
                Default linear embedding is denoted by `W_2` matrix in the eq. (4).
            aggregation_name: aggregation of incoming messages. For the moment, only 'sum' is supported.
            post_aggregation_embedding_name: a description of the embedding layer that will be applied to the
                aggregated messages. For simplicity, omitted in the paper.
        """
        super().__init__()
        self.self_embedding = (
            build_embedding_layer(hidden_dim, hidden_dim,
                                  self_embedding_name) if self_embedding_name != 'none' else None
        )
        self.parallel_embedding = (
            build_embedding_layer(hidden_dim, hidden_dim,
                                  parallel_embedding_name) if parallel_embedding_name != 'none' else None
        )
        if aggregation_name == 'sum':
            def _sum(all_msg: Tensor, all_mask: Tensor) -> Tensor:
                return torch.sum(torch.masked_fill(all_msg, ~all_mask.unsqueeze(-1), 0.0), dim=1)

            self.aggregation_fn = _sum
        else:
            raise NotImplementedError(f'Aggregation {aggregation_name} is not implemented.')
        self.post_aggregation_embedding = build_embedding_layer(hidden_dim, hidden_dim, post_aggregation_embedding_name)

    def forward(self, batch: Batch, msg: Tensor, mask: Tensor) -> Tensor:
        """
        Aggregates the messages from the incoming neighbors, the node itself and the parallel node.

        Args:
            batch: a batch of data containing the following attributes:
                - x: a tensor of shape (num_nodes, hidden_dim) containing the node features.
                - parallel_node_index: a tensor of shape (num_nodes,) containing the index of the parallel node.
            msg: a tensor of shape (num_nodes, max_num_neighbors, hidden_dim) containing the order-sensitive messages.
            mask: a tensor of shape (num_nodes, max_num_neighbors) containing the mask for the messages.

        Returns:
            a tensor of shape (num_nodes, hidden_dim) containing the aggregated messages.
        """
        msg_list = [msg]
        mask_list = [mask]
        if self.self_embedding is not None:
            self_msg = self.self_embedding(batch.x).unsqueeze(1)
            self_mask = torch.ones((batch.x.shape[0], 1)).bool().to(batch.x.device)
            msg_list.append(self_msg)
            mask_list.append(self_mask)
        if self.parallel_embedding is not None:
            parallel_msg = self.parallel_embedding(batch.x).unsqueeze(1)
            parallel_msg = parallel_msg[batch.parallel_node_index]
            parallel_mask = torch.ones((batch.x.shape[0], 1)).bool().to(batch.x.device)
            msg_list.append(parallel_msg)
            mask_list.append(parallel_mask)

        all_msg = torch.concat(msg_list, dim=1)
        all_mask = torch.concat(mask_list, dim=1)

        x = self.aggregation_fn(all_msg, all_mask)
        x = self.post_aggregation_embedding(x)
        all_masked_mask = torch.eq(torch.sum(all_mask, dim=-1), 0)
        x = torch.masked_fill(x, all_masked_mask.unsqueeze(1), 0.0)

        return x
