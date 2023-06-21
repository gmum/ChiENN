import sys

import pytest

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')

from graphgps.layer.chienn_layer import GraphScaleDotAttention, GraphConcatAttention
import torch


@pytest.mark.parametrize('n_nodes, max_len, dim, n_heads', [(12, 5, 12, 4)])
def test__graph_scale_dot_attention__returns_proper_shape(n_nodes, max_len, dim, n_heads):
    torch.random.manual_seed(0)
    x = torch.randn((n_nodes, max_len, dim))
    mask = torch.randint(2, (n_nodes, max_len)).bool()
    mask[:, 0] = True

    attention = GraphScaleDotAttention(dim, n_heads, value_embedding_name='linear', final_embedding_name='ELU')

    result = attention.forward(x, mask)

    assert result.shape == (n_nodes, dim)
    assert torch.sum(result.isnan()) == 0.0


@pytest.mark.parametrize('n_nodes, max_len, dim, n_heads', [(12, 5, 12, 4)])
def test__graph_scale_dot_attention__returns_sensible_values_for_single_nodes(n_nodes, max_len, dim, n_heads):
    torch.random.manual_seed(0)
    x = torch.randn((n_nodes, max_len, dim))
    mask = torch.zeros((n_nodes, max_len)).bool()
    mask[:, 0] = True

    attention = GraphScaleDotAttention(dim, n_heads, value_embedding_name='identity', final_embedding_name='identity')

    result = attention.forward(x, mask)

    assert torch.sum(result.isnan()) == 0.0
    assert torch.allclose(result.detach(), x[:, 0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('n_nodes, max_len, dim, n_heads', [(12, 5, 12, 4)])
def test__graph_concat_attention__returns_proper_shape(n_nodes, max_len, dim, n_heads):
    torch.random.manual_seed(0)
    x = torch.randn((n_nodes, max_len, dim))
    mask = torch.randint(2, (n_nodes, max_len)).bool()
    mask[:, 0] = True

    attention = GraphConcatAttention(dim, n_heads, embedding_name='identity', final_embedding_name='identity')

    result = attention.forward(x, mask)

    assert result.shape == (n_nodes, dim)
    assert torch.sum(result.isnan()) == 0.0


@pytest.mark.parametrize('n_nodes, max_len, dim, n_heads', [(12, 5, 12, 4)])
def test__graph_concat_attention__returns_sensible_values_for_single_nodes(n_nodes, max_len, dim, n_heads):
    torch.random.manual_seed(0)
    x = torch.randn((n_nodes, max_len, dim))
    mask = torch.zeros((n_nodes, max_len)).bool()
    mask[:, 0] = True

    attention = GraphConcatAttention(dim, n_heads, embedding_name='identity', final_embedding_name='identity')

    result = attention.forward(x, mask)

    assert torch.sum(result.isnan()) == 0.0
    assert torch.allclose(result.detach(), x[:, 0], rtol=1e-6, atol=1e-6)
