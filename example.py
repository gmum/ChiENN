from chienn import smiles_to_data_with_circle_index, collate_with_circle_index
from chienn.model.chienn_model import ChiENNModel

k_neighbors = 3
model = ChiENNModel(k_neighbors=k_neighbors)

smiles_list = ['C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O']
data_list = [smiles_to_data_with_circle_index(smiles) for smiles in smiles_list]
batch = collate_with_circle_index(data_list, k_neighbors=k_neighbors)

output = model(batch)
assert output[0] != output[1]
