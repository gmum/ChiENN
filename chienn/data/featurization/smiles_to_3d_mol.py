import logging

from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_3d_mol(smiles: str, max_number_of_atoms: int = 100, max_number_of_attempts: int = 5000):
    """
    Embeds the molecule in 3D space.
    Args:
        smiles: a smile representing molecule
        max_number_of_atoms: maximal number of atoms in a molecule. Molecules with more atoms will be omitted.
            max_number_of_attempts: maximal number of attempts during the embedding.
        max_number_of_attempts: max number of embeddings attempts.

    Returns:
        Embedded molecule.
    """

    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    smiles = smiles.split('.')[0]
    mol = Chem.MolFromSmiles(smiles)
    if len(mol.GetAtoms()) > max_number_of_atoms:
        logging.warning(f'Omitting molecule {smiles} as it contains more than {max_number_of_atoms} atoms.')
        return None
    if len(mol.GetAtoms()) == 0:
        logging.warning(f'Omitting molecule {smiles} as it contains no atoms after desaltization.')
        return None
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, maxAttempts=max_number_of_attempts, randomSeed=0)
    if res < 0:  # try to embed with different method
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=max_number_of_attempts,
                                    randomSeed=0)
    if res < 0:
        logging.warning(f'Omitting molecule {smiles} as cannot be embedded in 3D space properly.')
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        logging.warning(
            f"Omitting molecule {smiles} as cannot be properly optimized. "
            f"The original error message was: {e}."
        )
        return None
    return mol
