from torch_geometric.graphgym.register import register_config


@register_config("dataset_cfg")
def dataset_cfg(cfg):
    """Dataset-specific config options."""

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # Dimension of node attributes. Used in `LinearNodeEncoder`.
    cfg.dataset.node_encoder_in_dim = 0

    # Dimension of edge attributes. Used in `LinearEdgeEncoder`.
    cfg.dataset.edge_encoder_in_dim = 0

    # Used in ChIRo datasets. Should be set to True, when model is conformer invariant to save some RAM.
    cfg.dataset.single_conformer = True

    # Used in ChIRo datasets.
    cfg.dataset.single_enantiomer = False

    # Used in ChIRo datasets. Whether to use chiral tags. Is set to False, chiral information will be masked.
    cfg.dataset.chiral_tags = True

    # Used in TDC (ChIRo) dataset. Type of task (e.g. "Tox", "ADME").
    cfg.dataset.tdc_type = ""

    # Used in TDC (ChIRo) dataset. Name of assay (in case of tasks with multiple labels).
    cfg.dataset.tdc_assay_name = ""

    # Used in OGB (ChIRo) dataset. Type of task (e.g. "hiv", "pcba").
    cfg.dataset.ogb_dataset_name = ""

    # Used for scaling regression labels.
    cfg.dataset.scale_label = 1.0

    cfg.dataset.min_number_of_chiral_centers = 0

    # Used in our datasets.
    cfg.dataset.pre_transform_name = ""
