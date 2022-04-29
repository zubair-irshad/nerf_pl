from .blender import BlenderDataset
from .llff import LLFFDataset, LLFFDatasetNOCS

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_nocs': LLFFDatasetNOCS}

# dataset_dict = {'blender': BlenderDataset,
#                 'llff': LLFFDataset}