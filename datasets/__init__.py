from .blender import BlenderDataset
from .llff import LLFFDataset, LLFFDatasetNOCS
from .google_scanned import GoogleScannedDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_nocs': LLFFDatasetNOCS,
                'google_scanned': GoogleScannedDataset}

# dataset_dict = {'blender': BlenderDataset,
#                 'llff': LLFFDataset}