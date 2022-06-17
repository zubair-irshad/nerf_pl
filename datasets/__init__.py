from datasets.srn_multi import SRN_Multi
from .blender import BlenderDataset
from .llff import LLFFDataset, LLFFDatasetNOCS
from .google_scanned import GoogleScannedDataset
from .objectron import ObjectronDataset
from .srn import SRNDataset
from .srn_multi import SRN_Multi

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_nocs': LLFFDatasetNOCS,
                'google_scanned': GoogleScannedDataset,
                'objectron': ObjectronDataset,
                'srn': SRNDataset,
                'srn_multi': SRN_Multi}

# dataset_dict = {'blender': BlenderDataset,
#                 'llff': LLFFDataset}