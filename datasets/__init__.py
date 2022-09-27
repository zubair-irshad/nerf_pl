from datasets.objectron_multi import ObjectronMultiDataset
from datasets.srn_multi import SRN_Multi
from .blender import BlenderDataset
from .llff import LLFFDataset, LLFFDatasetNOCS, LLFFNOCSBackground, LLFFDatasetNSFF
from .google_scanned import GoogleScannedDataset
from .objectron import ObjectronDataset
from .objectron_multi import ObjectronMultiDataset
from .srn import SRNDataset
from .srn_multi import SRN_Multi
from .co3d import CO3D_Instance
from .pdmultiview import PDMultiView
from .pdmultiobject import PDMultiObject

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_nocs': LLFFDatasetNOCS,
                'nocs_bckg': LLFFNOCSBackground,
                'llff_nsff': LLFFDatasetNSFF,
                'google_scanned': GoogleScannedDataset,
                'objectron': ObjectronDataset,
                'objectron_multi': ObjectronMultiDataset,
                'srn': SRNDataset,
                'srn_multi': SRN_Multi,
                'co3d': CO3D_Instance,
                'pd': PDMultiView,
                'pdmultiobject': PDMultiObject}

# dataset_dict = {'blender': BlenderDataset,
#                 'llff': LLFFDataset}