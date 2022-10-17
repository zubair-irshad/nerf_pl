from datasets.objectron_multi import ObjectronMultiDataset
from datasets.pd_multi import PD_Multi
from datasets.srn_multi import SRN_Multi
from .blender import BlenderDataset
from .llff import LLFFDataset, LLFFDatasetNOCS, LLFFNOCSBackground, LLFFDatasetNSFF
from .google_scanned import GoogleScannedDataset
from .objectron import ObjectronDataset
from .objectron_multi import ObjectronMultiDataset
from .srn import SRNDataset
from .srn_multi import SRN_Multi

#functorch doesnt work with pytorch3d so revert to pytorch 1.10 with cuda 11.3 to make it work with p3d
# from .co3d import CO3D_Instance
from .pd import PDDataset
from .pdmultiobject import PDMultiObject
from .pd_multi import PD_Multi

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
                # 'co3d': CO3D_Instance,
                'pd': PDDataset,
                'pd_multi': PD_Multi,
                'pdmultiobject': PDMultiObject}

# dataset_dict = {'blender': BlenderDataset,
#                 'llff': LLFFDataset}