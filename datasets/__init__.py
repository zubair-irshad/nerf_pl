from datasets.objectron_multi import ObjectronMultiDataset
from datasets.srn_multi_ae import SRN_Multi_AE
from .blender import BlenderDataset
from .llff import LLFFDataset, LLFFDatasetNOCS, LLFFDatasetNOCSOrig
from .google_scanned import GoogleScannedDataset
from .objectron import ObjectronDataset
from .objectron_multi import ObjectronMultiDataset
from .srn_multi_ae import SRN_Multi_AE
from .pdmultiobj_ae import PDMultiObject_AE
from .pdmultiobject import PDMultiObject
from .pdmultiobject_ae_nocs import PDMultiObject_AE_NOCS
from .pdmultiobject_ae_cv import PDMultiObject_AE_CV
from .sapien import SapienDataset
from .sapien_multi import SapienDatasetMulti
from .pd_multi import PD_Multi

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_nocs': LLFFDatasetNOCS,
                'llff_nocs_test': LLFFDatasetNOCSOrig,
                'google_scanned': GoogleScannedDataset,
                'objectron': ObjectronDataset,
                'objectron_multi': ObjectronMultiDataset,
                'srn_multi_ae': SRN_Multi_AE,
                'pd_multi': PD_Multi,
                'pd_multi_obj': PDMultiObject,
                'pd_multi_obj_ae': PDMultiObject_AE,
                'pd_multi_obj_ae_nocs': PDMultiObject_AE_NOCS,
                'pd_multi_obj_ae_cv': PDMultiObject_AE_CV,
                'sapien': SapienDataset,
                'sapien_multi': SapienDatasetMulti}

# dataset_dict = {'blender': BlenderDataset,
#                 'llff': LLFFDataset}