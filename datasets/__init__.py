from datasets.objectron_multi import ObjectronMultiDataset
<<<<<<< HEAD
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
=======
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
from .pd_multi_ae import PD_Multi_AE
from .srn_multi_ae import SRN_Multi_AE
from .pdmultiobject_ae import PDMultiObject_AE
from .pdmultiobject_ae_nocs import PDMultiObject_AE_NOCS
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_nocs': LLFFDatasetNOCS,
<<<<<<< HEAD
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
=======
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
                'pd_multi_obj': PDMultiObject,
                'pd_multi_ae': PD_Multi_AE,
                'srn_multi_ae': SRN_Multi_AE,
                'pd_multi_obj_ae': PDMultiObject_AE,
                'pd_multi_obj_ae_nocs': PDMultiObject_AE_NOCS}
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0

# dataset_dict = {'blender': BlenderDataset,
#                 'llff': LLFFDataset}