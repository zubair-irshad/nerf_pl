import torch
from torch import nn
import sys
sys.path.append("./")
from opt import get_opts

class CodeLibrary(nn.Module):
    """
    Store various codes.
    """
    def __init__(self, hparams):
        super(CodeLibrary, self).__init__()

        self.embedding_instance = torch.nn.Embedding(
            hparams.N_max_objs, 
            hparams.N_obj_code_length
        )
    def forward(self, inputs):
        ret_dict = dict()

        # if 'frame_idx' in inputs:
        #     ret_dict['embedding_a'] = self.embedding_a(inputs['frame_idx'].squeeze())
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].squeeze().shape)
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].shape)
        
        # print("self.embedding_instance(inputs[instance_ids].squeeze()", self.embedding_instance(
        #         inputs["instance_ids"].squeeze().shape))
        if "instance_ids" in inputs:
            ret_dict["embedding_instance"] = self.embedding_instance(
                inputs["instance_ids"].squeeze()
            )

        return ret_dict

class CodeLibraryShapeAppearance(nn.Module):
    """
    Store various codes.
    """
    def __init__(self, hparams):
        super(CodeLibraryShapeAppearance, self).__init__()

        self.embedding_instance_shape = torch.nn.Embedding(
            hparams.N_max_objs, 
            hparams.N_obj_code_length
        )
        self.embedding_instance_appearance = torch.nn.Embedding(
            hparams.N_max_objs, 
            hparams.N_obj_code_length
        )

    def forward(self, inputs):
        ret_dict = dict()

        # if 'frame_idx' in inputs:
        #     ret_dict['embedding_a'] = self.embedding_a(inputs['frame_idx'].squeeze())
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].squeeze().shape)
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].shape)
        
        # print("self.embedding_instance(inputs[instance_ids].squeeze()", self.embedding_instance(
        #         inputs["instance_ids"].squeeze().shape))
        if "instance_ids" in inputs:
            ret_dict["embedding_instance_shape"] = self.embedding_instance_shape(
                inputs["instance_ids"].squeeze()
            )
            ret_dict["embedding_instance_appearance"] = self.embedding_instance_appearance(
                inputs["instance_ids"].squeeze()
            )

        return ret_dict

class CodeLibraryBckgObj(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryBckgObj, self).__init__()

        self.embedding_instance = torch.nn.Embedding(
            hparams.N_max_objs, 
            hparams.N_obj_code_length
        )

        self.embedding_backgrounds = torch.nn.Embedding(
            hparams.N_max_objs, 
            128
        )

    def forward(self, inputs):
        ret_dict = dict()

        # if 'frame_idx' in inputs:
        #     ret_dict['embedding_a'] = self.embedding_a(inputs['frame_idx'].squeeze())
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].squeeze().shape)
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].shape)
        
        # print("self.embedding_instance(inputs[instance_ids].squeeze()", self.embedding_instance(
        #         inputs["instance_ids"].squeeze().shape))
        if "instance_ids" in inputs:
            ret_dict["embedding_instance"] = self.embedding_instance(
                inputs["instance_ids"].squeeze()
            )
            ret_dict["embedding_backgrounds"] = self.embedding_backgrounds(
                inputs["instance_ids"].squeeze()
            )

        return ret_dict

class CodeLibraryBckgObjShapeApp(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryBckgObjShapeApp, self).__init__()

        # self.embedding_instance = torch.nn.Embedding(
        #     hparams.N_max_objs, 
        #     hparams.N_obj_code_length
        # )

        self.embedding_instance_shape = torch.nn.Embedding(
            hparams.N_max_objs, 
            hparams.N_obj_code_length
        )
        self.embedding_instance_appearance = torch.nn.Embedding(
            hparams.N_max_objs, 
            hparams.N_obj_code_length
        )

        self.embedding_backgrounds = torch.nn.Embedding(
            hparams.N_max_objs, 
            128
        )

    def forward(self, inputs):
        ret_dict = dict()

        # if 'frame_idx' in inputs:
        #     ret_dict['embedding_a'] = self.embedding_a(inputs['frame_idx'].squeeze())
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].squeeze().shape)
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].shape)
        
        # print("self.embedding_instance(inputs[instance_ids].squeeze()", self.embedding_instance(
        #         inputs["instance_ids"].squeeze().shape))
        if "instance_ids" in inputs:

            ret_dict["embedding_instance_shape"] = self.embedding_instance_shape(
                inputs["instance_ids"].squeeze()
            )
            ret_dict["embedding_instance_appearance"] = self.embedding_instance_appearance(
                inputs["instance_ids"].squeeze()
            )
            ret_dict["embedding_backgrounds"] = self.embedding_backgrounds(
                inputs["instance_ids"].squeeze()
            )

        return ret_dict

if __name__ == '__main__':
    # conf_cli = OmegaConf.from_cli()
    # # conf_dataset = OmegaConf.load(conf_cli.dataset_config)
    # conf_default = OmegaConf.load("../config/default_conf.yml")
    # # # merge conf with the priority
    # # conf_merged = OmegaConf.merge(conf_default, conf_dataset, conf_cli)

    hparams = get_opts()
    code_library = CodeLibrary(hparams)
    inputs = {}
    H = 480
    W = 640
    instance_id = 1
    instance_mask = torch.ones((H,W))
    instance_mask = instance_mask.view(-1)
    inputs["instance_ids"] = torch.ones_like(instance_mask).long() * instance_id
    ret_dict = code_library.forward(inputs)

    # print("ret_dict", ret_dict["embedding_instance"].shape)
