import torch
from torch import nn
from omegaconf import OmegaConf

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
        if "instance_ids" in inputs:
            ret_dict["embedding_instance"] = self.embedding_instance(
                inputs["instance_ids"].squeeze()
            )

        return ret_dict

if __name__ == '__main__':
    conf_cli = OmegaConf.from_cli()
    # conf_dataset = OmegaConf.load(conf_cli.dataset_config)
    conf_default = OmegaConf.load("../config/default_conf.yml")
    # # merge conf with the priority
    # conf_merged = OmegaConf.merge(conf_default, conf_dataset, conf_cli)

    code_library = CodeLibrary(conf_default)
    inputs = {}
    H= 480
    W=640
    instance_id = 1
    instance_mask = torch.ones((H,W))
    inputs["instance_ids"] = torch.ones_like(instance_mask).long() * instance_id
    ret_dict = code_library.forward(inputs)

    print("ret_dict", ret_dict["embedding_instance"].shape)
