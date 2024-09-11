
import torch

import folder_paths
import comfy.utils
from comfy.cldm.cldm import ControlNet
from nodes import ControlNetLoader

from ..modules.neural_texture_embedding import NeuralTextureEmbedding


class DiLightControlnet(ControlNet):
    def set_input_hint_block(self, input_hint_block):
        self.input_hint_block = input_hint_block

    def to(self, device):
        self.input_hint_block.to(device)

        return super().to(device)


class LoadDiLightControlNetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
        }}
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "apply"

    CATEGORY = "dilight"

    def apply(self, control_net_name):
        controlnet, = ControlNetLoader().load_controlnet(control_net_name)
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet_data = comfy.utils.load_torch_file(controlnet_path, safe_load=True)
        state_dict = {}
        for key in list(controlnet_data.keys()):
            if 'controlnet_cond_embedding.' in key:
                new_key = key.replace('controlnet_cond_embedding.', '')
                state_dict[new_key] = controlnet_data[key]
        
        # https://huggingface.co/dilightnet/DiLightNet/blob/main/config.json
        embedder = NeuralTextureEmbedding(
            conditioning_embedding_channels=320,
            conditioning_channels=4,
            block_out_channels=[16, 32, 96, 256],
            shading_hint_channels=12
        ).to(torch.float16)
        embedder.load_state_dict(state_dict)

        controlnet.control_model.__class__ = DiLightControlnet
        controlnet.control_model.set_input_hint_block(embedder)

        return (controlnet,)
