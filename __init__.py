from .nodes.load_dilight_controlnet_node import LoadDiLightControlNetNode
from .nodes.prepare_dilight_cond_node import PrepareDiLightCondNode
from .nodes.load_dilight_image_node import LoadDiLightImageNode


NODE_CLASS_MAPPINGS = {
    "LoadDiLightControlNet": LoadDiLightControlNetNode,
    "PrepareDiLightCond": PrepareDiLightCondNode,
    "LoadDiLightImage": LoadDiLightImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDiLightControlNet": "Load DiLight ControlNet",
    "PrepareDiLightCond": "Prepare DiLight CN Image",
    "LoadDiLightImage": "Load DiLight Image",
}
