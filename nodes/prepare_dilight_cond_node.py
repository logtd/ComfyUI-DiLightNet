import torch


class PrepareDiLightCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "mask": ("MASK",),
            "image": ("IMAGE",),
            "diff_image": ("IMAGE",),
            "ggx5": ("IMAGE",),
            "ggx13": ("IMAGE",),
            "ggx34": ("IMAGE",),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prepare"

    CATEGORY = "dilight"

    def prepare(self, mask, image, diff_image, ggx5, ggx13, ggx34):
        # shape: mask + [1, 3, 512, 512] * 4 -> [1, 16, 512, 512]
        mask = mask.unsqueeze(-1)
        cond = torch.cat([diff_image, ggx5, ggx13, ggx34], dim=3) * mask
        cond = torch.cat([mask, image, cond], dim=3)
        return (cond,)
