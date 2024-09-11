# ComfyUI-DiLightNet (WIP)

ComfyUI nodes to use [DiLightNet](https://github.com/iamNCJ/DiLightNet)

These nodes can run DiLightNet, but the Dust3r or BlenderPy implementations to create lighting are not included. Expect those to be added to seperate repos when time allows.

## Installation
You will need to download the [DiLightNet Controlnet](https://huggingface.co/dilightnet/DiLightNet/blob/main/diffusion_pytorch_model.safetensors) to your `ComfyUI/models/controlnet` directory.

Additionally, you will need the [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2) model and clip text encoder.

## Examples
Example workflow can be found in `example_workflows`

The following output is based on example inputs from the DiLightNet repo given a lighting environment that has a reddish hue.
![dilight_example](https://github.com/user-attachments/assets/2c143014-0e4d-443b-9d75-401e084101a3)
