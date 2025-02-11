# PantheonUI-LTXVideo
PantheonUI-LTXVideo is a collection of custom nodes for PantheonUI designed to integrate the LTXVideo diffusion model. These nodes enable workflows for text-to-video, image-to-video, and video-to-video generation. The main LTXVideo repository can be found [here](https://github.com/Lightricks/LTX-Video).

## 22.12.2024

Fixed a bug which caused the model to produce artifacts on short negative prompts when using a native CLIP Loader node.

## 19.12.2024 ⭐ Update ⭐
1. Improved model - removes "strobing texture" artifacts and generates better motion. Download from [here](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.1.safetensors).
2. STG support
3. Integrated image degradation system for improved motion generation.
3. Additional initial latent optional input to chain latents for high res generation.
4. Image captioning in image to video [flow](assets/ltxvideo-i2v.json).

## Installation

Installation via [PantheonUI-Manager](https://github.com/ltdrdata/PantheonUI-Manager) is preferred. Simply search for `PantheonUI-LTXVideo` in the list of nodes and follow installation instructions.

### Manual installation

1. Install PantheonUI
2. Clone this repository to `custom-nodes` folder in your PantheonUI installation directory.
3. Install the required packages:
```bash
cd custom_nodes/PantheonUI-LTXVideo && pip install -r requirements.txt
```
For portable PantheonUI installations, run
```
.\python_embeded\python.exe -m pip install -r .\PantheonUI\custom_nodes\PantheonUI-LTXVideo\requirements.txt
```

### Models

1. Download [ltx-video-2b-v0.9.1.safetensors](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) from Hugging Face and place it under `models/checkpoints`.
2. Install one of the t5 text encoders, for example [google_t5-v1_1-xxl_encoderonly](https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/tree/main). You can install it using PantheonUI Model Manager.

## Example workflows

Note that to run the example workflows, you need to have some additional custom nodes, like [PantheonUI-VideoHelperSuite](https://github.com/kosinkadink/PantheonUI-VideoHelperSuite) and others, installed. You can do it by pressing "Install Missing Custom Nodes" button in PantheonUI Manager.

### Image-to-video

[Download workflow](assets/ltxvideo-i2v.json)
![workflow](assets/ltxvideo-i2v.png)

### Text-to-video

[Download workflow](assets/ltxvideo-t2v.json)
![workflow](assets/ltxvideo-t2v.png)
