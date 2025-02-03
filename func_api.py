import os
import torch
import random
import nodes
import folder_paths as folders

from typing import List, Tuple, Union
from controlnet_aux.processor import Processor as ControlnetAux
from func_download import auto_download
from func_image import load_image, pil2tensor, tensor2pil
from PIL import Image

PREPROCESSORS = [
    'canny',
    'depth_leres',
    'depth_leres++',
    'depth_midas',
    'depth_zoe',
    'dwpose',
    'lineart_anime',
    'lineart_coarse',
    'lineart_realistic',
    'mediapipe_face',
    'mlsd',
    'normal_bae',
    'openpose',
    'openpose_face',
    'openpose_faceonly',
    'openpose_full',
    'openpose_hand',
    'scribble_hed',
    'scribble_hedsafe',
    'scribble_pidinet',
    'scribble_pidsafe',
    'shuffle',
    'softedge_hed',
    'softedge_hedsafe',
    'softedge_pidinet',
    'softedge_pidsafe',
]

CONTROLNET_LINKS = {
    'control_v11p_sd15_inpaint.safetensors': 'https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/diffusion_pytorch_model.safetensors',
    'control_v11p_sd15_lineart.safetensors': 'https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/diffusion_pytorch_model.safetensors',
    'control_v11f1p_sd15_depth.safetensors': 'https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/diffusion_pytorch_model.safetensors',
    'control_v11p_sd15_openpose.safetensors': 'https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors',
}

def create_controlnet_images(processor: str, images: List, device='cuda'):
    model = None
    if processor and processor != 'none':
        model = ControlnetAux(processor)
        if hasattr(model, 'to') and callable(model.to):
            model.to(device)

    result = []
    for image in images:
        cnet_image = load_image(image)
        if model:
            pred = model(cnet_image)
            cnet_image = pred.resize(cnet_image.size)
        result.append(cnet_image)
    del model
    return result

class SD15Container:
    def __init__(self, civitai_token: str = ''):
        self.unet = None
        self.unet_f = None
        self.clip = None
        self.clip_f = None
        self.cnet = {}
        self.civitai_token = civitai_token


    def load_checkpoint(self, ldm_path: str, vae_path: str = None):
        CheckpointLoader = nodes.CheckpointLoaderSimple()
        VAELoader = nodes.VAELoader()

        with torch.inference_mode():
            print('Loading Checkpoint, VAE, CLIP ...')
            ckpt_dir = os.path.join(folders.models_dir, 'checkpoints')
            ckpt_path = auto_download(ldm_path, ckpt_dir, self.civitai_token)
            ckpt_name = os.path.basename(ckpt_path)
            self.unet, self.clip, self.vae = CheckpointLoader.load_checkpoint(ckpt_name)
            if vae_path:
                vae_dir = os.path.join(folders.models_dir, 'vae')
                vae_path = auto_download(vae_path, vae_dir, self.civitai_token)
                self.vae = VAELoader.load_vae(os.path.basename(vae_path))[0]
            self.unet_f, self.clip_f = self.unet, self.clip

        return self.unet, self.clip, self.vae

    def load_loras(self, loras: List[Tuple[str, float]]):
        self.unet_f, self.clip_f = self.unet, self.clip

        LoraLoader = nodes.LoraLoader()
        lora_dir = os.path.join(folders.models_dir, 'loras')

        for index, item in enumerate(loras):
            path, weight = item
            if not path:
                continue

            path = auto_download(path, lora_dir, self.civitai_token)

            if not os.path.isfile(path):
                print(f'LoRA {path} is not found; pass')
                continue

            with torch.inference_mode():
                self.unet_f, self.clip_f = LoraLoader.load_lora(
                    self.unet_f,
                    self.clip_f,
                    os.path.basename(path),
                    weight,
                    weight,
                )
                print(f'LoRA {path} is loaded; weight: {weight}')

        return self.unet_f, self.clip_f
    
    def apply_controlnet(self, pos_cond, neg_cond, controlnet: str, image: Union[str, Image.Image], strength: float = 1.0, start: float = 0.0, end: float = 1.0):
        ControlNetLoader = nodes.ControlNetLoader()
        ControlNetSetter = nodes.ControlNetApplyAdvanced()

        if controlnet not in self.cnet:
            cnet_dir = os.path.join(folders.models_dir, 'controlnet')
            link = CONTROLNET_LINKS.get(controlnet, '')
            name = os.path.basename(controlnet) if link else None
            path = auto_download((link or path), cnet_dir, self.civitai_token, name)
            self.cnet[controlnet] = ControlNetLoader.load_controlnet(name)[0]
            print(f'Controlnet {path} is first loaded')

        image = pil2tensor(load_image(image))
        pos_cond, neg_cond = ControlNetSetter.apply_controlnet(
            pos_cond,
            neg_cond,
            self.cnet[controlnet],
            image,
            float(strength),
            float(start),
            float(end),
        )
        return pos_cond, neg_cond

    @torch.inference_mode()
    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        base_image: Union[str, Image.Image] = None,
        mask_image: Union[str, Image.Image] = None,
        controlnet: List[Tuple] = [],
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        steps: int = 30,
        cfg: float = 7.5,
        sampler_name: str = 'euler',
        scheduler: str = 'simple',
        denoise: float = 1.0,
        batch_size: int = 1,
        hires_fix: bool = False,
        hires_fix_scale_by: float = 1.5,
    ):
        KSampler = nodes.KSampler()
        CLIPTextEncode = nodes.CLIPTextEncode()
        EmptyLatentImage = nodes.EmptyLatentImage()
        LatentUpscaleBy = nodes.LatentUpscaleBy()
        LoadImage = nodes.LoadImage()
        LoadImageMask = nodes.LoadImageMask()
        InpaintCondition = nodes.InpaintModelConditioning()
        VAEDecode = nodes.VAEDecode()
        VAEEncode = nodes.VAEEncode()

        pos_cond = CLIPTextEncode.encode(self.clip_f, positive_prompt)[0]
        neg_cond = CLIPTextEncode.encode(self.clip_f, negative_prompt)[0]

        for item in controlnet:
            pos_cond, neg_cond = self.apply_controlnet(pos_cond,neg_cond, item[0], item[1], item[2], item[3], item[4])

        latent_image = EmptyLatentImage.generate(width, height)[0]

        if base_image and not mask_image:
            base_image = pil2tensor(load_image(base_image))
            latent_image = VAEEncode.encode(self.vae, base_image)[0]
        elif base_image and mask_image:
            mask_image = pil2tensor(load_image(mask_image).convert('L'))
            base_image = pil2tensor(load_image(base_image))
            pos_cond, neg_cond, latent_image = InpaintCondition.encode(pos_cond, neg_cond, base_image, self.vae, mask_image)

        images = []
        for i in range(batch_size):
            seed_it = seed if seed > 1 else random.randint(0, 18446744073709552000)
            print(f'#{i} used seed:', seed_it)

            sample = KSampler.sample(
                self.unet_f,
                seed=seed_it,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=pos_cond,
                negative=neg_cond,
                latent_image=latent_image,
                denoise=1.0,
            )[0]

            if hires_fix:
                print('hires fix ...')
                latent_image = LatentUpscaleBy.upscale(sample, 'bislerp', hires_fix_scale_by)[0]
                sample = KSampler.sample(
                    self.unet_f,
                    seed=seed_it,
                    steps=int(steps * denoise),
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=pos_cond,
                    negative=neg_cond,
                    latent_image=latent_image,
                    denoise=denoise,
                )[0]

            decoded = VAEDecode.decode(self.vae, sample)[0].detach()[0]
            images.append(tensor2pil(decoded))
        return images
