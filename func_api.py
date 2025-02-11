import os
import torch
import random
import nodes
import folder_paths as folders

from typing import List, Tuple, Union
from controlnet_aux.processor import Processor as ControlnetAux
from func_image import load_image, mask_grow, mask_blur, pil2tensor, tensor2pil
from PIL import Image
from custom_nodes.crop_and_stitch.inpaint_cropandstitch import InpaintCrop, InpaintStitch
from pantheon_extras.nodes_mask import ImageCompositeMasked

def create_controlnet_images(processor: str, images: List, device='cuda'):
    model = None
    if processor and processor != 'none':
        model = ControlnetAux(processor)
        if hasattr(model.processor, 'to') and callable(model.processor.to):
            print(f'{processor} is to {device}')
            model.processor.to(device)

    result = []
    for image in images:
        cnet_image = load_image(image)
        if model:
            pred = model(cnet_image)
            cnet_image = pred.resize(cnet_image.size)
        result.append(cnet_image)
    del model
    return result

def preparse_controlnet_inputs(model: str, base_image: str, mask_image: str = None, preprocessor: str='none', strength: float = 1.0, start: float = 0.0, end: float = 1.0):
    c_img = load_image(base_image).convert('RGB')

    if preprocessor:
        c_img = create_controlnet_images(preprocessor, [base_image], 'cuda')[0]

    if mask_image:
        b_img = Image.new('RGB', c_img.size) #black background
        m_img = Image.open(mask_image).convert('L')
        c_img = Image.composite(c_img, b_img, m_img)

    return (model, c_img, strength, start, end)

class SDContainer:
    def __init__(self, civitai_token: str = ''):
        self.unet = None
        self.unet_f = None
        self.clip = None
        self.clip_f = None
        self.vae = None
        self.cnet = {}
        self.civitai_token = civitai_token
        self.ldm_name = ''
        self.vae_name = ''


    def load_checkpoint(self, ldm_name: str, vae_name: str = None):
        CheckpointLoader = nodes.CheckpointLoaderSimple()
        VAELoader = nodes.VAELoader()

        with torch.inference_mode():
            if self.ldm_name != ldm_name:
                self.unet, self.clip, vae_b = CheckpointLoader.load_checkpoint(ldm_name)
                if not vae_name:
                    self.vae = vae_b
                    self.vae_name = vae_name
                self.ldm_name = ldm_name
                print(f'LDM {ldm_name} is loaded;')

            if self.vae_name != vae_name:
                self.vae = VAELoader.load_vae(vae_name)[0]
                self.vae_name = vae_name
                print(f'VAE {vae_name} is loaded;')

            self.unet_f, self.clip_f = self.unet, self.clip
        return self.unet, self.clip, self.vae

    def load_loras(self, loras: List[Tuple[str, float]]):
        self.unet_f, self.clip_f = self.unet, self.clip
        LoraLoader = nodes.LoraLoader()

        for item in loras:
            name, weight = item
            if not name:
                continue

            if not folders.get_full_path('loras', name):
                print(f'LoRA {name} is not found; passed')
                continue

            with torch.inference_mode():
                self.unet_f, self.clip_f = LoraLoader.load_lora(
                    self.unet_f,
                    self.clip_f,
                    name,
                    weight,
                    weight,
                )
                print(f'LoRA {name} is loaded; weight: {weight}')

        return self.unet_f, self.clip_f

    def apply_controlnet(self, pos_cond, neg_cond, controlnet: str, image: Union[str, Image.Image], strength: float = 1.0, start: float = 0.0, end: float = 1.0):
        if not folders.get_full_path('controlnet', controlnet):
            print(f'Controlnet {controlnet} is not found; passed')
            return pos_cond, neg_cond

        ControlNetLoader = nodes.ControlNetLoader()
        ControlNetSetter = nodes.ControlNetApplyAdvanced()

        if controlnet not in self.cnet:
            self.cnet[controlnet] = ControlNetLoader.load_controlnet(controlnet)[0]
            print(f'Controlnet {controlnet} is loaded;')

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
        print(f'Controlnet {controlnet} is append; strength: {strength}; {start} to {end}')
        return pos_cond, neg_cond

    @torch.inference_mode()
    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        base_image: Union[str, Image.Image] = None,
        mask_image: Union[str, Image.Image] = None,
        mask_grow_size: int = 0,
        mask_blur_size: int = 16,
        use_mask_invert: bool = False,
        use_stitch: bool = False,
        lora: List[Tuple[str, float]] = [],
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
        InpaintCondition = nodes.InpaintModelConditioning()
        ImageCompositer = ImageCompositeMasked()
        VAEDecode = nodes.VAEDecode()
        VAEEncode = nodes.VAEEncode()

        pos_cond = CLIPTextEncode.encode(self.clip_f, positive_prompt)[0]
        neg_cond = CLIPTextEncode.encode(self.clip_f, negative_prompt)[0]

        stitch = None

        if not base_image and not mask_image:
            task = 'txt2img'
            latent_image = EmptyLatentImage.generate(width, height)[0]
        elif base_image and mask_image:
            task = 'inpaint'
            mask_image = pil2tensor(mask_blur(mask_grow(load_image(mask_image).convert('L'), mask_grow_size), mask_blur_size))
            base_image = pil2tensor(load_image(base_image))
            if use_stitch:
                cropper = InpaintCrop()
                stitch, base_image, mask_image = cropper.inpaint_crop_single_image(base_image, mask_image, blur_mask_pixels=0, invert_mask=use_mask_invert)
            VAEEncodeInpaint = nodes.VAEEncodeForInpaint()
            latent_image = VAEEncodeInpaint.encode(self.vae, base_image, mask_image)[0]
        else:
            task = 'img2img'
            base_image = pil2tensor(load_image(base_image))
            latent_image = VAEEncode.encode(self.vae, base_image)[0]

        self.load_loras(lora)
        for item in controlnet:
            pos_cond, neg_cond = self.apply_controlnet(pos_cond,neg_cond, item[0], item[1], item[2], item[3], item[4])

        print('task:', task)
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
                denoise=denoise,
            )[0]

            if hires_fix:
                print('hires fix ...')

                if task == 'inpaint':
                    sample = VAEDecode.decode(self.vae, sample)[0]
                    ImageScaleBy = nodes.ImageScaleBy()
                    sample = ImageScaleBy.upscale(sample, 'bicubic', hires_fix_scale_by)[0]
                    latent_image = VAEEncode.encode(self.vae, sample)[0]
                    hires_denoise = 0.2
                else:
                    # latent upscale require denoise strength upper 0.5
                    hires_denoise = 0.5
                    latent_image = LatentUpscaleBy.upscale(sample, 'bislerp', hires_fix_scale_by)[0]

                sample = KSampler.sample(
                    self.unet_f,
                    seed=seed_it,
                    steps=int(steps * hires_denoise),
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=pos_cond,
                    negative=neg_cond,
                    latent_image=latent_image,
                    denoise=hires_denoise,
                )[0]

            decoded = VAEDecode.decode(self.vae, sample)[0].detach()

            if task == 'inpaint':
                decoded = ImageCompositer.composite(base_image, decoded, 0, 0, True, mask_image)[0]
                if use_stitch:
                    stitcher = InpaintStitch()
                    decoded = stitcher.inpaint_stitch_single_image(stitch, decoded)[0]

            images.append(tensor2pil(decoded))
        return images
