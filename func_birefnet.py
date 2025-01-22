import os
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image
from func_image import refine_foreground


def load_birefnet(repo, device: str = 'cuda'):
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        repo,
        trust_remote_code=True,
    )
    birefnet.to(device)
    birefnet.eval()
    return birefnet


def birefnet_remove_background(birefnet, images, resolution=(1024, 1024), device: str = 'cuda'):
    items = []
    for idx_image, image_src in enumerate(images):
        if isinstance(image_src, str):
            if os.path.isfile(image_src):
                image_ori = Image.open(image_src)
        elif isinstance(image_src, Image.Image):
            image_ori = image_src
        else:
            raise Exception('Unsupported image type')

        transform_image = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = image_ori.convert('RGB')
        # Preprocess the image
        image_proc = transform_image(image)
        image_proc = image_proc.unsqueeze(0)

        # Prediction
        with torch.no_grad():
            preds = birefnet(image_proc.to(device))[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Show Results
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(image.size)
        image_masked = refine_foreground(image, pred_pil)
        image_masked.putalpha(pred_pil.resize(image.size))

        torch.cuda.empty_cache()
        items.append((pred_pil, image_masked))
    return items
