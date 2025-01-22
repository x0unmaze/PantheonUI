import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from typing import Dict, List, Tuple

import torch


def cv22pil(cv2_img: np.ndarray) -> Image.Image:
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)


def pil2cv2(pil_img: Image.Image) -> np.array:
    np_img_array = np.asarray(pil_img)
    return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)


def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def mask_grow(mask: Image.Image, size: int) -> Image.Image:
    if size == 0:
        return mask
    size = size - 1 if size % 2 == 0 else size
    if size > 0:
        return mask.filter(ImageFilter.MaxFilter(size))
    else:
        return mask.filter(ImageFilter.MinFilter(size * -1))


def mask_blur(mask: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(size))


def resize_and_center_crop(image: Image.Image, max_size: int = 1024, div_size: int = 8) -> Image.Image:
    width, height = image.size
    ratio = max(width / max_size, height / max_size)
    width = int(round(width / ratio))
    height = int(round(height / ratio))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    final_width = div_size * round(width / div_size)
    final_height = div_size * round(height / div_size)
    left = (width - final_width) // 2
    top = (height - final_height) // 2
    right = final_width + left
    bottom = final_height + top
    result = resized.crop((left, top, right, bottom))
    return result


def make_image_grid(images: List[Image.Image], cols: int = 2, size: int = None, background: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color=background)
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    if size:
        grid = resize_and_center_crop(grid, size, 8)
    return grid


def blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0

    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]
    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    d = image - alpha * blurred_F - (1 - alpha) * blurred_B
    F = blurred_F + alpha * d
    F = np.clip(F, 0, 1)
    return F, blurred_B


def blur_fusion_foreground_estimator_with_alpha(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = blur_fusion_foreground_estimator(
        image,
        image,
        image,
        alpha,
        r,
    )
    return blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]


def refine_foreground(image, mask, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0
    f = blur_fusion_foreground_estimator_with_alpha(image, mask, r)
    return Image.fromarray((f * 255.0).astype(np.uint8))


def calculate_mean_std(image: Image.Image):
    mean, std = cv2.meanStdDev(image)
    mean = np.hstack(np.around(mean, decimals=2))
    std = np.hstack(np.around(std, decimals=2))
    return mean, std


def color_adapter(image: Image.Image, ref_image: Image.Image) -> Image.Image:
    image = pil2cv2(image)
    ref_image = pil2cv2(ref_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_mean, image_std = calculate_mean_std(image)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2LAB)
    ref_image_mean, ref_image_std = calculate_mean_std(ref_image)
    _image = (image - image_mean) * (ref_image_std / image_std)
    _image = _image + ref_image_mean
    np.putmask(_image, _image > 255, values=255)
    np.putmask(_image, _image < 0, values=0)
    ret_image = cv2.cvtColor(cv2.convertScaleAbs(_image), cv2.COLOR_LAB2BGR)
    return cv22pil(ret_image)


def image_blend(background: Image.Image, foreground: Image.Image, percentage: float):
    blend_mask = Image.new('L', background.size, round(percentage * 255))
    blend_mask = ImageOps.invert(blend_mask)
    image_result = Image.composite(background, foreground, blend_mask)
    return image_result


def gradient_image(size: Tuple[int, int], mode: str = 'horizontal', colors: Dict = None, tolerance: int = 0) -> Image.Image:
    if colors is None:
        colors = {0: [0, 0, 0], 50: [50, 50, 50], 100: [100, 100, 100]}

    colors = {int(k): [int(c) for c in v] for k, v in colors.items()}
    colors[0] = colors[min(colors.keys())]
    colors[255] = colors[max(colors.keys())]

    img = Image.new('RGB', size, color=(0, 0, 0))

    color_stop_positions = sorted(colors.keys())
    color_stop_count = len(color_stop_positions)
    spectrum = []
    for i in range(256):
        start_pos = max(p for p in color_stop_positions if p <= i)
        end_pos = min(p for p in color_stop_positions if p >= i)
        start = colors[start_pos]
        end = colors[end_pos]

        if start_pos == end_pos:
            factor = 0
        else:
            factor = (i - start_pos) / (end_pos - start_pos)

        r = round(start[0] + (end[0] - start[0]) * factor)
        g = round(start[1] + (end[1] - start[1]) * factor)
        b = round(start[2] + (end[2] - start[2]) * factor)
        spectrum.append((r, g, b))

    draw = ImageDraw.Draw(img)
    if mode == 'horizontal':
        for x in range(size[0]):
            pos = int(x * 100 / (size[0] - 1))
            color = spectrum[pos]
            if tolerance > 0:
                color = tuple([
                    round(c / tolerance) * tolerance for c in color
                ])
            draw.line((x, 0, x, size[1]), fill=color)
    elif mode == 'vertical':
        for y in range(size[1]):
            pos = int(y * 100 / (size[1] - 1))
            color = spectrum[pos]
            if tolerance > 0:
                color = tuple([
                    round(c / tolerance) * tolerance for c in color
                ])
            draw.line((0, y, size[0], y), fill=color)

    blur = 1.5
    if size[0] > 512 or size[1] > 512:
        multiplier = max(size[0], size[1]) / 512
        if multiplier < 1.5:
            multiplier = 1.5
        blur = blur * multiplier

    img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return img


def RGB2YCbCr(t):
    YCbCr = t.detach().clone()
    YCbCr[:, :, :, 0] = 0.2123 * t[:, :, :, 0] + \
        0.7152 * t[:, :, :, 1] + 0.0722 * t[:, :, :, 2]
    YCbCr[:, :, :, 1] = 0 - 0.1146 * t[:, :, :, 0] - \
        0.3854 * t[:, :, :, 1] + 0.5 * t[:, :, :, 2]
    YCbCr[:, :, :, 2] = 0.5 * t[:, :, :, 0] - \
        0.4542 * t[:, :, :, 1] - 0.0458 * t[:, :, :, 2]
    return YCbCr


def YCbCr2RGB(t):
    RGB = t.detach().clone()
    RGB[:, :, :, 0] = t[:, :, :, 0] + 1.5748 * t[:, :, :, 2]
    RGB[:, :, :, 1] = t[:, :, :, 0] - 0.1873 * \
        t[:, :, :, 1] - 0.4681 * t[:, :, :, 2]
    RGB[:, :, :, 2] = t[:, :, :, 0] + 1.8556 * t[:, :, :, 1]
    return RGB


def cv_blur_tensor(images, dx, dy):
    if min(dx, dy) > 100:
        np_img = torch.nn.functional.interpolate(images.detach().clone(
        ).movedim(-1, 1), scale_factor=0.1, mode='bilinear').movedim(1, -1).cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(
                image, (dx // 20 * 2 + 1, dy // 20 * 2 + 1), 0)
        return torch.nn.functional.interpolate(torch.from_numpy(np_img).movedim(-1, 1), size=(images.shape[1], images.shape[2]), mode='bilinear').movedim(1, -1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx, dy), 0)
        return torch.from_numpy(np_img)


def image_add_grain(image: Image, scale: float = 0.5, strength: float = 0.5, saturation: float = 0.7, toe: float = 0.0, seed: int = 0) -> Image:
    image = pil2tensor(image.convert("RGB"))
    t = image.detach().clone()
    torch.manual_seed(seed)
    grain = torch.rand(t.shape[0], int(
        t.shape[1] // scale), int(t.shape[2] // scale), 3)

    YCbCr = RGB2YCbCr(grain)
    YCbCr[:, :, :, 0] = cv_blur_tensor(YCbCr[:, :, :, 0], 3, 3)
    YCbCr[:, :, :, 1] = cv_blur_tensor(YCbCr[:, :, :, 1], 15, 15)
    YCbCr[:, :, :, 2] = cv_blur_tensor(YCbCr[:, :, :, 2], 11, 11)

    grain = (YCbCr2RGB(YCbCr) - 0.5) * strength
    grain[:, :, :, 0] *= 2
    grain[:, :, :, 2] *= 3
    grain += 1
    grain = grain * saturation + \
        grain[:, :, :, 1].unsqueeze(3).repeat(1, 1, 1, 3) * (1 - saturation)

    grain = torch.nn.functional.interpolate(
        grain.movedim(-1, 1),
        size=(t.shape[1], t.shape[2]),
        mode='bilinear',
    ).movedim(1, -1)
    t[:, :, :, :3] = torch.clip(
        (1 - (1 - t[:, :, :, :3]) * grain) * (1 - toe) + toe, 0, 1)
    return tensor2pil(t)


def replace_subject_v1(base_img: Image.Image, dest_img: Image.Image, mask_img: Image.Image, blur_size: int = 12, grow_size: int = -8, blend_percentage: float = 0.3):
    simple_mask = mask_blur(mask_grow(mask_img, -1), 1)
    simple = Image.composite(base_img, dest_img, simple_mask)
    simple = color_adapter(simple, dest_img)
    blended = image_blend(simple, dest_img, blend_percentage)
    subject_mask = mask_blur(mask_grow(mask_img, grow_size), blur_size)
    final = Image.composite(blended, dest_img, subject_mask)
    return final
