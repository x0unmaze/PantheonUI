import os
from datetime import datetime

def save_images(images, out_dir: str, prefix: str = '', subfix: str = '', ext: str = '.png'):
    os.makedirs(out_dir, exist_ok=True)
    for i, image in enumerate(images, start=1):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'{now}-{str(i).zfill(3)}'
        filename = f'{prefix}-{filename}' if prefix else filename
        filename = f'{filename}-{subfix}' if subfix else filename
        basename = f'{filename}{ext}'
        image.save(os.path.join(out_dir, basename))
