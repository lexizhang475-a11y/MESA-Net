import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
import albumentations as A
from skimage.draw import disk
from skimage.filters import gaussian


class AddLightSpots(A.ImageOnlyTransform):
    """Synthetic light spots for endoscopic-style augmentation."""

    def __init__(self, radius_range=(5, 20), intensity=0.9, num_spots=1, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.radius_range = radius_range
        self.intensity = intensity
        self.num_spots = num_spots

    def apply(self, image, **params):
        h, w, c = image.shape
        light_layer = np.zeros((h, w), dtype=np.float32)

        for _ in range(self.num_spots):
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            radius = np.random.randint(self.radius_range[0], self.radius_range[1])
            rr, cc = disk((center_y, center_x), radius, shape=(h, w))
            light_layer[rr, cc] += self.intensity

        light_layer = gaussian(light_layer, sigma=np.mean(self.radius_range) / 2)
        light_layer = np.clip(light_layer, 0, 1)

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        for i in range(c):
            image[:, :, i] = np.clip(image[:, :, i] + light_layer, 0, 1)

        return (image * 255).astype(np.uint8)
