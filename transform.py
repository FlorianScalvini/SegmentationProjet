import torch
import torchvision.transforms
from torchvision import transforms as T
from torchvision.transforms.functional import hflip, rotate, InterpolationMode
import random
import numpy as np


class Transform:
    def __init__(self, transforms=None):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self._addTotensor(transforms)
        self.transforms = torchvision.transforms.Compose(transforms=transforms)

    @staticmethod
    def _addTotensor(transforms):
        for i in range(len(transforms)):
            if isinstance(transforms[i], ToTensor):
                return
            elif isinstance(transforms[i], Normalize):
                transforms.insert(i, ToTensor())
                return
        transforms.append(ToTensor())
        return transforms

    def __call__(self, image, label=None):
        if label is not None:
            image, label = self.transforms(image, label)
            return image, label
        else:
            image = self.transforms(image)
            return image

class ToTensor:
    def __init__(self):
        self.toTensor = T.ToTensor()

    def __call__(self, image, label):
        image = self.toTensor(image)
        return image, label


class Blur:
    def __init__(self):
        return

    def __call__(self, image, label=None):
        sigma = random.random()
        ksize = int(3.3 * sigma)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        gaussian_blur = T.GaussianBlur(ksize, sigma)
        image = gaussian_blur(image)
        return image, label


class Resize:
    def __init__(self, size=(512, 512), interpolation='LINEAR'):
        interpolation_dict = {
            'NEAREST': InterpolationMode.NEAREST,
            'LINEAR': InterpolationMode.BILINEAR,
            'CUBIC': InterpolationMode.BICUBIC,
            'LANCZOS': InterpolationMode.LANCZOS,
        }
        self.interpolation = interpolation_dict[interpolation]
        if len(size) != 2:
            raise ValueError("Size invalid")
        self.resize = size
        self.resizeIm = T.Resize(self.resize, interpolation=InterpolationMode.BILINEAR)
        self.resizeLabel = T.Resize(self.resize, interpolation=InterpolationMode.NEAREST)

    def __call__(self, image, label=None):
        image = self.resizeIm(image)
        return image, label


class HorizontalFlip:
    def __init__(self, prob=0.5):
        try:
            prob = float(prob)
        except:
            raise ValueError("The probability is not a float")
        self.prob = prob
        return

    def __call__(self, image, label=None):
        if random.random() > self.prob:
            image = hflip(image)
            if label is not None:
                label = hflip(label)
        return image, label


class ColorJitter:
    def __init__(self, brightness, contrast, saturation):
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

    def __call__(self, image, label=None):
        image = self.color_jitter(image)
        return image, label


class RandomScaleCrop:
    def __init__(self, scale=(0.5, 2), size=(512,512)):
        if scale[0] > scale [1]:
            raise ValueError("Maximum value is lower than minimun value")
        self.min = scale[0]
        self.max = scale[1]
        self.cropSize = size

    def __call__(self, image, label=None, *args, **kwargs):
        w, h = image.size
        scale = random.uniform(self.min, self.max)
        w_new, h_new = int(w * scale), int(h * scale)
        if w_new < self.cropSize[1]:
            w_new = self.cropSize[1]
        if h_new < self.cropSize[0]:
            h_new = self.cropSize[0]
        img_resize = torchvision.transforms.Resize((h_new, w_new), InterpolationMode.BILINEAR)
        image = img_resize(image)
        t, l, h, w = T.RandomCrop.get_params(image, output_size=self.cropSize)
        image = T.functional.crop(image, t, l, h, w)
        if label is not None:
            lbl_resize = torchvision.transforms.Resize((h_new, w_new), InterpolationMode.NEAREST)
            label = lbl_resize(label)
            label = T.functional.crop(label, t, l, h, w)
        return image, label


class RandomScale:
    def __init__(self, min=0.5, max=2):
        if max < min:
            raise ValueError('RandomScale : Maximum value is lower than minimum value')
        self.min = min
        self.max = max
        self.random_crop = RandomCrop()


    def __call__(self, image, label=None, *args, **kwargs):
        w, h = image.size
        scale = random.uniform(self.min, self.max)
        w_new, h_new = int(w*scale), int(h*scale)
        img_resize = torchvision.transforms.Resize((w_new, h_new), InterpolationMode.BILINEAR)
        image = img_resize(image)
        if label is not None:
            lbl_resize = torchvision.transforms.Resize((w_new, h_new), InterpolationMode.NEAREST)
            label = lbl_resize(label)
        return image, label


class RandomCrop:
    def __init__(self, size=(512, 512)):
        if len(size) != 2:
            raise ValueError("Size invalid")
        self.cropSize = size

    def __call__(self, image, label=None):
        t, l, h, w = T.RandomCrop.get_params(image, output_size=self.cropSize)
        image = T.functional.crop(image, t, l, h, w)
        if label is not None:
            label = T.functional.crop(label, t, l, h, w)
        return image, label


class Rotate:
    def __init__(self,  angleMax=10):
        self.angleMax = angleMax

    def __call__(self, image, label=None):
        angle = random.randint(-self.angleMax, self.angleMax)
        image = rotate(image, angle=angle)
        if label is not None:
            label = rotate(label, angle=angle)
        return image, label


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.toNormalize = T.Normalize(mean=self.mean, std=self.std)

    def __call__(self, image, label=None):
        image = self.toNormalize(image)
        return image, label

