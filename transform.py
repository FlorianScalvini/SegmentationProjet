import torch
import torchvision.transforms
from torchvision import transforms as T
from torchvision.transforms.functional import hflip, rotate, InterpolationMode
import random
import numpy as np


def returnValue(image, depth, label):
    if depth is not None:
        return image, depth, label
    else:
        return image, label


class Transform:
    def __init__(self, transforms=None):
        if transforms is None:
            transforms = {
                "commun" : [],
                "depth": [],
                "color": [],
            }
        if not isinstance(transforms, dict):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        if "commun" in transforms.keys():
            self.trans_commun = transforms["commun"]
        else:
            self.trans_commun = []

        if "depth" in transforms.keys():
            self.trans_depth = transforms["depth"]
            self._addTotensor(self.trans_depth)
        else:
            self.trans_depth = []

        if "color" in transforms.keys():
            self.trans_color = transforms["color"]
            self._addTotensor(self.trans_color)
        else:
            self.trans_color = []
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

    def __call__(self, image, depth=None, label=None):
        for transform in self.trans_commun:
            image, depth, label = transform(image=image, depth=depth, label=label)
        if depth is not None:
            for transform in self.trans_depth:
                depth = transform(image=depth)
        for transform in self.trans_color:
            image = transform(image=image)
        if label is not None:
            label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return image, depth, label



class ToTensor:
    def __init__(self):
        self.toTensor = T.ToTensor()

    def __call__(self, image):
        image = self.toTensor(image)
        return image


class Blur:
    def __init__(self):
        return

    def __call__(self, image):
        sigma = random.random()
        ksize = int(3.3 * sigma)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        gaussian_blur = T.GaussianBlur(ksize, sigma)
        image = gaussian_blur(image)
        return image


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

    def __call__(self, image, label=None, depth=None):
        image = self.resizeIm(image)
        if depth is not None:
            depth = self.resizeIm(depth)
        return image, depth, label


class HorizontalFlip:
    def __init__(self, prob=0.5):
        try:
            prob = float(prob)
        except:
            raise ValueError("The probability is not a float")
        self.prob = prob
        return

    def __call__(self, image, label=None, depth=None):
        if random.random() > self.prob:
            image = hflip(image)
            if label is not None:
                label = hflip(label)
            if depth is not None:
                depth = hflip(depth)
        return image, depth, label


class ColorJitter:
    def __init__(self, brightness, contrast, saturation):
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

    def __call__(self, image):
        image = self.color_jitter(image)
        return image


class RandomScaleCrop:
    def __init__(self, scale=(0.5, 2), size=(512,512)):
        if scale[0] > scale [1]:
            raise ValueError("Maximum value is lower than minimun value")
        self.min = scale[0]
        self.max = scale[1]
        self.cropSize = size

    def __call__(self, image, label=None, depth=None, *args, **kwargs):
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
        if depth is not None:
            depth = T.functional.crop(depth, t, l, h, w)
        if label is not None:
            lbl_resize = torchvision.transforms.Resize((h_new, w_new), InterpolationMode.NEAREST)
            label = lbl_resize(label)
            label = T.functional.crop(label, t, l, h, w)
        if depth is not None:
            lbl_resize = torchvision.transforms.Resize((h_new, w_new), InterpolationMode.NEAREST)
            depth = lbl_resize(depth)
            depth = T.functional.crop(depth, t, l, h, w)
        return image, depth, label


class RandomScale:
    def __init__(self, min=0.5, max=2):
        if max < min:
            raise ValueError('RandomScale : Maximum value is lower than minimum value')
        self.min = min
        self.max = max
        self.random_crop = RandomCrop()


    def __call__(self, image, label=None, depth=None, *args, **kwargs):
        w, h = image.size
        scale = random.uniform(self.min, self.max)
        w_new, h_new = int(w*scale), int(h*scale)
        img_resize = torchvision.transforms.Resize((w_new, h_new), InterpolationMode.BILINEAR)
        image = img_resize(image)
        if label is not None:
            lbl_resize = torchvision.transforms.Resize((w_new, h_new), InterpolationMode.NEAREST)
            label = lbl_resize(label)
        if depth is not None:
            lbl_resize = torchvision.transforms.Resize((w_new, h_new), InterpolationMode.NEAREST)
            depth = lbl_resize(depth)
        return image, depth, label


class RandomCrop:
    def __init__(self, size=(512, 512)):
        if len(size) != 2:
            raise ValueError("Size invalid")
        self.cropSize = size

    def __call__(self, image, label=None, depth=None):
        t, l, h, w = T.RandomCrop.get_params(image, output_size=self.cropSize)
        image = T.functional.crop(image, t, l, h, w)
        if depth is not None:
            depth = T.functional.crop(depth, t, l, h, w)
        if label is not None:
            label = T.functional.crop(label, t, l, h, w)
        return image, depth, label


class Rotate:
    def __init__(self,  angleMax=10):
        self.angleMax = angleMax

    def __call__(self, image, label=None, depth=None):
        angle = random.randint(-self.angleMax, self.angleMax)
        image = rotate(image, angle=angle)
        if label is not None:
            label = rotate(label, angle=angle)
        if depth is not None:
            depth = rotate(depth, angle=angle)
        return image, depth, label


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.toNormalize = T.Normalize(mean=self.mean, std=self.std)

    def __call__(self, image):
        image = self.toNormalize(image)
        return image

