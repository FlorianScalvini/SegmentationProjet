import torch
from torchvision import transforms as T
from torchvision.transforms.functional import hflip, rotate, InterpolationMode
import random

class Transform:
    def __init__(self, transforms = []):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        for i in range(len(self.transforms)):
            if isinstance(self.transforms[i], Normalize):
                self.transforms.insert(i, ToTensor())
                return
        self.transforms.append(ToTensor())


    def __call__(self, im, label=None):
        for transform in self.transforms:
            outputs = transform(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]

        return (im, label)

class ToTensor():
    def __init__(self):
        self.toTensor =  T.ToTensor()

    def __call__(self, image, label=None):
        image = self.toTensor(image)
        if label is not None:
            label = self.toTensor(label)
        return (image, label)

class Blur():
    def __init__(self):
        return

    def __call__(self, image, label=None):
        sigma = random.random()
        ksize = int(3.3 * sigma)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        gaussian_blur = T.GaussianBlur(ksize, sigma)
        image = gaussian_blur(image)
        return (image, label)

class Resize():
    def __init__(self, size=(512, 512), interpolation='LINEAR'):

        interpolationDict = {
            'NEAREST': InterpolationMode.NEAREST,
            'LINEAR': InterpolationMode.BILINEAR,
            'CUBIC': InterpolationMode.BICUBIC,
            'LANCZOS': InterpolationMode.LANCZOS,
        }
        self.interpolation = interpolationDict[interpolation];
        if len(size) != 2:
            raise ValueError("Size invalid")
        self.resize = size
        self.resizeIm = T.Resize(self.resize, interpolation=InterpolationMode.BILINEAR)
        self.resizeLabel = T.Resize(self.resize, interpolation=InterpolationMode.NEAREST)

    def __call__(self, image, label=None):
        image = self.resizeIm(image)
        if label is not None:
            label = self.resizeLabel(label)
        return (image, label)

class HorizontalFlip():
    def __init__(self, prob=0.5):
        try:
            prob = float(prob)
        except:
            print("The probability is not a float")
        self.prob = prob
        return

    def __call__(self, image, label=None):
        if random.random() > self.prob:
            image = hflip(image)
            if label is not None:
                label = hflip(label)
        return image, label


class RandomCrop():
    def __init__(self, size=(512, 512)):
        if len(size) != 2:
            raise ValueError("Size invalid")
        self.cropSize = size

    def __call__(self, image, label=None):
        t, l, h, w = T.RandomCrop.get_params(image, output_size=self.cropSize)
        image = T.functional.crop(image, t, l, h, w)
        if label is not None:
            label = T.functional.crop(label, t, l, h, w)
        return (image, label)

class Rotate():
    def __init__(self,  angleMax=10):
        self.angleMax = angleMax

    def __call__(self, image, label=None):
        angle = random.randint(-self.angleMax, self.angleMax)
        image = rotate(image, angle=angle)
        if label is not None:
            label = rotate(label, angle=angle)
        return image, label


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.toNormalize = T.Normalize(mean=self.mean, std=self.std)
        self.toTensor = T.ToTensor()

    def __call__(self, image, label=None):
        if not isinstance(image, torch.Tensor):
            image = self.toTensor(image)
        image = self.toNormalize(image)
        if label is not None:
            if not isinstance(image, torch.Tensor):
                label = self.toTensor(label)
            label = self.toNormalize(label)
        return (image, label)



    def _val_augmentation(self, image, label):
        if self.resize:
            self._resize(image, label)
        if self.crop_size:
            image, label = self._crop(image, label)
        return image, label

    def _augmentation(self, image, label):
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            image = rotate(image, angle=angle)
            label = rotate(label, angle=angle)

        if self.resize:
            self._resize(image, label)

        # Padding to return the correct crop size
        if self.crop_size:
            image, label = self._crop(image, label)

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = hflip(image)
                label = hflip(label)

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            gaussian_blur = T.GaussianBlur(ksize, sigma)
            image = gaussian_blur(image)
        return image, label

import numpy as np
import PIL.Image as Image

if "__main__" == __name__:
    a = Image.open("/home/ubuntu/Bureau/grille.png")
    tr = Transform([Resize((224,244)), Normalize(32,32)])
    b = tr(a)
    print("")