import torch
from torchvision.transforms import v2 as transforms


class AddGaussianNoise(object):

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            return img
        return img + torch.randn(img.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_train_transforms():
    """Photometric and mild geometric augmentation. Built on `torchvision.transforms.v2`, so a
    call ``t(image, keypoints)`` with `tv_tensors.KeyPoints` moves the points along with the ink."""
    return transforms.Compose([

        transforms.RandomApply([
            transforms.RandomAffine(degrees=2.0, translate=(0.02, 0.02), shear=2.0, fill=255)
        ], p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.RandomGrayscale(p=0.1),
    ])
