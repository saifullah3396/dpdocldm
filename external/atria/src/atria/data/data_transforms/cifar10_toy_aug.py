from typing import List, Optional, Union

from atria.core.data.data_transforms import DataTransform


class Cifar10ToyAug(DataTransform):
    def __init__(self, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self._transform = self._initialize_transform()

    def _initialize_transform(self):
        from torchvision import transforms

        aug = [
            transforms.Resize(
                (32, 32)
            ),  # resizes the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(),  # Flips the image w.r.t horizontal axis
            transforms.RandomRotation(10),  # Rotates the image to a specified angle
            transforms.RandomAffine(
                0, shear=10, scale=(0.8, 1.2)
            ),  # Performs actions like zooms, change shear angles.
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),  # Set the color params
            transforms.ToTensor(),  # convert the image to tensor so that it can work with torch
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize all the images
        ]

        # generate torch transformation
        return transforms.Compose(aug)

    def __str__(self):
        return str(self._transform)

    def _apply_transform(self, image: "PIL.Image.Image") -> "torch.Tensor":
        return self._transform(image)
