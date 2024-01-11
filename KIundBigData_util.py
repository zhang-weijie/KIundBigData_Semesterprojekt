from PIL import ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from PIL import Image


# move this class to util due to num_workers
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class EnhanceContrast:
    def __init__(self, enhancement_factor=1.5):
        self.enhancement_factor = enhancement_factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.enhancement_factor)
        return img


class GaussianBlur:
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))
