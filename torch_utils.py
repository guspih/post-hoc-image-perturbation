import numpy as np
import torch
import torch.nn as nn

from image_perturbers import cast_image


class TorchModelWrapper(nn.Module):
    def __init__(self, model, input_transforms, output_idxs=...):
        super().__init__()
        self.model = model
        self.input_transform = input_transforms
        self.output_idxs = output_idxs
    
    def __call__(self, x):
        x = self.input_transform(x)
        y = self.model(x).detach().numpy()
        return y[self.output_idxs]
    
class ImageToTorch(nn.Module):
    def __call__(self, pic):
        tensor = torch.from_numpy(cast_image(pic, dtype=np.float32))
        if len(tensor.shape) == 3:
            tensor = torch.unsqueeze(tensor, dim=0)
        return tensor.permute((0,3,1,2))