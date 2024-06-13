import numpy as np
import torch
import torch.nn as nn


class TorchModelWrapper(nn.Module):
    def __init__(
        self, model, input_transforms, softmax_out=False, output_idxs=...,
        contrast=None, gpu=False
    ):
        super().__init__()
        self.model = model
        self.input_transforms = input_transforms
        self.softmax = None
        if softmax_out:
            self.softmax = nn.Softmax(dim=1)
        self.output_idxs = output_idxs
        self.contrast = contrast
        self.gpu = gpu
        
        if gpu:
            model.cuda()

    def __str__(self):
        content = f'{self.model.__class__.__name__}'
        if not self.softmax is None: content += f',Softmax()'
        return f'TorchModelWrapper({content})'

    def __call__(self, x):
        x = self.input_transforms(x)
        if self.gpu:
            x = x.cuda()
        y = self.model(x)
        if not self.softmax is None:
            y = self.softmax(y)
        y = y.numpy(force=True)[:, self.output_idxs]
        if not self.contrast is None:
            positive = np.sum(y[:,self.contrast[0]], axis=1)
            y = positive - np.sum(y[:,self.contrast[1]], axis=1)
        return y
    
class ImageToTorch(nn.Module):
    def __call__(self, pic):
        tensor = torch.from_numpy(pic.astype(np.float32))
        if len(tensor.shape) == 3:
            tensor = torch.unsqueeze(tensor, dim=0)
        return tensor.permute((0,3,1,2))