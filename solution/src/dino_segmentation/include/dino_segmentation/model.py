import ctypes
import os

# DINO imports
import numpy as np
import torch

from dt_segmentation import DINOSeg, parse_class_names

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Wrapper():
    def __init__(self, model_name):
        weight_file_path = f"/code/solution/nn_models/{model_name}"
        self.model = AMD64Model(weight_file_path)

    def predict(self, image):
        return self.model.infer(image)


class Model():
    def __init__(self):
        pass

    def infer(self, image):
        raise NotImplementedError()


class AMD64Model():
    def __init__(self, weight_file_path):
        super().__init__()

        import torch

        torch.hub.set_dir('/code/solution/nn_models')
        self.model = DINOSeg.load_from_checkpoint(f'{weight_file_path}.ckpt')
        self.labels_path = f'/code/solution/nn_models/labels.txt'
        try:
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()
        except Exception:
            self.model = self.model.cpu()

    def infer(self, image):

        # Get class names and length
        class_names, _ = parse_class_names(self.labels_path)
        pred = self.model.predict(image)

        return pred, class_names
