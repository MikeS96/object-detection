import ctypes
import os
from integration import MODEL_NAME, DT_TOKEN

# DINO imports
import numpy as np
import torch

from PIL import Image
from dt_segmentation.pl_torch_modules import DINOSeg, get_transforms
from dt_segmentation.dt_utils import parse_class_names

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def run(input, exception_on_failure=False):
    print(input)
    try:
        import subprocess
        program_output = subprocess.check_output(f"{input}", shell=True, universal_newlines=True,
                                                 stderr=subprocess.STDOUT)
    except Exception as e:
        if exception_on_failure:
            print(e.output)
            raise e
        program_output = e.output
    print(program_output)
    return program_output.strip()


class Wrapper():
    def __init__(self, aido_eval=False):
        model_name = MODEL_NAME()

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

        with torch.no_grad():
            # Get transforms
            transforms = get_transforms()

            # Get class names and length
            class_names, _ = parse_class_names(self.labels_path)

            # TODO size should be read from one place
            x_transformed = transforms(Image.fromarray(np.uint8(image)).convert('RGB'))
            pred = self.model.predict(x_transformed.unsqueeze(0)).reshape((60, 60))

        return pred, class_names


class TRTModel(Model):
    def __init__(self, weight_file_path):
        super().__init__()
        ctypes.CDLL(weight_file_path + ".so")
        from dino_segmentation.tensorrt_model import YoLov5TRT
        self.model = YoLov5TRT(weight_file_path + ".engine")

    def infer(self, image):
        # todo ensure this is in boxes, classes, scores format
        results = self.model.infer_for_robot([image])
        boxes = results[0][0]
        confs = results[0][1]
        classes = results[0][2]

        if classes.shape[0] > 0:
            return boxes, classes, confs
        return [], [], []
