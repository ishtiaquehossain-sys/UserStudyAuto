import random
import numpy as np
from itertools import product
from drawer import Drawer
from PIL import Image, ImageOps
import parameter
from importlib import reload
from model import Predictor


class Shape:
    def __init__(self):
        reload(parameter)
        self.param_vector_info = parameter.ParameterVectorInfo()
        self.param_vectors = [None]*50
        self.drawer = None
        self.images = None
        self.setup_parameter_vector()
        self.predictor = Predictor()

    def add_parameter_vector_element(self, param_name: str, param_type: str, min_value=None, max_value=None):
        self.param_vector_info.add_element(param_name, param_type, min_value, max_value)

    def assign_parameter_vector_to_shape(self, index: int, param_vector: list):
        self.param_vectors[index] = parameter.ParameterVector(self.param_vector_info, param_vector)

    def make_shapes(self, param_vectors: list):
        images = []
        for i in range(len(param_vectors)):
            image = Image.new('L', (512, 512))
            if isinstance(param_vectors[i], parameter.ParameterVector) and param_vectors[i].is_valid():
                self.drawer = Drawer(image)
                self.procedure(param_vectors[i])
                image = ImageOps.invert(image)
            else:
                image = ImageOps.invert(image)
            images.append(image)
        return images

    def make_training_data(self):
        max_len = 6000
        param_domain = []
        for param in self.param_vector_info:
            if param.param_type == 's':
                param_domain.append(np.linspace(param.min_value, param.max_value, 10))
            else:
                param_domain.append(np.linspace(param.min_value, param.max_value, param.max_value-param.min_value+1))
        if not param_domain:
            param_vectors = []
        else:
            param_vectors = [p for p in product(*param_domain)]
        random.shuffle(param_vectors)
        if max_len is not None:
            if len(param_vectors) > max_len:
                param_vectors = random.choices(param_vectors, k=max_len)
        param_vectors = [parameter.ParameterVector(self.param_vector_info, p) for p in param_vectors]
        images = self.make_shapes(param_vectors)
        return images, param_vectors

    def arc(self, xy, start, end):
        self.drawer.arc(xy, start, end)

    def chord(self, xy, start, end):
        self.drawer.chord(xy, start, end)

    def ellipse(self, xy):
        self.drawer.ellipse(xy)

    def line(self, xy):
        self.drawer.line(xy)

    def pieslice(self, xy, start, end):
        self.drawer.pieslice(xy, start, end)

    def point(self, xy):
        self.drawer.point(xy)

    def regular_polygon(self, bounding_circle, n_sides, rotation=0):
        self.drawer.regular_polygon(bounding_circle, n_sides, rotation=rotation)

    def rectangle(self, xy):
        self.drawer.rectangle(xy)

    def rounded_rectangle(self, xy, radius=0):
        self.drawer.rounded_rectangle(xy, radius=radius)

    def get_image(self, index: int):
        return self.images[index]

    def setup_parameter_vector(self):
        raise NotImplementedError()

    def procedure(self, param_vector: parameter.ParameterVector):
        raise NotImplementedError()

    def specify_parameter_vectors(self):
        for i in range(50):
            self.assign_parameter_vector_to_shape(i, self.predictor.predict(i))
