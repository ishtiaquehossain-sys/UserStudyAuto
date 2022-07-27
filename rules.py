from PIL import Image, ImageOps
from importlib import reload
import parameter
from drawer import Drawer


class Rules:
    def __init__(self):
        reload(parameter)
        self.func_map = {
            'bench': self.__make_bench,
            'chair': self.__make_chair
        }

    def __is_valid_param(self, shape_name: str, param_vector: list):
        rel_tol = 1e-6
        if param_vector is None:
            return False
        shape = parameter.Shape(shape_name)
        if len(shape.params) != len(param_vector):
            print('Warning: Parameter vector length is incorrect')
            return False
        for i, p in enumerate(param_vector):
            min_value = shape.params[i].info['min_value']
            max_value = shape.params[i].info['max_value']
            if p < min_value and abs(p - min_value) > rel_tol:
                print('Warning: Parameter value out of range')
                return False
            elif p > max_value and abs(p - max_value) > rel_tol:
                print('Warning: Parameter value out of range')
                return False
        return True

    def make_shape(self, shape_name: str, param_vector: list = None):
        image = Image.new('L', (512, 512))
        if not self.__is_valid_param(shape_name, param_vector):
            return ImageOps.invert(image)
        drawer = Drawer(image)
        image = self.func_map[shape_name](image, drawer, param_vector)
        image = ImageOps.invert(image)
        return image

    def __make_bench(self, image: Image, drawer: Drawer, param_vector: list):
        '''
        :param image: PIL image
        :param drawer: draws shapes on input image
        :param param_vector: procedural parameters
        :return: procedurally constructed image
        '''
        # As example, rules for the bench class are implemented between here
        height = param_vector[0]
        leg_height = param_vector[1]
        leg_width = param_vector[2]
        seat_height = param_vector[3]
        num_hbars = int(param_vector[4])
        bottom_bar = int(param_vector[5])
        # draw the legs
        xy = (0.045, 0.5 - height / 2, 0.045 + leg_width, 0.5 + height / 2)
        drawer.rectangle(xy)
        xy = (0.955 - leg_width, 0.5 - height / 2, 0.955, 0.5 + height / 2)
        drawer.rectangle(xy)
        # draw the seat
        seat_location = 0.5 - height / 2 + (height - leg_height)
        xy = (0.045, seat_location - seat_height / 2, 0.955, seat_location + seat_height / 2)
        drawer.rectangle(xy)
        # draw horizontal bars
        start = 0.5 - height / 2
        offset = (seat_location - seat_height / 2 - 0.5 + height / 2) / num_hbars
        for i in range(num_hbars):
            xy = (0.045, start, 0.955, start + offset / 2)
            drawer.rectangle(xy)
            start = start + offset
        # draw the bottom bar
        if bottom_bar != 0:
            bbar_location = 0.5 - height / 2 + (height - leg_height) + leg_height / 2
            xy = (0.045, bbar_location - 0.006, 0.955, bbar_location + 0.006)
            drawer.rectangle(xy)
        # and here
        return image

    def __make_chair(self, image: Image, drawer: Drawer, param_vector: list):
        '''
        :param image: PIL image
        :param drawer: draws shapes on input image
        :param param_vector: procedural parameters
        :return: procedurally constructed image
        '''
        # Implement your rule for the chair class between here
        # and here
        return image
