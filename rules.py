from PIL import Image, ImageDraw, ImageOps
from importlib import reload
import parameter


class Rules:
    def __init__(self):
        reload(parameter)
        self.func_map = {
            'bench': self.__make_bench,
            'chair': self.__make_chair,
            'sofa': self.__make_sofa,
            'table': self.__make_table
        }

    def __is_valid_param(self, shape_name: str, param_vector: list):
        if param_vector is None:
            return False
        shape = parameter.Shape(shape_name)
        if len(shape.params) != len(param_vector):
            return False
        for i, p in enumerate(param_vector):
            if p < shape.params[i].info['min_value'] or p > shape.params[i].info['max_value']:
                return False
        return True

    def make_shape(self, shape_name: str, param_vector: list = None):
        image = Image.new('L', (512, 512))
        if not self.__is_valid_param(shape_name, param_vector):
            return ImageOps.invert(image)
        drawer = ImageDraw.Draw(image)
        image = self.func_map[shape_name](image, drawer, param_vector)
        image = ImageOps.invert(image)
        return image

    def __make_bench(self, image: Image, drawer: ImageDraw, param_vector: list):
        '''
        :param image: PIL image
        :param drawer: draws shapes on input image
        :param param_vector: procedural parameters
        :return: procedurally constructed image
        '''
        # Implement your rule between here
        height = param_vector[0]
        leg_height_percent = param_vector[1]
        leg_width = param_vector[2]
        seat_height = param_vector[3]
        num_hbars = int(param_vector[4])
        bottom_bar = int(param_vector[5])
        # draw the legs
        xy = (22, 256 - height // 2, 22 + leg_width, 256 + height // 2)
        drawer.rectangle(xy, fill=255)
        xy = (488 - leg_width, 256 - height // 2, 488, 256 + height // 2)
        drawer.rectangle(xy, fill=255)
        # draw the seat
        seat_location = 256 - height // 2 + int(height * (1.0 - leg_height_percent))
        xy = (22, seat_location - seat_height // 2, 488, seat_location + seat_height // 2)
        drawer.rectangle(xy, fill=255)
        # draw horizontal bars
        start = 256 - height // 2
        offset = (seat_location - seat_height // 2 - 256 + height // 2) // num_hbars
        for i in range(num_hbars):
            xy = (22, start, 488, start + offset // 2)
            drawer.rectangle(xy, fill=255)
            start = start + offset
        # draw the bottom bar
        if bottom_bar != 0:
            bbar_location = 256 - height // 2 + int(height * (1.0 - leg_height_percent/2))
            xy = (22, bbar_location - 2, 488, bbar_location + 2)
            drawer.rectangle(xy, fill=255)
        # and here
        return image

    def __make_chair(self, image: Image, drawer: ImageDraw, param_vector: list):
        '''
        :param image: PIL image
        :param drawer: draws shapes on input image
        :param param_vector: procedural parameters
        :return: procedurally constructed image
        '''
        # Implement your rule between here
        # and here
        return image

    def __make_sofa(self, image: Image, drawer: ImageDraw, param_vector: list):
        '''
        :param image: PIL image
        :param drawer: draws shapes on input image
        :param param_vector: procedural parameters
        :return: procedurally constructed image
        '''
        # Implement your rule between here
        # and here
        return image

    def __make_table(self, image: Image, drawer: ImageDraw, param_vector: list):
        '''
        :param image: PIL image
        :param drawer: draws shapes on input image
        :param param_vector: procedural parameters
        :return: procedurally constructed image
        '''
        # Implement your rule between here
        # and here
        return image
