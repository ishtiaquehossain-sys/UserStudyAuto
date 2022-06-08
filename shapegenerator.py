from model import Predictor
from rules import Rules


class ShapeGenerator:
    def __init__(self):
        self.rules = Rules()
        self.predictors = {
            'bench': Predictor('bench'),
            'chair': Predictor('chair')
        }

    def get_image(self, shape_name: str, index: int):
        '''
        :param shape_name: shape category
        :param index: shape index
        :return: A 512X512 PIL image
        '''
        param_vector = self.predictors[shape_name].predict(index)
        return self.rules.make_shape(shape_name, param_vector)
