from shapesorter import ShapeSorter
import shapegenerator
from importlib import reload


class Controller:
    def __init__(self):
        super(Controller, self).__init__()
        self.generator = None
        self.shape_sorter = ShapeSorter()
        self.shape_names = ['bench', 'chair']
        self.true_images = {}
        self.pred_images = {}
        self.rearranged_ids = {}
        self.summary_images = {}

    def start_training(self, shape_name: str):
        reload(shapegenerator)
        generator = shapegenerator.ShapeGenerator()
        shape = generator.shapes[shape_name]
        images, vectors = shape.make_training_data()
        if images:
            shape.predictor.init_dataset(images, vectors)
            shape.predictor.start()
            return shape.predictor
        else:
            print('No training data can be generated, make sure parameter vector is non-empty')
            return None

    def get_summary(self, shape_name: str):
        return self.shape_sorter.get_summary(shape_name)

    def get_details(self, shape_name: str, index: int):
        return self.shape_sorter.get_details(shape_name, index)
