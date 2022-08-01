import bench
import chair
from importlib import reload


class ShapeGenerator:
    def __init__(self):
        reload(bench)
        reload(chair)
        self.shapes = dict()
        self.shapes['bench'] = bench.Bench()
        self.shapes['chair'] = chair.Chair()

    def get_image(self, shape_name: str, index: int):
        image = self.shapes[shape_name].get_image(index)
        return image
