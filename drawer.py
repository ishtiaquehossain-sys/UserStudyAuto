from PIL.ImageDraw import ImageDraw


class Drawer(ImageDraw):
    def __init__(self, im):
        super(Drawer, self).__init__(im, mode=None)
        self.fill = 255

    def arc(self, xy, start, end):
        super().arc(xy, start, end, fill=self.fill, width=1)

    def bitmap(self, xy, bitmap):
        super().bitmap(xy, bitmap, fill=self.fill)

    def chord(self, xy, start, end):
        super().chord(xy, start, end, fill=self.fill, outline=None, width=1)

    def ellipse(self, xy):
        super().ellipse(xy, fill=self.fill, outline=None, width=1)

    def line(self, xy):
        super().line(xy, fill=self.fill, width=0, joint=None)

    def shape(self, shape):
        super().shape(shape, fill=self.fill, outline=None)

    def pieslice(self, xy, start, end):
        super().pieslice(xy, start, end, fill=self.fill, outline=None, width=1)

    def point(self, xy):
        super().point(xy, fill=self.fill)

    def polygon(self, xy):
        super().polygon(xy, fill=self.fill, outline=None)

    def regular_polygon(self, bounding_circle, n_sides, rotation=0):
        super().regular_polygon(bounding_circle, n_sides, rotation=0, fill=self.fill, outline=None)

    def rectangle(self, xy):
        super().rectangle(xy, fill=self.fill, outline=None, width=1)

    def rounded_rectangle(self, xy, radius=0):
        super().rounded_rectangle(xy, radius=0, fill=self.fill, outline=None, width=1)
