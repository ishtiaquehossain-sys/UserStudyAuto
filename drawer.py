from PIL import Image
from PIL.ImageDraw import ImageDraw


class Drawer(ImageDraw):
    def __init__(self, im: Image):
        super(Drawer, self).__init__(im, mode=None)
        self.fill = 255
        self.width = im.width
        self.height = im.height

    def relative_bounding_box_to_absolute(self, xy):
        top_left_x = self.width * xy[0]
        top_left_y = self.height * xy[1]
        bottom_right_x = self.width * xy[2]
        bottom_right_y = self.height * xy[3]
        return top_left_x, top_left_y, bottom_right_x, bottom_right_y

    def relative_bounding_circle_to_absolute(self, bounding_circle):
        center_x = self.width * bounding_circle[0]
        center_y = self.height * bounding_circle[1]
        radius = self.height * bounding_circle[2]
        return center_x, center_y, radius

    def arc(self, xy, start, end):
        xy = self.relative_bounding_box_to_absolute(xy)
        super().arc(xy, start, end, fill=self.fill, width=1)

    def chord(self, xy, start, end):
        xy = self.relative_bounding_box_to_absolute(xy)
        super().chord(xy, start, end, fill=self.fill, outline=None, width=1)

    def ellipse(self, xy):
        xy = self.relative_bounding_box_to_absolute(xy)
        super().ellipse(xy, fill=self.fill, outline=None, width=1)

    def line(self, xy):
        xy = self.relative_bounding_box_to_absolute(xy)
        super().line(xy, fill=self.fill, width=0, joint=None)

    def pieslice(self, xy, start, end):
        super().pieslice(xy, start, end, fill=self.fill, outline=None, width=1)

    def point(self, xy):
        xy = self.relative_bounding_box_to_absolute(xy)
        super().point(xy, fill=self.fill)

    def regular_polygon(self, bounding_circle, n_sides, rotation=0):
        bounding_circle = self.relative_bounding_circle_to_absolute(bounding_circle)
        super().regular_polygon(bounding_circle, n_sides, rotation=rotation, fill=self.fill, outline=None)

    def rectangle(self, xy):
        xy = self.relative_bounding_box_to_absolute(xy)
        super().rectangle(xy, fill=self.fill, outline=None, width=1)

    def rounded_rectangle(self, xy, radius=0):
        xy = self.relative_bounding_box_to_absolute(xy)
        super().rounded_rectangle(xy, radius=0, fill=self.fill, outline=None, width=1)
