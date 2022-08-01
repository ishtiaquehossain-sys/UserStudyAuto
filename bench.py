import parameter
from shape import Shape


class Bench(Shape):
    def __init__(self):
        super(Bench, self).__init__()
        self.predictor.init_model('bench', self.param_vector_info)
        self.specify_parameter_vectors()
        self.images = self.make_shapes(self.param_vectors)

    def setup_parameter_vector(self):
        self.add_parameter_vector_element('height', 's', 0.293, 0.43)       # Element at index 0
        self.add_parameter_vector_element('leg_height', 's', 0.14, 0.24)    # Element at index 1
        self.add_parameter_vector_element('leg_width', 's', 0.01, 0.05)     # Element at index 2
        self.add_parameter_vector_element('seat_height', 's', 0.01, 0.05)   # Element at index 3
        self.add_parameter_vector_element('num_hbars', 'i', 2, 14)          # Element at index 4
        self.add_parameter_vector_element('bottom_bar', 'b')                # Element at index 5
        return

    def procedure(self, param_vector: parameter.ParameterVector):
        # As example, rules for the bench class are implemented here

        # Retrieve the parameter values
        height = param_vector.get_parameter('height')
        leg_height = param_vector.get_parameter('leg_height')
        leg_width = param_vector.get_parameter('leg_width')
        seat_height = param_vector.get_parameter('seat_height')
        num_hbars = param_vector.get_parameter('num_hbars')
        bottom_bar = param_vector.get_parameter('bottom_bar')

        # draw the legs
        xy = (0.045, 0.5 - height / 2, 0.045 + leg_width, 0.5 + height / 2)
        self.rectangle(xy)
        xy = (0.955 - leg_width, 0.5 - height / 2, 0.955, 0.5 + height / 2)
        self.rectangle(xy)

        # draw the seat
        seat_location = 0.5 - height / 2 + (height - leg_height)
        xy = (0.045, seat_location - seat_height / 2, 0.955, seat_location + seat_height / 2)
        self.rectangle(xy)

        # draw horizontal bars
        start = 0.5 - height / 2
        offset = (seat_location - seat_height / 2 - 0.5 + height / 2) / num_hbars
        for i in range(num_hbars):
            xy = (0.045, start, 0.955, start + offset / 2)
            self.rectangle(xy)
            start = start + offset

        # draw the bottom bar
        if bottom_bar != 0:
            bbar_location = 0.5 - height / 2 + (height - leg_height) + leg_height / 2
            xy = (0.045, bbar_location - 0.006, 0.955, bbar_location + 0.006)
            self.rectangle(xy)
        return
