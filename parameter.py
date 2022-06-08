import sys


def quit_program(message: str):
    print(message, file=sys.stderr)
    sys.exit()


class Parameter:
    def __init__(self, param_name: str, param_type: str, min_value=None, max_value=None, num_levels: int = None):
        if param_type == 's':
            if min_value is None or max_value is None or num_levels is None:
                quit_program('Scalar parameter requires additional specification')
        else:
            if param_type == 'b':
                min_value = 0
                max_value = 1
            elif param_type == 'i':
                if min_value is None or max_value is None:
                    quit_program('Scalar and integer parameters require additional specification')
            else:
                quit_program('Invalid parameter type')
            num_levels = max_value - min_value + 1
        self.info = {
            'param_name': param_name,
            'param_type': param_type,
            'min_value': min_value,
            'max_value': max_value,
            'num_levels': num_levels
        }


class Shape:
    def __init__(self, shape_name: str):
        self.params = []
        if shape_name == 'bench':
            self.params.append(Parameter('height', 's', 150, 220, 3))
            self.params.append(Parameter('leg_height_percent', 's', 0.4, 0.6, 3))
            self.params.append(Parameter('leg_width', 's', 5, 30, 3))
            self.params.append(Parameter('seat_height', 's', 5, 30, 3))
            self.params.append(Parameter('num_hbars', 'i', 2, 14))
            self.params.append(Parameter('bottom_bar', 'b'))
        elif shape_name == 'chair':
            self.params.append(Parameter('dummy_scalar', 's', 60, 80, 3))
            self.params.append(Parameter('dummy_binary', 'b'))
            self.params.append(Parameter('dummy_integer', 'i', 0, 5))
        else:
            quit_program('Invalid shape name')
