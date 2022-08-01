import sys


def quit_program(message: str):
    print(message, file=sys.stderr)
    sys.exit()


class ParameterInfo:
    def __init__(self, param_name: str, param_type: str, min_value=None, max_value=None):
        self.param_name = param_name
        self.param_type = param_type
        self.min_value = min_value
        self.max_value = max_value


class ParameterVectorInfo:
    def __init__(self):
        self.param_info = []

    def __len__(self):
        return len(self.param_info)

    def __getitem__(self, item):
        return self.param_info[item]

    def add_element(self, param_name: str, param_type: str, min_value=None, max_value=None):
        if param_type == 's' or param_type == 'i':
            if min_value is None or max_value is None:
                quit_program('Scalar and integer parameters require min and max values')
        elif param_type == 'b':
            min_value = 0
            max_value = 1
        else:
            quit_program('Invalid parameter type')
        self.param_info.append(ParameterInfo(param_name, param_type, min_value, max_value))


class ParameterVector:
    def __init__(self, param_vector_info: ParameterVectorInfo, param_vector: list):
        self.param_vector_info = param_vector_info
        self.param_vector = param_vector

    def is_valid(self):
        rel_tol = 1e-6
        if self.param_vector is None:
            return False
        if len(self.param_vector_info) != len(self.param_vector):
            print('Warning: Parameter vector length is incorrect')
            return False
        for i, p in enumerate(self.param_vector):
            min_value = self.param_vector_info[i].min_value
            max_value = self.param_vector_info[i].max_value
            if p < min_value and abs(p - min_value) > rel_tol:
                print('Warning: Parameter value out of range')
                return False
            elif p > max_value and abs(p - max_value) > rel_tol:
                print('Warning: Parameter value out of range')
                return False
        return True

    def get_parameter(self, param_name: str):
        for i in range(len(self.param_vector_info)):
            if self.param_vector_info[i].param_name == param_name:
                if self.param_vector_info[i].param_type == 's':
                    return self.param_vector[i]
                else:
                    return int(self.param_vector[i])
        return None