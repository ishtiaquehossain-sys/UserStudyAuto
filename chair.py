import parameter
from shape import Shape


class Chair(Shape):
    def __init__(self):
        super(Chair, self).__init__()
        self.predictor.init_model('chair', self.param_vector_info)
        self.specify_parameter_vectors()
        self.images = self.make_shapes(self.param_vectors)

    def setup_parameter_vector(self):
        """
        Define the parameter vector in this function
        Use the following format when adding element to the parameter vector
        self.add_parameter_vector_element
            (name_of_parameter,             # String, specifying name of the parameter
            type_of_parameter,              # 's', 'i' or 'b', for scalar, integer or binary respectively
            minimum_allowable_value,        # Ignore if binary parameter
            maximum_allowable_value)        # Ignore if binary parameter
        """
        return

    def procedure(self, param_vector: parameter.ParameterVector):
        # Implement your procedure for the chair class in this function
        return
