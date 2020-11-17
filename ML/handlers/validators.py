from ML.handlers.errors import *


class Validator:
    pass


class LayerValidator(Validator):

    @staticmethod
    def validate_number_of_neurons(number_of_neurons):
        if number_of_neurons < 0:
            raise LayerError("Number of neurons in a layer must be positive!")


class NNValidator(Validator):

    @staticmethod
    def validate_input_data(input_data, input_layer):
        if len(input_data) != len(input_layer):
            raise NNError("Input data has different len than input layer!")
