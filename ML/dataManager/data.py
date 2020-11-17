class TrainingData:
    """
    A class that stores data

    ...

    Attributes
    __________

    input_data : a list of values
        - represents the input data
    output_data : a list of values
        - represents the expected output data
    """
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def get_input(self, x=None):
        return self.input_data if x is None else self.input_data[x]

    def get_output(self):
        return self.output_data
