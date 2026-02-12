class Neuron:
        
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        weighted_sum = 0.0
        for weight, input_val in zip(self.weights, inputs):
            weighted_sum += weight * input_val
        output = weighted_sum + self.bias
        
        return output