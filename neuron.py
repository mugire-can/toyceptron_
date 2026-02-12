class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError(
                "Input length must match weights length: "
                f"{len(inputs)} != {len(self.weights)}"
            )
        weighted_sum = 0.0
        for weight, input_val in zip(self.weights, inputs):
            weighted_sum += weight * input_val
        return weighted_sum + self.bias