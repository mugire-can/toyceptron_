from layer import Layer


class Network:
    def __init__(self, input_size, activation):
        self.input_size = input_size
        self.activation = activation
        self.layers = []

    def add(self, weights, biases):
        if len(weights) != len(biases):
            raise ValueError(
                "weights and biases must have the same length: "
                f"{len(weights)} != {len(biases)}"
            )

        expected_input_size = (
            self.input_size if not self.layers else len(self.layers[-1].neurons)
        )
        for index, neuron_weights in enumerate(weights):
            if len(neuron_weights) != expected_input_size:
                raise ValueError(
                    "Layer weights size must match previous layer size: "
                    f"neuron {index} has {len(neuron_weights)}, "
                    f"expected {expected_input_size}"
                )

        layer = Layer(weights_list=weights, biases_list=biases)
        self.layers.append(layer)

    def feedforward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            raw_outputs = layer.forward(current_input)
            activated_outputs = [self.activation(out) for out in raw_outputs]
            current_input = activated_outputs
        return current_input
