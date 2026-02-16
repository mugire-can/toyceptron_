def act_identity(x):
    return x


def act_threshold(x):
    return 1 if x > 0 else 0


def act_relu(x):
    return max(0.0, x)
