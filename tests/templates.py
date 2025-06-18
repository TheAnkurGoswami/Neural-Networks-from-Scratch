from typing import Literal


def get_weight_template(framework: Literal["pt", "tf"] = "pt") -> str:
    value = "Tensorflow" if framework == "tf" else "PyTorch"
    return f"Weights are not close between custom implementation and {value}"


def get_bias_template(framework: Literal["pt", "tf"] = "pt") -> str:
    value = "Tensorflow" if framework == "tf" else "PyTorch"
    return f"Biases are not close between custom implementation and {value}"


def get_loss_template(framework: Literal["pt", "tf"] = "pt") -> str:
    value = "Tensorflow" if framework == "tf" else "PyTorch"
    return f"Loss are not close between custom implementation and {value}"


def get_output_template(framework: Literal["pt", "tf"] = "pt") -> str:
    value = "Tensorflow" if framework == "tf" else "PyTorch"
    return f"Outputs are not close between custom implementation and {value}"
