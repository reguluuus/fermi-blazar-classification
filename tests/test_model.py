import torch

from agn_classifier.model import BlazarClassifier


def test_model_output_shape():
    model = BlazarClassifier(input_features=44, hidden_size=64, dropout=0.1)
    x = torch.randn(8, 44)
    assert model(x).shape == (8, 1)
