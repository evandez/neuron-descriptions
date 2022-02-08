"""Unit tests for lv/zoo/configs module."""
from src.zoo import configs


def test_models():
    """Test models runs and returns a dict."""
    actual = configs.models()
    assert isinstance(actual, dict)


def test_datasets():
    """Test datasets runs and returns a dict."""
    actual = configs.datasets()
    assert isinstance(actual, dict)
