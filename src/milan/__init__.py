"""Models trained on latent vocabulary data."""
# flake8: noqa
from src.milan.classifiers import ImageClassifier, classifier
from src.milan.decoders import Decoder, decoder
from src.milan.encoders import (Encoder, PyramidConvEncoder,
                                SpatialConvEncoder, encoder)
from src.milan.lms import LanguageModel, lm
