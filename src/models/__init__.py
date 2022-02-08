"""Models trained on latent vocabulary data."""
# flake8: noqa
from src.models.classifiers import ImageClassifier, classifier
from src.models.decoders import Decoder, decoder
from src.models.encoders import (Encoder, PyramidConvEncoder,
                                 SpatialConvEncoder, encoder)
from src.models.lms import LanguageModel, lm
