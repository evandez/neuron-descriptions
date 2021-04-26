"""Models for annotating new, unseen neurons."""
import dataclasses
from typing import (Any, Mapping, Optional, Sequence, Tuple, Type, TypeVar,
                    overload)

from lv.models import featurizers
from lv.utils import lang, training
from lv.utils.typing import Device

import numpy
import torch
import tqdm
from sklearn import metrics
from torch import nn, optim
from torch.utils import data


class WordClassifierHead(nn.Module):
    """Classifier that predicts word distribution from visual features."""

    def __init__(self, feature_size: int, vocab_size: int):
        """Initialize the model.

        Args:
            feature_size (int): Visual feature size..
            vocab_size (int): Vocab size.

        """
        super().__init__()

        self.feature_size = feature_size
        self.vocab_size = vocab_size

        self.classifier = nn.Linear(feature_size, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the words that describe the images and masks.

        Args:
            features (torch.Tensor): Unit top images. Should have shape
                (batch_size, n_top_images, feature_size).

        Returns:
            torch.Tensor: Shape (batch_size, vocab_size) tensor containing
                probability each word describes the top images.

        """
        logits = self.classifier(features)
        predictions = torch.sigmoid(logits).mean(dim=1)
        return predictions


@dataclasses.dataclass(frozen=True)
class WordAnnotations:
    """Word annotations predicted by our model."""

    # Probabilities for *every word*, even those not predicted.
    # Will have shape like (batch_size, vocab_size).
    probabilities: torch.Tensor

    # Predicted words and corresponding indices. Each has length
    # of batch_size. Length of internal lists could be anything.
    words: Sequence[Sequence[str]]
    indices: Sequence[Sequence[int]]

    def __post_init__(self) -> None:
        """Validate all the data has the right shape."""
        lengths = {len(self.probabilities), len(self.words), len(self.indices)}
        if len(lengths) != 1:
            raise ValueError(f'found multiple batch sizes: {lengths}')


WordAnnotatorT = TypeVar('WordAnnotatorT', bound='WordAnnotator')


class WordAnnotator(nn.Module):
    """Predicts words that would appear in the caption of masked images."""

    def __init__(self, indexer: lang.Indexer,
                 featurizer: featurizers.Featurizer,
                 classifier: WordClassifierHead):
        """Initialize the model.

        Args:
            featurizer (featurizers.Featurizer): Model mapping images and masks
                to visual features.
            indexer (lang.Indexer): Indexer mapping words to indices.
            classifier (WordClassifierHead): The classification head.

        """
        super().__init__()

        self.indexer = indexer
        self.featurizer = featurizer
        self.classifier = classifier

    @overload
    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor,
                threshold: float = ...) -> WordAnnotations:
        """Predict the words that describe the images and masks.

        Args:
            images (torch.Tensor): Unit top images. Should have shape
                (batch_size, n_top_images, 3, height, width).
            masks (torch.Tensor): Unit top image masks. Should have shape
                (batch_size, n_top_images, 1, height, width).
            threshold (float, optional): Cutoff for whether or not a word
                is predicted or not.

        Returns:
            WordAnnotations: Predicted annotations for images/masks.

        """
        ...

    @overload
    def forward(self,
                features: torch.Tensor,
                threshold: float = ...) -> WordAnnotations:
        """Predict the words that describe the given image features.

        Args:
            features (torch.Tensor): Image features. Should have shape
                (batch_size, *featurizer.feature_shape).
            threshold (float, optional): Cutoff for whether or not a word
                is predicted or not.

        Returns:
            WordAnnotations: Predicted annotations for features.

        """
        ...

    def forward(self, images, masks=None, threshold=.5):
        """Implementat overloaded functions above."""
        batch_size, n_top_images, *_ = images.shape
        if masks is not None:
            images = images.view(-1, 3, *images.shape[-2:])
            masks = masks.view(-1, 1, *masks.shape[-2:])
            features = self.featurizer(images, masks)
        else:
            features = images
        features = features.view(batch_size, n_top_images, -1)

        # Compute probability of each word.
        probabilities = self.classifier(features)

        # Dereference words with probabilities over the threshold, sorted
        # by their probability.
        words, indices = [], []
        for s_ps in probabilities:
            s_indices = s_ps.gt(threshold).nonzero().squeeze().tolist()

            # Sometimes this chain of calls returns a single element. Rewrap it
            # in a list for consistency.
            if isinstance(s_indices, int):
                s_indices = [s_indices]

            s_indices = sorted(s_indices, key=lambda index: s_ps[index].item())
            s_words = self.indexer.undo(s_indices)
            words.append(s_words)
            indices.append(s_indices)

        return WordAnnotations(probabilities, tuple(words), tuple(indices))

    def score(self,
              dataset: data.Dataset,
              annotation_index: int = 4,
              batch_size: int = 16,
              threshold: float = .5,
              **kwargs: Any) -> Tuple[float, WordAnnotations]:
        """Compute F1 score of this model on the given dataset.

        Args:
            dataset (data.Dataset): Test dataset.
            annotation_index (int, optional): Index of annotations in dataset
                samples. Defaults to 4 to be compatible with
                AnnotatedTopImagesDataset.
            batch_size (int, optional): Number of samples to process at once.
                Defaults to 16.
            threshold (float, optional): Probability threshold for whether or
                not the model predicts a word. Defaults to .5.

        Returns:
            Tuple[float, WordAnnotations]: F1 score and predictions for every
                sample.

        """
        predictions = self.predict(dataset,
                                   batch_size=batch_size,
                                   threshold=threshold,
                                   **kwargs)
        y_pred = predictions.probabilities.gt(threshold).int().cpu().numpy()

        annotations = []
        for index in range(len(y_pred)):
            annotation = dataset[index][annotation_index]
            annotation = lang.join(annotation)
            annotations += annotation

        y_true = numpy.zeros((len(y_pred), len(self.indexer.vocab)))
        for index, annotation in enumerate(annotations):
            indices = self.indexer(annotation)
            y_true[index, sorted(set(indices))] = 1

        f1 = metrics.f1_score(y_pred=y_pred,
                              y_true=y_true,
                              average='weighted',
                              zero_division=0.)
        return f1, predictions

    def predict(self,
                dataset: data.Dataset,
                image_index: int = 2,
                mask_index: int = 3,
                batch_size: int = 16,
                features: Optional[data.TensorDataset] = None,
                device: Optional[Device] = None,
                display_progress: bool = True,
                **kwargs: Any) -> WordAnnotations:
        """Feed entire dataset through the annotation model.

        Keyword arguments are passed to forward.

        Args:
            dataset (data.Dataset): The dataset of images/masks.
            image_index (int, optional): Index of images in dataset samples.
                Defaults to 2 to be compatible with AnnotatedTopImagesDataset.
            mask_index (int, optional): Index of masks in dataset samples.
                Defaults to 3 to be compatible with AnnotatedTopImagesDataset.
            batch_size (int, optional): Number of samples to process on at
                once. Defaults to 16.
            features (Optional[data.TensorDataset], optional): Precomputed
                image features. Defaults to None.
            device (Optional[Device], optional): Send model and data to this
                device. Defaults to None.
            display_progress (bool, optional): Show progress for pre-
                featurizing dataset and for predicting features.
                Defaults to True.

        Returns:
            WordAnnotations: Predicted annotations for every sample in dataset.

        """
        if device is not None:
            self.to(device)
        if features is None:
            features = self.featurizer.map(dataset,
                                           image_index=image_index,
                                           mask_index=mask_index,
                                           batch_size=batch_size,
                                           device=device,
                                           display_progress=display_progress)

        loader = data.DataLoader(features, batch_size=batch_size)

        predictions = []
        for (inputs,) in tqdm.tqdm(loader) if display_progress else loader:
            outputs = self(inputs, **kwargs)
            predictions.append(outputs)

        probabilities = torch.cat([pred.probabilities for pred in predictions])

        words, indices = [], []
        for prediction in predictions:
            words.extend(prediction.words)
            indices.extend(prediction.indices)

        return WordAnnotations(probabilities, tuple(words), tuple(indices))

    @classmethod
    def fit(cls: Type[WordAnnotatorT],
            dataset: data.Dataset,
            featurizer: featurizers.Featurizer,
            image_index: int = 2,
            mask_index: int = 3,
            annotation_index: int = 4,
            batch_size: int = 16,
            max_epochs: int = 100,
            patience: int = 4,
            optimizer_t: Type[optim.Optimizer] = optim.Adam,
            optimizer_kwargs: Optional[Mapping[str, Any]] = None,
            indexer_kwargs: Optional[Mapping[str, Any]] = None,
            features: Optional[data.TensorDataset] = None,
            device: Optional[Device] = None,
            display_progress: bool = True) -> WordAnnotatorT:
        """Train a new WordAnnotator from scratch.

        Args:
            dataset (data.Dataset): Training dataset.
            featurizer (featurizers.Featurizer): Image featurizer.
            image_index (int, optional): Index of images in dataset samples.
                Defaults to 2 to be compatible with AnnotatedTopImagesDataset.
            mask_index (int, optional): Index of masks in dataset samples.
                Defaults to 3 to be compatible with AnnotatedTopImagesDataset.
            annotation_index (int, optional): Index of language annotations in
                dataset samples. Defaults to 4 to be compatible with
                AnnotatedTopImagesDataset.
            batch_size (int, optional): Number of samples to train on at once.
                Defaults to 16.
            max_epochs (int, optional): Maximum number of epochs to train for.
                Defaults to 100.
            patience (int, optional): If loss does not improve for this many
                epochs, stop training. Defaults to 4.
            optimizer_t (Type[optim.Optimizer], optional): Optimizer to use.
                Defaults to Adam.
            optimizer_kwargs (Optional[Mapping[str, Any]], optional): Optimizer
                keyword arguments to pass at construction. Defaults to None.
            indexer_kwargs (Optional[Mapping[str, Any]], optional): Indexer
                keyword arguments to pass at construction. Defaults to None.
            features (Optional[data.TensorDataset], optional): Precomputed
                image features. By default, computed from the full dataset.
            device (Optional[Device], optional): Send model and all data to
                this device. Defaults to None.
            display_progress (bool, optional): Show progress bars for pre-
                featurizing dataset and for training model. Defaults to True.

        Returns:
            WordAnnotatorT: The trained WordAnnotator.

        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if indexer_kwargs is None:
            indexer_kwargs = {}
        if features is None:
            features = featurizer.map(dataset,
                                      image_index=image_index,
                                      mask_index=mask_index,
                                      batch_size=batch_size,
                                      device=device,
                                      display_progress=display_progress)

        annotations = []
        for index in range(len(features)):
            annotation = dataset[index][annotation_index]
            annotation = lang.join(annotation)
            annotations.append(annotation)

        indexer = lang.indexer(annotations, **indexer_kwargs)

        targets = torch.zeros(len(features), len(indexer.vocab), device=device)
        for index, annotation in enumerate(annotations):
            indices = indexer(annotation)
            targets[index, sorted(set(indices))] = 1

        features_loader = data.DataLoader(features, batch_size=batch_size)
        targets_loader = data.DataLoader(data.TensorDataset(targets),
                                         batch_size=batch_size)

        feature_size = numpy.prod(featurizer.feature_shape).item()
        vocab_size = len(indexer.vocab)
        classifier = WordClassifierHead(feature_size, vocab_size).to(device)

        optimizer = optimizer_t(classifier.parameters(), **optimizer_kwargs)

        stopper = training.EarlyStopping(patience=patience)

        # Balance the dataset using the power of BAYESIAN STATISTICS, BABY!
        n_positives = targets.sum(dim=0)
        n_negatives = len(targets) - n_positives
        pos_weight = n_negatives / n_positives
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        progress = range(max_epochs)
        if display_progress:
            progress = tqdm.tqdm(progress)

        for _ in progress:
            train_loss = 0.
            for (inputs,), (targets,) in zip(features_loader, targets_loader):
                inputs = inputs.view(*inputs.shape[:2], feature_size)
                predictions = classifier(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= len(features_loader)

            if display_progress:
                assert not isinstance(progress, range)
                progress.set_description(f'loss={train_loss:.3f}')

            if stopper(train_loss):
                break

        return cls(indexer, featurizer, classifier)
