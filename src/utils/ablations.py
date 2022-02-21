"""Utilities for altering unit activations real-time."""
import collections
import contextlib
from typing import (Any, Callable, Dict, Iterator, Mapping, Optional, Sequence,
                    Sized, Type, Union, cast)

from src.deps.netdissect import nethook
from src.utils import training
from src.utils.typing import Device, Layer, Unit

import torch
from torch import nn, optim
from torch.utils import data
from tqdm.auto import tqdm

Rule = Callable[[torch.Tensor], torch.Tensor]
RuleFactory = Callable[[Sequence[int]], Rule]


def zero(units: Sequence[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Zero the given units.

    Args:
        units (Sequence[int]): The units to zero.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Function that takes layer
            features and zeros the given units, returning the result.

    """

    def fn(features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4:
            raise ValueError(f'expected 4D features, got {features.dim()}')
        # Make sure we don't break autograd by editing values in place.
        # Just use a mask. Fauci said it first.
        shape = (*features.shape[:2], 1, 1)
        mask = features.new_ones(*shape)
        mask[:, units] = 0
        return features * mask

    return fn


@contextlib.contextmanager
def ablated(
    model: nn.Module,
    units: Sequence[Unit],
    rule: RuleFactory = zero,
) -> Iterator[nethook.InstrumentedModel]:
    """Ablate the given units according to the given rule.

    Args:
        model (nn.Module): The model to ablate.
        units (Sequence[Unit]): The (layer, unit) pairs to ablate.
        rule (RuleFactory, optional): The rule to ablate to. Defaults to
            zeroing the units.

    Yields:
        Iterator[nethook.InstrumentedModel]: An InstrumentedModel configured to
            ablate the units.

    """
    with nethook.InstrumentedModel(model) as instrumented:
        edits = collections.defaultdict(list)
        for la, un in units:
            edits[la].append(un)
        for la, uns in edits.items():
            instrumented.edit_layer(la, rule=rule(sorted(uns)))
        yield instrumented


class ImageClassifier(nn.Module):
    """Wraps an image classifier and adds some ablation utilities."""

    def __init__(self, model: nn.Module):
        """Initialize the classifier.

        Args:
            model (nn.Module): The classification model.

        """
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call `self.model.forward`."""
        return self.model(*args, **kwargs)

    def fit(self,
            dataset: data.Dataset,
            image_index: int = 0,
            target_index: int = 1,
            batch_size: int = 128,
            max_epochs: int = 100,
            patience: int = 4,
            hold_out: Union[float, Sequence[int]] = .1,
            optimizer_t: Type[optim.Optimizer] = optim.AdamW,
            optimizer_kwargs: Optional[Mapping[str, Any]] = None,
            num_workers: int = 0,
            ablate: Optional[Sequence[Unit]] = None,
            layers: Optional[Sequence[Layer]] = None,
            device: Optional[Device] = None,
            display_progress_as: Optional[str] = 'train classifer') -> None:
        """Train the classifier on the given dataset.

        Args:
            dataset (data.Dataset): The training dataset.
            image_index (int, optional): Index of images in dataset samples.
                Defaults to 0 to be compatible with `ImageFolder`.
            target_index (int, optional): Index of target labels in dataset
                samples. Defaults to 1 to be compatible with `ImageFolder`.
            batch_size (int, optional): Number of samples to process at once.
                Defaults to 128.
            max_epochs (int, optional): Max number of epochs to train for.
                Defaults to 100.
            patience (int, optional): Stop training if validation loss does not
                improve for this many epochs. Defaults to 4.
            hold_out (Union[float, Sequence[int]], optional): If a float, hold
                out this fraction of the training data as validation set.
                If an integer sequence, use samples at these indices to
                construct the validation set. Defaults to .1.
            optimizer_t (Type[optim.Optimizer], optional): Optimizer to use.
                Defaults to `torch.optim.Adam`.
            optimizer_kwargs (Optional[Mapping[str, Any]], optional): Optimizer
                options. Defaults to None.
            num_workers (int, optional): Number of worker threads to use in the
                `torch.utils.data.DataLoader`. Defaults to 0.
            ablate (Optional[Sequence[Unit]], optional): Ablate these neurons
                when training. Defaults to None.
            layers (Optional[Sequence[Layer]], optional) Layers to optimize.
            device (Optional[Device], optional): Send this model and all
                tensors to this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this label while training. Defaults to 'train classifer'.

        """
        if device is not None:
            self.to(device)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if isinstance(hold_out, float):
            train, val = training.random_split(dataset, hold_out=hold_out)
        else:
            train, val = training.fixed_split(dataset, hold_out)

        train_loader = data.DataLoader(train,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=True)
        val_loader = data.DataLoader(val,
                                     batch_size=batch_size,
                                     num_workers=num_workers)

        if layers is None:
            parameters = list(self.parameters())
        else:
            missing = {str(layer) for layer in layers}

            parameters = []
            for name, submodule in self.model.named_modules():
                if name in missing:
                    parameters += list(submodule.parameters())
                    missing -= {name}

            if missing:
                raise KeyError(f'could not find layers: {sorted(missing)}')

        optimizer = optimizer_t(parameters, **optimizer_kwargs)
        criterion = nn.CrossEntropyLoss()
        stopper = training.EarlyStopping(patience=patience)

        progress = range(max_epochs)
        if display_progress_as is not None:
            progress = tqdm(progress, desc=display_progress_as)

        with ablated(self.model, ablate or []) as model:
            best = self.state_dict()
            for _ in progress:
                model.train()
                train_loss = 0.
                for batch in train_loader:
                    images = batch[image_index].to(device)
                    targets = batch[target_index].to(device)
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0.
                for batch in val_loader:
                    images = batch[image_index].to(device)
                    targets = batch[target_index].to(device)
                    with torch.no_grad():
                        predictions = model(images)
                        loss = criterion(predictions, targets)
                    val_loss += loss.item()
                val_loss /= len(val_loader)

                if display_progress_as is not None:
                    assert not isinstance(progress, range)
                    progress.set_description(f'{display_progress_as} '
                                             f'[train_loss={train_loss:.3f}, '
                                             f'val_loss={val_loss:.3f}]')

                if stopper(val_loss):
                    self.load_state_dict(best)
                    break

                if stopper.improved:
                    best = self.state_dict()

    def predict(
        self,
        dataset: data.Dataset,
        image_index: int = 0,
        batch_size: int = 128,
        num_workers: int = 0,
        ablate: Optional[Sequence[Unit]] = None,
        device: Optional[Device] = None,
        display_progress_as: Optional[str] = 'classify images',
    ) -> torch.Tensor:
        """Run the model on every element in the dataset.

        Args:
            dataset (data.Dataset): The dataset.
            image_index (int, optional): Index of images in dataset.
                Defaults to 0 to be compatible with
                `torchvision.datasets.ImageFolder`.
            batch_size (int, optional): Number of samples to process at once.
                Defaults to 128.
            num_workers (int, optional): Number of workers for DataLoader
                to use. Defaults to 0.
            ablate (Optional[Sequence[Unit]], optional): Ablate these units
                before testing. Defaults to None.
            device (Optional[Device], optional): Send this model and all
                tensors to this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this message while testing. Defaults to 'test classifier'.

        Returns:
            torch.Tensor: Long tensor containing class predictions for every
                item in the dataset, with shape (len(dataset),).

        """
        if device is not None:
            self.to(device)

        # Prepare data loader.
        loader = data.DataLoader(dataset,
                                 num_workers=num_workers,
                                 batch_size=batch_size)
        if display_progress_as is not None:
            loader = tqdm(loader, desc=display_progress_as)

        # Compute predictions.
        predictions = []
        with ablated(self.model, ablate or []) as model:
            for batch in loader:
                images = batch[image_index].to(device)
                with torch.no_grad():
                    predictions.append(model(images).argmax(dim=-1))

        return torch.cat(predictions)

    def accuracy(
        self,
        dataset: data.Dataset,
        predictions: Optional[torch.Tensor] = None,
        target_index: int = 1,
        device: Optional[Device] = None,
        display_progress_as: Optional[str] = 'test classifer',
        **kwargs: Any,
    ) -> float:
        """Compute accuracy of this model on the given dataset.

        The **kwargs are forwarded to `ImageClassifier.predict`.

        Args:
            dataset (data.Dataset): The dataset.
            predictions (torch.Tensor): Precomputed predictions.
                By default, computed from dataset.
            target_index (int, optional): Index of target labels in dataset.
                Defaults to 1 to be compatible with
                `torchvision.datasets.ImageFolder`.
            device (Optional[Device], optional): Send this model and all
                tensors to this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this message while testing. Defaults to 'test classifier'.

        Returns:
            float: Accuracy on the dataset.

        """
        if predictions is None:
            predictions = self.predict(dataset,
                                       device=device,
                                       display_progress_as=display_progress_as,
                                       **kwargs)
        size = len(cast(Sized, dataset))
        targets = torch.tensor(
            [dataset[index][target_index] for index in range(size)],
            dtype=torch.long,
            device=device,
        )
        correct = predictions.eq(targets).sum().item()
        return correct / size

    def accuracies(
        self,
        dataset: data.Dataset,
        predictions: Optional[torch.Tensor] = None,
        target_index: int = 1,
        device: Optional[Device] = None,
        display_progress_as: Optional[str] = 'test classifer (class-wise)',
        **kwargs: Any,
    ) -> Mapping[int, float]:
        """Compute class-by-class accuracy of this model on the given dataset.

        The **kwargs are forwarded to `ImageClassifier.predict`.

        Args:
            dataset (data.Dataset): The dataset.
            predictions (torch.Tensor): Precomputed predictions.
                By default, computed from dataset.
            target_index (int, optional): Index of target labels in dataset.
                Defaults to 1 to be compatible with
                `torchvision.datasets.ImageFolder`.
            device (Optional[Device], optional): Send this model and all
                tensors to this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this message while testing. Defaults to
                'test classifier (class-wise)'.

        Returns:
            Mapping[int, float]: Class-by-class accuracy on this dataset.

        """
        if predictions is None:
            predictions = self.predict(dataset,
                                       device=device,
                                       display_progress_as=display_progress_as,
                                       **kwargs)

        size = len(cast(Sized, dataset))
        targets = torch.tensor(
            [dataset[index][target_index] for index in range(size)],
            dtype=torch.long,
            device=device,
        )

        correct: Dict[int, int] = collections.defaultdict(int)
        total: Dict[int, int] = collections.defaultdict(int)
        for prediction, target in zip(predictions.tolist(), targets.tolist()):
            correct[target] += prediction == target
            total[target] += 1
        assert correct.keys() == total.keys()

        return {
            target: correct[target] / total[target]
            for target in correct.keys()
        }
