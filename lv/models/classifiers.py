"""Wrappers around image classifiers."""
import collections
from typing import Any, Mapping, Optional, Sequence, Sized, Type, Union, cast

from lv.utils import ablations, training
from lv.utils.typing import Device, Layer, Unit

import torch
from torch import nn, optim
from torch.utils import data
from tqdm.auto import tqdm


class ImageClassifier(nn.Sequential):
    """Wraps an image classifier and adds some nice utilities."""

    def __init__(self, classifier: nn.Sequential):
        """Initialize the classifier.

        Args:
            classifier (nn.Sequential): The classification model.

        """
        super().__init__(collections.OrderedDict(classifier.named_children()))

    def fit(self,
            dataset: data.Dataset,
            image_index: int = 0,
            target_index: int = 1,
            batch_size: int = 128,
            max_epochs: int = 100,
            patience: int = 4,
            hold_out: Union[float, data.Dataset] = .1,
            optimizer_t: Type[optim.Optimizer] = optim.Adam,
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
            hold_out (Union[float, data.Dataset], optional): If a float, hold
                out this fraction of the training data as validation set.
                If a `torch.utils.data.Dataset`, use this as the validation
                set. Defaults to .1.
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

        if isinstance(hold_out, data.Dataset):
            train = dataset
            val = hold_out
        else:
            train, val = training.random_split(dataset, hold_out=hold_out)

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
            parameters = []
            for layer in layers:
                parameters += list(self[layer].parameters())

        optimizer = optimizer_t(parameters, **optimizer_kwargs)
        criterion = nn.CrossEntropyLoss()
        stopper = training.EarlyStopping(patience=patience)

        progress = range(max_epochs)
        if display_progress_as is not None:
            progress = tqdm(progress, desc=display_progress_as)

        with ablations.ablated(self, ablate or []) as model:
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
                    break

    def accuracy(
        self,
        dataset: data.Dataset,
        image_index: int = 0,
        target_index: int = 1,
        batch_size: int = 128,
        num_workers: int = 0,
        ablate: Optional[Sequence[Unit]] = None,
        device: Optional[Device] = None,
        display_progress_as: Optional[str] = 'test classifer',
    ) -> float:
        """Compute accuracy of this model on the given dataset.

        Args:
            dataset (data.Dataset): The dataset.
            image_index (int, optional): [description]. Defaults to 0.
            target_index (int, optional): [description]. Defaults to 1.
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
            float: Accuracy on the dataset.

        """
        if device is not None:
            self.to(device)

        with ablations.ablated(self, ablate or []) as model:
            loader = data.DataLoader(dataset,
                                     num_workers=num_workers,
                                     batch_size=batch_size)
            if display_progress_as is not None:
                loader = tqdm(loader, desc=display_progress_as)

            correct = 0
            for batch in loader:
                images = batch[image_index].to(device)
                targets = batch[target_index].to(device)
                with torch.no_grad():
                    predictions = model(images)
                correct += predictions.argmax(dim=-1).eq(targets).sum().item()

        return correct / len(cast(Sized, dataset))
