"""Generic RNN language models."""
from typing import Any, Mapping, Optional, Sized, Type, cast

from lv.utils import lang, training
from lv.utils.typing import Device, StrSequence

import torch
from torch import nn, optim
from torch.utils import data
from tqdm.auto import tqdm


class LanguageModel(nn.Module):
    """A simple LSTM language model."""

    def __init__(self,
                 indexer: lang.Indexer,
                 embedding_size: int = 128,
                 hidden_size: int = 512,
                 layers: int = 2,
                 dropout: float = .5):
        """Initialize the LM.

        Args:
            indexer (lang.Indexer): Sequence indexer.
            embedding_size (int, optional): Size of input word embeddings.
                Defaults to 128.
            hidden_size (int, optional): Size of hidden state. Defaults to 512.
            layers (int, optional): Number of layers to use in the LSTM.
                Defaults to 2.
            dropout (float, optional): Dropout rate to use between recurrent
                connections. Defaults to .5.

        """
        super().__init__()

        self.indexer = indexer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout

        self.embedding = nn.Embedding(len(indexer),
                                      embedding_size,
                                      padding_idx=indexer.pad_index)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=layers,
                            dropout=dropout,
                            batch_first=True)
        self.output = nn.Sequential(nn.Linear(hidden_size, len(indexer)),
                                    nn.LogSoftmax())

    def forward(self,
                inputs: torch.Tensor,
                reduce: bool = False) -> torch.Tensor:
        """Compute the log probability of the given sequence.

        Args:
            inputs (torch.Tensor): The sequence. Should have shape
                (batch_size, length) and have type `torch.long`.
            reduce (bool, optional): Instead of returning probability for
                each token, return log probability of whole sequence.
                Defaults to False.

        Returns:
            torch.Tensor: Shape (batch_size, length, vocab_size) tensor of
                log probabilites for each token if `reduce=False`, other shape
                (batch_size,) tensor of log probabilities for whole sequences.

        """
        embeddings = self.embedding(inputs)
        hiddens, _ = self.lstm(embeddings)
        lps = self.output(hiddens)
        if reduce:
            batch_size, length, _ = lps.shape
            batch_idx = torch.arange(batch_size).repeat_interleave(length)
            seq_idx = torch.arange(batch_size).repeat(length)
            word_idx = inputs.flatten()
            lps = lps[batch_idx, seq_idx, word_idx]
            lps = lps.view(batch_size, length)
            lps = lps.sum(dim=-1)
        return lps

    def predict(
        self,
        sequences: StrSequence,
        batch_size: int = 64,
        device: Optional[Device] = None,
        display_progress_as: Optional[str] = 'compute lm probs',
    ) -> torch.Tensor:
        """Compute log probability of each sequence.

        Args:
            sequences (StrSequence): Text sequences.
            batch_size (int, optional): Number of sequences to process at once.
                Defaults to 64.
            device (Optional[Device], optional): Send this model and all
                tensors to this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show progress bar
                with this message if set. Defaults to 'compute lm probs'.

        Returns:
            torch.Tensor: Shape (len(sequences),) tensor containing logprob
                for each sequence.

        """
        if device is not None:
            self.to(device)
        indices = self.indexer(sequences,
                               start=True,
                               stop=True,
                               pad=True,
                               unk=True)
        dataset = data.TensorDataset(torch.tensor(indices, device=device))
        loader = data.DataLoader(dataset, batch_size=batch_size)
        if display_progress_as is not None:
            loader = tqdm(loader, desc=display_progress_as)

        outputs = []
        for (inputs,) in loader:
            with torch.no_grad():
                lps = self(inputs, reduce=True)
            outputs.append(lps)
        return torch.cat(outputs)

    def fit(self,
            dataset: data.Dataset,
            annotation_index: int = 4,
            batch_size: int = 128,
            max_epochs: int = 100,
            patience: int = 4,
            hold_out: float = .1,
            optimizer_t: Type[optim.Optimizer] = optim.Adam,
            optimizer_kwargs: Optional[Mapping[str, Any]] = None,
            device: Optional[Device] = None,
            display_progress_as: Optional[str] = 'train lm') -> None:
        """Train this LM on the given dataset.

        Args:
            dataset (data.Dataset): The dataset.
            annotation_index (int, optional): Index of the sequences to
                model in each dataset sample. Defaults to 4 to be compatible
                with `AnnotatedTopImagesDataset`.
            batch_size (int, optional): Number of samples to process at once.
                Defaults to 128.
            max_epochs (int, optional): Maximum number of epochs to train for.
                Defaults to 100.
            patience (int, optional): Stop training if validation loss does
                not improve for this many epochs. Defaults to 4.
            hold_out (float, optional): Hold out this fraction of the dataset
                as a validation set. Defaults to .1.
            optimizer_t (Type[optim.Optimizer], optional): Optimizer type.
                Defaults to optim.Adam.
            optimizer_kwargs (Optional[Mapping[str, Any]], optional): Optimizer
                options. Defaults to None.
            device (Optional[Device], optional): Send this model and all data
                to this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                prefixed with this message while training.
                Defaults to 'train lm'.

        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if device is not None:
            self.to(device)

        # An anonymous wrapper dataset that uses just the sequences.
        class SequenceDataset(data.Dataset):

            def __init__(self, dataset, annotation_index=4):
                self.sequences = []
                for index in range(len(cast(Sized, dataset))):
                    annotation = dataset[index][annotation_index]
                    if isinstance(annotation, str):
                        self.sequences.append(annotation)
                    else:
                        self.sequences += annotation

            def __getitem__(self, index):
                return self.sequences[index]

            def __len__(self):
                return len(self.sequences)

        # Prepare training data.
        dataset = SequenceDataset(dataset, annotation_index=annotation_index)
        train, val = training.random_split(dataset, hold_out=hold_out)
        train_loader = data.DataLoader(train,
                                       batch_size=batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(val, batch_size=batch_size)

        # Prepare optimizer, loss, training utils.
        optimizer = optimizer_t(self.parameters(), **optimizer_kwargs)
        criterion = nn.NLLLoss(ignore_index=self.indexer.pad_index)
        stopper = training.EarlyStopping(patience=patience)

        progress = range(max_epochs)
        if display_progress_as is not None:
            progress = tqdm(progress, desc=display_progress_as)

        # Begin training!
        for _ in progress:
            self.train()
            train_loss = 0.
            for sequences in train_loader:
                inputs = torch.tensor(self.indexer(sequences, pad=True),
                                      device=device)
                predictions = self(inputs)
                loss = criterion(predictions.permute(0, 2, 1), inputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.eval()
            val_loss = 0.
            for sequences in val_loader:
                inputs = torch.tensor(self.indexer(sequences, pad=True),
                                      device=device)
                with torch.no_grad():
                    predictions = self(inputs)
                    loss = criterion(predictions.permute(0, 2, 1), inputs)
                val_loss += loss.item()
            val_loss /= len(val_loader)

            if display_progress_as is not None:
                assert not isinstance(progress, range)
                progress.set_description(f'{display_progress_as} '
                                         f'[train_loss={train_loss:.3f}, '
                                         f'val_loss={val_loss:.3f}]')

            if stopper(val_loss):
                break


def lm(dataset: data.Dataset,
       annotation_index: int = 4,
       indexer_kwargs: Optional[Mapping[str, Any]] = None,
       **kwargs: Any) -> LanguageModel:
    """Initialize the langauge model.

    The **kwargs are forwarded to the constructor.

    Args:
        dataset (data.Dataset): Dataset on which LM will be trained.
        annotation_index (int, optional): Index on language annotations in
            the dataset. Defaults to 4 to be compatible with
            AnnotatedTopImagesDataset.
        indexer_kwargs (Optional[Mapping[str, Any]], optional): Indexer
            options. By default, indexer is configured to not ignore stop
            words and punctuation.

    Returns:
        LanguageModel: The instantiated model.

    """
    if indexer_kwargs is None:
        indexer_kwargs = {}

    annotations = []
    for index in range(len(cast(Sized, dataset))):
        annotation = dataset[index][annotation_index]
        annotation = lang.join(annotation)
        annotations.append(annotation)

    indexer_kwargs = dict(indexer_kwargs)
    if 'tokenize' not in indexer_kwargs:
        indexer_kwargs['tokenize'] = lang.tokenizer(lemmatize=False,
                                                    ignore_stop=False,
                                                    ignore_punct=False)
    for key in ('start', 'stop', 'pad', 'unk'):
        indexer_kwargs.setdefault(key, True)
    indexer = lang.indexer(annotations, **indexer_kwargs)

    return LanguageModel(indexer, **kwargs)
