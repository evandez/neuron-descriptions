"""The language model component of MILAN.

This is integrated with the decoder to compute PMI instead of probability
while decoding descriptions.
"""
from typing import Any, Mapping, Optional, Sequence, Sized, Type, Union, cast

from src.utils import lang, serialize, training
from src.utils.typing import Device, StrSequence

import torch
from torch import nn, optim
from torch.utils import data
from tqdm.auto import tqdm


class LanguageModel(serialize.SerializableModule):
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
                                    nn.LogSoftmax(dim=-1))

    def forward(self,
                inputs: torch.Tensor,
                reduce: bool = False,
                masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the log probability of the given sequence.

        Args:
            inputs (torch.Tensor): The sequence. Should have shape
                (batch_size, length) and have type `torch.long`.
            reduce (bool, optional): If set, reduce token-wise logprobs to
                sequence-wise logprobs. Returned tensor will now have shape
                (batch_size,) containing the log probability of each sequence
                according to the model. Note that this technically ignores the
                first token in `inputs`, which is assumed to be a start token!
                Defaults to False.
            masks (Optional[torch.Tensor], optional): Apply this mask to token
                probabilities when reducing (e.g., to ignore specific tokens).
                Must have shape (batch_size, len(inputs) - 1). By default,
                everything after the first stop token is masked.

        Returns:
            torch.Tensor: Shape (batch_size, len(inputs), vocab_size) tensor of
                log probabilites for each token, or shape (batch_size,) tensor
                containing log probabilities for each sequence if `reduce` is
                set.

        """
        embeddings = self.embedding(inputs)
        hiddens, _ = self.lstm(embeddings)
        lps = self.output(hiddens)
        if reduce:
            batch_size, length = inputs.shape
            idx_batch = torch.arange(batch_size).repeat_interleave(length - 1)
            idx_time = torch.arange(length - 1).repeat(batch_size)
            idx_tokens = inputs[:, 1:].reshape(-1)
            if masks is None:
                masks = inputs.new_ones((batch_size, length - 1))
                for i, j in inputs.eq(self.indexer.stop_index).nonzero():
                    masks[i, j + 1:] = 0
            lps = lps[:, :-1][idx_batch, idx_time, idx_tokens]\
                .view(batch_size, length - 1)\
                .mul(masks)\
                .sum(dim=-1)
        return lps

    def logp(self,
             sequences: StrSequence,
             device: Optional[Device] = None) -> torch.Tensor:
        """Compute log probability of each sequence.

        Args:
            sequences (StrSequence): Text sequences.
            device (Optional[Device], optional): Send this model and all
                tensors to this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show progress bar
                with this message if set. Defaults to 'compute lm probs'.

        Returns:
            torch.Tensor: Shape (len(sequences),) tensor containing the
                log probability for each sequence.

        """
        if device is not None:
            self.to(device)
        self.eval()

        inputs = torch.tensor(self.indexer(sequences,
                                           start=True,
                                           stop=True,
                                           pad=True,
                                           unk=True),
                              device=device)
        with torch.no_grad():
            logps = self(inputs, reduce=True)
        return logps

    def fit(self,
            dataset: data.Dataset,
            annotation_index: int = 4,
            batch_size: int = 128,
            max_epochs: int = 100,
            patience: int = 4,
            hold_out: Union[float, Sequence[int]] = .1,
            optimizer_t: Type[optim.Optimizer] = optim.AdamW,
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
            hold_out (Union[float, Sequence[int]], optional): Fraction of
                dataset to use for validation, or indices of samples to hold
                out. Defaults to .1.
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

            def __init__(self,
                         dataset: data.Dataset,
                         annotation_index: int = 4):
                self.sequences = []
                for index in range(len(cast(Sized, dataset))):
                    annotation = dataset[index][annotation_index]
                    if isinstance(annotation, str):
                        self.sequences.append(annotation)
                    else:
                        self.sequences += annotation

            def __getitem__(self, index: int) -> str:
                return self.sequences[index]

            def __len__(self) -> int:
                return len(self.sequences)

        # Prepare training data.
        dataset = SequenceDataset(dataset, annotation_index=annotation_index)
        if isinstance(hold_out, float):
            train, val = training.random_split(dataset, hold_out=hold_out)
        else:
            train, val = training.fixed_split(dataset, hold_out)
        train_loader = data.DataLoader(train,
                                       batch_size=batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(val, batch_size=batch_size)

        # Prepare optimizer, loss, training utils.
        optimizer = optimizer_t(self.parameters(), **optimizer_kwargs)
        criterion = nn.NLLLoss(ignore_index=self.indexer.pad_index)
        stopper = training.EarlyStopping(patience=patience)

        def lossify(sequences: StrSequence) -> torch.Tensor:
            inputs = torch.tensor(self.indexer(sequences,
                                               start=True,
                                               stop=False,
                                               pad=True,
                                               unk=True),
                                  device=device)
            targets = torch.tensor(self.indexer(sequences,
                                                start=False,
                                                stop=True,
                                                pad=True,
                                                unk=True),
                                   device=device)
            predictions = self(inputs)
            return criterion(predictions.permute(0, 2, 1), targets)

        progress = range(max_epochs)
        if display_progress_as is not None:
            progress = tqdm(progress, desc=display_progress_as)

        # Begin training!
        best = self.state_dict()
        for _ in progress:
            self.train()
            train_loss = 0.
            for sequences in train_loader:
                loss = lossify(sequences)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.eval()
            val_loss = 0.
            for sequences in val_loader:
                with torch.no_grad():
                    loss = lossify(sequences)
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

    def properties(self) -> serialize.Properties:
        """Override `Serializable.properties`."""
        return {
            'indexer': self.indexer,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'layers': self.layers,
            'dropout': self.dropout,
        }

    @classmethod
    def resolve(cls, children: serialize.Children) -> serialize.Resolved:
        """Override `Serializable.resolve`."""
        return {'indexer': lang.Indexer}


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
