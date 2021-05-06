"""Models for captioning neurons."""
from typing import (Any, Mapping, NamedTuple, Optional, Type, TypeVar, Union,
                    overload)

from lv.models import annotators, embeddings, featurizers
from lv.utils import lang, training
from lv.utils.typing import Device, StrSequence

import rouge
import sacrebleu
import torch
from torch import nn, optim
from torch.distributions import categorical
from torch.utils import data
from tqdm.auto import tqdm


class Attention(nn.Module):
    """Attention mechanism from Show, Attend, and Tell [Xu et al., 2015]."""

    def __init__(self,
                 query_size: int,
                 key_size: int,
                 hidden_size: Optional[int] = None):
        """Initialize the attention mechanism.

        Args:
            query_size (int): Size of query vectors.
            key_size (int): Size of key vectors.
            hidden_size (Optional[int], optional): Query and key will be mapped
                to this size before computing score. Defaults to min of
                query_size and key_size.

        """
        super().__init__()

        self.query_size = query_size
        self.key_size = key_size
        self.hidden_size = hidden_size or min(query_size, key_size)

        self.query_to_hidden = nn.Linear(query_size, self.hidden_size)
        self.key_to_hidden = nn.Linear(key_size, self.hidden_size)
        self.output = nn.Sequential(nn.Linear(self.hidden_size, 1),
                                    nn.Softmax(dim=1))

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Compute scores for (distribution over) keys given query.

        Args:
            query (torch.Tensor): Query vector, should have shape
                (batch_size, query_size).
            keys (torch.Tensor): Key vectors, should have shape
                (batch_size, n_keys, key_size).

        Returns:
            torch.Tensor: Scores for each key, with shape (batch_size, n_keys).

        """
        q_hidden = self.query_to_hidden(query).unsqueeze(1)
        k_hidden = self.key_to_hidden(keys)
        hidden = torch.tanh(q_hidden + k_hidden)
        return self.output(hidden).squeeze()


class DecoderOutput(NamedTuple):
    """Wraps output of the caption decoder."""

    logprobs: torch.Tensor
    tokens: torch.Tensor
    attention_vs: torch.Tensor
    attention_ws: torch.Tensor
    captions: StrSequence


DecoderT = TypeVar('DecoderT', bound='Decoder')
Strategy = Union[torch.Tensor, str]

STRATEGY_GREEDY = 'greedy'
STRATEGY_SAMPLE = 'sample'
STRATEGIES = (STRATEGY_GREEDY, STRATEGY_SAMPLE)


class Decoder(nn.Module):
    """Neuron caption decoder.

    Roughly mimics the architecture described in Show, Attend, and Tell
    [Xu et al., 2015]. The main difference is that the decoder attends over
    two sets of features independently during each step: visual features
    and word features. Word features can be extracted from ground truth
    captions or predicted from the `WordAnnotator` model.
    """

    def __init__(self,
                 indexer: lang.Indexer,
                 annotator: annotators.WordAnnotator,
                 featurizer_w: Optional[nn.Embedding] = None,
                 copy: bool = True,
                 embedding_size: int = 128,
                 hidden_size: int = 512,
                 attention_hidden_size: Optional[int] = None,
                 threshold: float = .5,
                 dropout: float = .5):
        """Initialize the decoder.

        Args:
            indexer (lang.Indexer): Indexer for captions. The vocab used by
                this indexer must be a superset of the `WordAnnotator` vocab.
            annotator (annotators.WordAnnotator): Annotator for predicting
                which words will appear in an image. Used to construct word
                features that are attended over at every decoding step.
            featurizer_w (Optional[nn.Embedding], optional): Embedding for
                words predicted by the annotator. Defaults to the built in
                word vectors of spacy's large model.
            copy (bool, optional): Use a copy mechanism in the model.
                Defaults to True.
            embedding_size (int, optional): Size of previous-word embeddings
                that are input to the LSTM. Defaults to 128.
            hidden_size (int, optional): Size of LSTM hidden states.
                Defaults to 512.
            attention_hidden_size (Optional[int], optional): Size of attention
                mechanism hidden layer. Defaults to minimum of visual feature
                size plus word feature size and `hidden_size`.
            threshold (float, optional): Probability cutoff for whether
                `WordAnnotator` predicts a word or not. Defaults to .5.
            dropout (float, optional): Dropout probability, applied before
                output layer of LSTM. Defaults to .5.

        Raises:
            ValueError: If annotator vocabulary is not a subset of captioner
                vocabulary, or if `featurizer_w` does not have correct number
                of vectors.

        """
        super().__init__()

        if not annotator.indexer.unique <= indexer.unique:
            raise ValueError('annotator vocabulary must be subset of indexer '
                             'vocabulary, but indexer is missing these words: '
                             f'{annotator.indexer.unique - indexer.unique} ')

        if featurizer_w is None:
            featurizer_w = embeddings.spacy(annotator.indexer)
        elif featurizer_w.num_embeddings != len(annotator.indexer):
            raise ValueError(f'featurizer_w has {featurizer_w.num_embeddings} '
                             'tokens but annotator has '
                             f'{len(annotator.indexer)} in its vocab')

        self.indexer = indexer
        self.annotator = annotator
        self.featurizer_w = featurizer_w
        self.copy = copy
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.threshold = threshold
        self.dropout = dropout

        self.feature_v_size = feature_v_size = annotator.feature_size
        self.feature_w_size = feature_w_size = featurizer_w.embedding_dim
        self.feature_size = feature_size = feature_v_size + feature_w_size
        self.vocab_size = vocab_size = len(indexer)

        self.attend_v = Attention(hidden_size,
                                  feature_v_size,
                                  hidden_size=attention_hidden_size)
        self.attend_w = Attention(hidden_size,
                                  feature_w_size,
                                  hidden_size=attention_hidden_size)
        self.feature_gate = nn.Sequential(nn.Linear(hidden_size, feature_size),
                                          nn.Sigmoid())

        self.init_h = nn.Sequential(nn.Linear(feature_size, hidden_size),
                                    nn.Tanh())
        self.init_c = nn.Sequential(nn.Linear(feature_size, hidden_size),
                                    nn.Tanh())

        self.copy_gate = None
        if copy:
            self.copy_gate = nn.Sequential(nn.Linear(hidden_size, 1),
                                           nn.Sigmoid())

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.output = nn.Sequential(nn.Dropout(p=dropout),
                                    nn.Linear(hidden_size, vocab_size),
                                    nn.LogSoftmax(dim=-1))

    @property
    def featurizer_v(self) -> featurizers.Featurizer:
        """Return the visual featurizer for this captioner."""
        return self.annotator.featurizer

    @overload
    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor,
                length: int = ...,
                strategy: Strategy = ...,
                captions: Optional[StrSequence] = ...,
                threshold: Optional[float] = ...) -> DecoderOutput:
        """Decode captions for the given top images and masks.

        Args:
            images (torch.Tensor): Top-k images for a neuron.
                Should have shape (batch_size, k, 3, height, width).
            masks (torch.Tensor): Top-k image masks for a neuron.
                Should have shape (batch_size, k, 1, height, width).
            length (int, optional): Decode for this many steps. Defaults to 15.
            strategy (Strategy, optional): Decoding strategy. If a tensor,
                values will be used as inputs at each time step, so it should
                have shape (batch_size, length). Other options include 'greedy'
                and 'sample'. Defaults to 'greedy'.
            captions (Optional[StrSequence], optional): Instead of using word
                annotator, extract words from these captions. Defaults to None.
            threshold (Optional[float], optional): Threshold for whether
                annotator predicts a word. Overrides the `threshold` field on
                this class if set. Defaults to None.

        Returns:
            DecoderOutput: Decoder outputs.

        """
        ...

    @overload
    def forward(self, features_v: torch.Tensor,
                **kwargs: Any) -> DecoderOutput:
        """Decode captions for the given visual features.

        Same as the other overload, but inputs are visual features.
        """
        ...

    def forward(self,
                images,
                masks=None,
                length=15,
                strategy=STRATEGY_GREEDY,
                captions=None,
                threshold=None):
        """Implement both overloads."""
        if isinstance(strategy, str) and strategy not in STRATEGIES:
            raise ValueError(f'unknown strategy: {strategy}')

        batch_size = len(images)
        if captions is not None and len(captions) != batch_size:
            raise ValueError(f'expected {batch_size} grount truth captions, '
                             f'got {len(captions)}')

        if threshold is None:
            threshold = self.threshold

        # If necessary, obtain visual features. Technically, backpropagating
        # through the featurizer is acceptable, so avoid using no_grad.
        if masks is not None:
            features_v = self.featurizer_v(images, masks)
        else:
            features_v = images

        # Obtain word features from word annotator or ground truth captions.
        if captions is None:
            with torch.no_grad():
                annos = self.annotator(features_v, threshold=threshold)
            annos_words = annos.words
            annos_idx = self.annotator.indexer.index(annos_words, pad=True)
        else:
            annos_idx = self.annotator.indexer(captions, pad=True)
            annos_words = self.annotator.indexer.unindex(annos_idx, pad=False)

        annos_idx_t = torch.tensor(annos_idx,
                                   dtype=torch.long,
                                   device=features_v.device)
        with torch.no_grad():
            features_w = self.featurizer_w(annos_idx_t)

        # Prepare outputs.
        tokens = features_v.new_zeros(batch_size, length, dtype=torch.long)
        logprobs = features_v.new_zeros(batch_size, length, self.vocab_size)
        attention_vs = features_v.new_zeros(batch_size, length,
                                            features_v.shape[1])
        attention_ws = features_v.new_zeros(batch_size, length,
                                            annos_idx_t.shape[-1])

        # Compute initial hidden state and cell value.
        features = (features_v, features_w)
        pooled = torch.cat([fs.mean(dim=1) for fs in features], dim=-1)
        h, c = self.init_h(pooled), self.init_c(pooled)

        # Begin decoding.
        currents = tokens.new_empty(batch_size).fill_(self.indexer.start_index)
        for time in range(length):
            # Attend over visual features.
            attention_vs[:, time] = attention_v = self.attend_v(h, features_v)
            attenuated_v = attention_v.unsqueeze(-1).mul(features_v).sum(dim=1)

            # Attend over word featuers.
            attention_ws[:, time] = attention_w = self.attend_w(h, features_w)
            attenuated_w = attention_w.unsqueeze(-1).mul(features_w).sum(dim=1)

            # Concatenate and gate attenuated features.
            attenuated = torch.cat((attenuated_v, attenuated_w), dim=-1)
            gate = self.feature_gate(h)
            gated = attenuated * gate

            # Prepare LSTM inputs and take a step.
            embeddings = self.embedding(currents)
            inputs = torch.cat((embeddings, gated), dim=-1)
            h, c = self.lstm(inputs, (h, c))
            logprobs[:, time] = log_p_w = self.output(h)

            # If copy mechanism is enabled, apply it.
            if self.copy_gate is not None:
                word_idx = [self.indexer[w] for ws in annos_words for w in ws]
                batch_idx = [
                    idx for idx, ws in enumerate(annos_words)
                    for _ in range(len(ws))
                ]

                p_copy = self.copy_gate(h)
                p_copy_w = torch.zeros_like(log_p_w)
                p_copy_w[batch_idx, word_idx] = torch.cat([
                    attention_w[idx, :len(words)]
                    for idx, words in enumerate(annos_words)
                ])
                p_w = (1 - p_copy) * torch.exp(log_p_w) + p_copy * p_copy_w
                logprobs[:, time] = log_p_w = torch.log(p_w)

            # Pick next token by applying the decoding strategy.
            if isinstance(strategy, torch.Tensor):
                currents = strategy[:, time]
            elif strategy == STRATEGY_GREEDY:
                currents = logprobs[:, time].argmax(dim=1)
            else:
                assert strategy == STRATEGY_SAMPLE
                for index, lp in enumerate(logprobs[:, time]):
                    distribution = categorical.Categorical(probs=torch.exp(lp))
                    currents[index] = distribution.sample()
            tokens[:, time] = currents

        return DecoderOutput(
            logprobs=logprobs,
            tokens=tokens,
            attention_vs=attention_vs,
            attention_ws=attention_ws,
            captions=self.indexer.reconstruct(tokens.tolist()),
        )

    def bleu(self,
             dataset: data.Dataset,
             annotation_index: int = 4,
             predictions: Optional[StrSequence] = None,
             **kwargs: Any) -> sacrebleu.BLEUScore:
        """Compute BLEU score of this model on the given dataset.

        Keyword arguments forwarded to `Decoder.predict` if `predictions` not
        provided.

        Args:
            dataset (data.Dataset): The test dataset.
            annotation_index (int, optional): Index of language annotations in
                dataset samples. Defaults to 4 to be compatible with
                AnnotatedTopImagesDataset.
            predictions (Optional[StrSequence], optional): Precomputed
                predicted captions for all images in the dataset.
                By default, computed from the dataset using `Decoder.predict`.

        Returns:
            sacrebleu.BLEUScore: Corpus BLEU score.

        """
        if predictions is None:
            predictions = self.predict(dataset, **kwargs)

        references = []
        for index in range(len(predictions)):
            annotations = dataset[index][annotation_index]
            if isinstance(annotations, str):
                annotations = [annotations]
            references.append(annotations)

        return sacrebleu.corpus_bleu(predictions, list(zip(*references)))

    def rouge(self,
              dataset: data.Dataset,
              annotation_index: int = 4,
              predictions: Optional[StrSequence] = None,
              **kwargs: Any) -> Mapping[str, Mapping[str, float]]:
        """Compute ROUGe score of this model on the given dataset.

        Keyword arguments forwarded to `Decoder.predict` if `predictions` not
        provided.

        Args:
            dataset (data.Dataset): The test dataset.
            annotation_index (int, optional): Index of language annotations in
                dataset samples. Defaults to 4 to be compatible with
                AnnotatedTopImagesDataset.
            predictions (Optional[StrSequence], optional): Precomputed
                predicted captions for all images in the dataset.
                By default, computed from the dataset using `Decoder.predict`.

        Returns:
            Mapping[str, Mapping[str, float]]: Average ROUGe (1, 2, l) scores.

        """
        if predictions is None:
            predictions = self.predict(dataset, **kwargs)

        hypotheses, references = [], []
        for index, prediction in enumerate(predictions):
            annotations = dataset[index][annotation_index]
            if isinstance(annotations, str):
                annotations = [annotations]
            for annotation in annotations:
                hypotheses.append(prediction)
                references.append(annotation)

        scorer = rouge.Rouge()
        return scorer.get_scores(hypotheses, references, avg=True)

    def predict(self,
                dataset: data.Dataset,
                image_index: int = 2,
                mask_index: int = 3,
                batch_size: int = 16,
                features: Optional[data.TensorDataset] = None,
                device: Optional[Device] = None,
                display_progress: bool = True,
                **kwargs: Any) -> StrSequence:
        """Feed entire dataset through the decoder.

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
            StrSequence: Captions for entire dataset.

        """
        if 'captions' in kwargs:
            raise ValueError('setting captions= not supported')
        if device is not None:
            self.to(device)
        if features is None:
            features = self.featurizer_v.map(dataset,
                                             image_index=image_index,
                                             mask_index=mask_index,
                                             batch_size=batch_size,
                                             device=device,
                                             display_progress=display_progress)

        loader = data.DataLoader(features, batch_size=batch_size)

        outputs = []
        for (inputs,) in tqdm(loader) if display_progress else loader:
            with torch.no_grad():
                output = self(inputs, **kwargs)
            outputs.append(output)

        captions = []
        for output in outputs:
            captions += output.captions

        return tuple(captions)

    @classmethod
    def fit(cls: Type[DecoderT],
            dataset: data.Dataset,
            annotator: annotators.WordAnnotatorT,
            image_index: int = 2,
            mask_index: int = 3,
            annotation_index: int = 4,
            batch_size: int = 64,
            max_epochs: int = 100,
            patience: Optional[int] = None,
            hold_out: float = .1,
            regularization_weight: float = 1.,
            use_ground_truth_words: bool = True,
            optimizer_t: Type[optim.Optimizer] = optim.Adam,
            optimizer_kwargs: Optional[Mapping[str, Any]] = None,
            indexer_kwargs: Optional[Mapping[str, Any]] = None,
            features: Optional[data.TensorDataset] = None,
            device: Optional[Device] = None,
            display_progress: bool = True,
            **kwargs: Any) -> DecoderT:
        """Train a new decoder on the given data.

        Args:
            dataset (data.Dataset): Dataset to train on.
            annotator (annotators.WordAnnotatorT): Word annotator model.
            image_index (int, optional): Index of images in dataset samples.
                Defaults to 2 to be compatible with AnnotatedTopImagesDataset.
            mask_index (int, optional): Index of masks in dataset samples.
                Defaults to 3 to be compatible with AnnotatedTopImagesDataset.
            annotation_index (int, optional): Index of language annotations in
                dataset samples. Defaults to 4 to be compatible with
                AnnotatedTopImagesDataset.
            batch_size (int, optional): Number of samples to train on at once.
                Defaults to 64.
            max_epochs (int, optional): Maximum number of epochs to train for.
                Defaults to 1000.
            patience (int, optional): If loss does not improve for this many
                epochs, stop training. By default, no early stopping.
            hold_out (float, optional): Fraction of data to hold out as a
                validation set. Defaults to .1.
            regularization_weight (float, optional): Weight for double
                stochasticity regularization. See [Xu et al., 2015] for
                details. Defaults to 1..
            use_ground_truth_words (bool, optional): Instead of conditioning
                on words predicted by the word annotator, condition on words
                from the ground truth captions instead. Defaults to True.
            optimizer_t (Type[optim.Optimizer], optional): Optimizer type.
                Defaults to optim.Adam.
            optimizer_kwargs (Optional[Mapping[str, Any]], optional): Optimizer
                options. By default, no kwargs are passed to optimizer.
            indexer_kwargs (Optional[Mapping[str, Any]], optional): Indexer
                options. By default, indexer is configured to not ignore stop
                words and punctuation.
            features (Optional[data.TensorDataset], optional): Precomputed
                visual features. By default, computed before training the
                captioner.
            device (Optional[Device], optional): Send all models and data
                to this device. Defaults to None.
            display_progress (bool, optional): Show progress bar while
                training. Defaults to True.

        Returns:
            DecoderT: The trained decoder.

        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if indexer_kwargs is None:
            indexer_kwargs = {}
        if features is None:
            features = annotator.featurizer.map(
                dataset,
                image_index=image_index,
                mask_index=mask_index,
                batch_size=batch_size,
                device=device,
                display_progress=display_progress)

        # Prepare indexer.
        annotations = []
        for index in range(len(features)):
            annotation = dataset[index][annotation_index]
            annotation = lang.join(annotation)
            annotations.append(annotation)

        indexer_kwargs = dict(indexer_kwargs)
        if 'tokenize' not in indexer_kwargs:
            indexer_kwargs['tokenize'] = lang.tokenizer(ignore_stop=False,
                                                        ignore_punct=False)
        for key in ('start', 'stop', 'pad', 'unk'):
            indexer_kwargs.setdefault(key, True)
        indexer = lang.indexer(annotations, **indexer_kwargs)

        # Prepare dataset and data loader. Use an anonymous dataset class to
        # make this easier, since we want to split train/val by neuron, but
        # iterate by annotation (for which there are ~3x per neuron).
        class WrapperDataset(data.Dataset):

            def __init__(self, subset):
                self.samples = []
                for index in subset.indices:
                    feature, = features[index]

                    annotations = dataset[index][annotation_index]
                    if isinstance(annotations, str):
                        sample = (feature, annotations)
                        self.samples.append(sample)
                        continue

                    for annotation in annotations:
                        sample = (feature, annotation)
                        self.samples.append(sample)

            def __getitem__(self, index):
                return self.samples[index]

            def __len__(self):
                return len(self.samples)

        val_size = int(hold_out * len(features))
        train_size = len(features) - val_size
        train, val = data.random_split(dataset, (train_size, val_size))
        train_loader = data.DataLoader(WrapperDataset(train),
                                       batch_size=batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(WrapperDataset(val),
                                     batch_size=batch_size)

        # Prepare model and training tools.
        model = cls(indexer, annotator, **kwargs).to(device)
        optimizer = optimizer_t(model.parameters(), **optimizer_kwargs)
        criterion = nn.NLLLoss(ignore_index=indexer.pad_index)

        stopper = None
        if patience is not None:
            stopper = training.EarlyStopping(patience=patience)

        progress = range(max_epochs)
        if display_progress:
            progress = tqdm(progress)

        # Begin training!
        for _ in progress:
            model.train()
            train_loss, train_reg = 0., 0.
            for features_v, captions in train_loader:
                targets = torch.tensor(indexer(captions), device=device)[:, 1:]
                _, length = targets.shape

                outputs = model(
                    features_v,
                    length=length,
                    strategy=targets,
                    captions=captions if use_ground_truth_words else None)

                loss = criterion(outputs.logprobs.permute(0, 2, 1), targets)
                train_loss += loss.item()

                regularizer = ((1 - outputs.attention_vs.sum(dim=1))**2).mean()
                train_reg += regularizer.item()
                loss += regularization_weight * regularizer

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_reg /= len(train_loader)

            model.eval()
            val_loss = 0.
            for features_v, captions in val_loader:
                targets = torch.tensor(indexer(captions), device=device)[:, 1:]
                _, length = targets.shape

                with torch.no_grad():
                    outputs = model(
                        features_v,
                        length=length,
                        strategy=targets,
                        captions=captions if use_ground_truth_words else None)
                    loss = criterion(outputs.logprobs.permute(0, 2, 1),
                                     targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)

            if display_progress:
                assert not isinstance(progress, range)
                progress.set_description(f'train_loss={train_loss:.3f}, '
                                         f'train_reg={train_reg:.3f}, '
                                         f'val_loss={val_loss:.3f}')

            if stopper is not None and stopper(val_loss):
                break

        return model
