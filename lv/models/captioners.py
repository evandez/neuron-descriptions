"""Models for captioning neurons."""
from typing import (Any, Mapping, NamedTuple, Optional, Sequence, Sized, Tuple,
                    Type, TypeVar, Union, cast, overload)

from lv.models import annotators, featurizers, lms, vectors
from lv.utils import lang, serialize, training
from lv.utils.typing import Device, StrSequence

import numpy
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
        return self.output(hidden).view(*keys.shape[:2])


WORD2VEC_SPACY = 'spacy'
WORD2VECS = (WORD2VEC_SPACY,)


class WordFeaturizer(serialize.SerializableModule):
    """Wrap a WordAnnotator and pretrained word vectors."""

    def __init__(self,
                 annotator: annotators.WordAnnotator,
                 word2vec: str = WORD2VEC_SPACY,
                 threshold: float = .5,
                 num_words: int = 10):
        """Initialize the word featurizer.

        Args:
            annotator (annotators.WordAnnotator): Word annotation model that
                predicts words given visual features.
            word2vec (str, optional): Pretrained word vectors to use.
                Defaults to WORD2VEC_SPACY.
            threshold (float, optional): Default threshold to use when
                predicting applicable words. Defaults to .5.
            num_words (int, optional): Exact number of words to take from
                the predictor. If ground truth captions are provided and
                contain fewer than this many words, the difference will be
                filled with padding tokens. Defaults to 10.

        Raises:
            ValueError: If unknown word vectors model are specified.

        """
        super().__init__()
        self.annotator = annotator
        self.word2vec = word2vec
        self.threshold = threshold
        self.num_words = num_words

        if word2vec == WORD2VEC_SPACY:
            self.vectors = vectors.spacy(annotator.indexer)
        else:
            raise ValueError(f'unknown word2vec type: {word2vec}')

    @property
    def feature_size(self) -> int:
        """Return the size of the word vectors."""
        return self.vectors.embedding_dim

    @overload
    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        threshold: Optional[float] = ...,
        num_words: Optional[int] = ...,
        captions: Optional[StrSequence] = ...,
    ) -> Tuple[torch.Tensor, Sequence[StrSequence]]:
        """Predict words from visual features and return their word vectors.

        Args:
            images (torch.Tensor): Top images. Should have shape
                (batch_size, k, 3, height, width).
            masks (torch.Tensor): Top image masks. Should have shape
                (batch_size, k, 1, height, width).
            threshold (Optional[float], optional): Only look at words predicted
                with this probability or higher. Defaults to `self.threshold`.
            num_words (Optional[int], optional): Maximum number of words to
                return. Defaults to `self.num_words`.
            captions (Optional[StrSequence], optional): Take words from these
                captions instead of the word predictor. Must have length same
                as batch_size. Defaults to None.

        Raises:
            ValueError: If captions is set and has bad length.

        Returns:
            Tuple[torch.Tensor, Sequence[StrSequence]]: Word vectors with
                shape (batch_size, num_words, vector_size).

        """
        ...

    @overload
    def forward(self, features_v: torch.Tensor,
                **kwargs: Any) -> Tuple[torch.Tensor, Sequence[StrSequence]]:
        """Predict words from (precomputed) visual features.

        Same as other overload, but assumes features already are computed.
        """
        ...

    def forward(self,
                images,
                masks=None,
                threshold=None,
                num_words=None,
                captions=None):
        """Implement both overloads."""
        if captions is not None and len(captions) != len(images):
            raise ValueError(f'expected {len(images)} ground truth '
                             f'captions, got {len(captions)}')

        if threshold is None:
            threshold = self.threshold
        if num_words is None:
            num_words = self.num_words

        if captions is None:
            annos: annotators.WordAnnotations
            with torch.no_grad():
                annos = self.annotator(images, masks, threshold=threshold)
            idx = annos.probabilities.topk(k=self.num_words).indices.tolist()
            words = self.annotator.indexer.unindex(idx, specials=False)
        else:
            idx = self.annotator.indexer(captions,
                                         pad=True,
                                         unk=False,
                                         length=self.num_words)
            words = self.annotator.indexer.unindex(idx, specials=False)

        idx_t = torch.tensor(idx, dtype=torch.long, device=images.device)
        with torch.no_grad():
            features_w = self.vectors(idx_t)

        return features_w, words

    def properties(self, **kwargs):
        """Override `SerializableModule.properties`."""
        properties = super().properties(**kwargs)
        properties.update({
            'annotator': self.annotator,
            'word2vec': self.word2vec,
            'threshold': self.threshold,
            'num_words': self.num_words,
        })

        state_dict = properties.get('state_dict', {})

        # If the featurizer is serializable, remove its parameters from the
        # state dict and serialize instead. We only have one Serializable
        # featurizer type, so just check for that.
        featurizer_v = self.annotator.featurizer
        if isinstance(featurizer_v, featurizers.MaskedPyramidFeaturizer):
            keys = [
                key for key in state_dict
                if key.startswith('annotator.featurizer.')
            ]
            for key in keys:
                del state_dict[key]

        return properties

    @classmethod
    def recurse(cls):
        """Override `SerializableModule.recurse`."""
        return {'annotator': annotators.WordAnnotator}


class DecoderOutput(NamedTuple):
    """Wraps output of the caption decoder."""

    captions: StrSequence
    logprobs: torch.Tensor
    tokens: torch.Tensor
    attention_vs: Optional[torch.Tensor]
    attention_ws: Optional[torch.Tensor]


DecoderT = TypeVar('DecoderT', bound='Decoder')
Strategy = Union[torch.Tensor, str]

STRATEGY_GREEDY = 'greedy'
STRATEGY_SAMPLE = 'sample'
STRATEGIES = (STRATEGY_GREEDY, STRATEGY_SAMPLE)


class Decoder(serialize.SerializableModule):
    """Neuron caption decoder.

    Roughly mimics the architecture described in Show, Attend, and Tell
    [Xu et al., 2015]. The main difference is that the decoder attends over
    two sets of features independently during each step: visual features
    and word features. Word features can be extracted from ground truth
    captions or predicted from the `WordAnnotator` model.
    """

    def __init__(self,
                 indexer: lang.Indexer,
                 featurizer_v: Optional[featurizers.Featurizer] = None,
                 featurizer_w: Optional[WordFeaturizer] = None,
                 lm: Optional[lms.LanguageModel] = None,
                 copy: bool = False,
                 embedding_size: int = 128,
                 hidden_size: int = 512,
                 attention_hidden_size: Optional[int] = None,
                 dropout: float = .5):
        """Initialize the decoder.

        Args:
            indexer (lang.Indexer): Indexer for captions. The vocab used by
                this indexer must be a superset of the `WordAnnotator` vocab.
            featurizer_v (Optional[featurizers.Featurizer], optional): Visual
                featurizer. If this is not set, `featurizer_w` must be.
            featurizer_w (Optional[WordFeaturizer], optional): Word featurizer.
                If this is not set, `featurizer_v` must be. By default, only
                visual features are used.
            lm (Optional[lms.LanguageModel], optional): Language model. Changes
                decoding to use PMI [p(caption | image) / p(caption)] instead
                of likelihood [p(caption | image)].
            copy (bool, optional): Use a copy mechanism in the model.
                Defaults to False.
            embedding_size (int, optional): Size of previous-word embeddings
                that are input to the LSTM. Defaults to 128.
            hidden_size (int, optional): Size of LSTM hidden states.
                Defaults to 512.
            attention_hidden_size (Optional[int], optional): Size of attention
                mechanism hidden layer. Defaults to minimum of visual feature
                size plus word feature size and `hidden_size`.
            dropout (float, optional): Dropout probability, applied before
                output layer of LSTM. Defaults to .5.

        Raises:
            ValueError: If annotator vocabulary is not a subset of captioner
                vocabulary, or if `featurizer_w` does not have correct number
                of vectors.

        """
        super().__init__()

        if featurizer_v is None and featurizer_w is None:
            raise ValueError('must set at least one of '
                             'featurizer_v and featurizer_w')

        if copy:
            if featurizer_w is None:
                raise ValueError('must set featurizer_w if copy=True')
            elif featurizer_w.annotator.indexer.unique <= indexer.unique:
                raise ValueError(
                    'when using a copy mechanism, annotator vocab must be a '
                    'subset of indexer vocab, but indexer is missing words: '
                    f'{featurizer_w.annotator.indexer.unique - indexer.unique}'
                )

        self.indexer = indexer
        self.featurizer_v = featurizer_v
        self.featurizer_w = featurizer_w
        self.lm = lm
        self.copy = copy
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.dropout = dropout

        self.feature_v_size, self.attend_v = None, None
        if featurizer_v is not None:
            self.feature_v_size = numpy.prod(featurizer_v.feature_shape).item()
            self.attend_v = Attention(hidden_size,
                                      self.feature_v_size,
                                      hidden_size=attention_hidden_size)

        self.feature_w_size, self.attend_w = None, None
        if featurizer_w is not None:
            self.feature_w_size = featurizer_w.vectors.embedding_dim
            self.attend_w = Attention(hidden_size,
                                      self.feature_w_size,
                                      hidden_size=attention_hidden_size)

        self.feature_size = self.feature_v_size or 0
        self.feature_size += self.feature_w_size or 0
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_size, self.feature_size),
            nn.Sigmoid(),
        )

        self.init_h = nn.Sequential(nn.Linear(self.feature_size, hidden_size),
                                    nn.Tanh())
        self.init_c = nn.Sequential(nn.Linear(self.feature_size, hidden_size),
                                    nn.Tanh())

        self.copy_gate = None
        if copy:
            self.copy_gate = nn.Sequential(nn.Linear(hidden_size, 1),
                                           nn.Sigmoid())

        self.vocab_size = len(indexer)
        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.lstm = nn.LSTMCell(embedding_size + self.feature_size,
                                hidden_size)
        self.output = nn.Sequential(nn.Dropout(p=dropout),
                                    nn.Linear(hidden_size, self.vocab_size),
                                    nn.LogSoftmax(dim=-1))

    @property
    def featurizer_v_(self) -> featurizers.Featurizer:
        """Return the visual featurizer used in this model.

        This function ends with _ because it descends into the word annotator,
        if necessary, to find its visual featurizer, which is guaranteed to
        be there.
        """
        featurizer_v = self.featurizer_v
        if featurizer_v is None:
            assert self.featurizer_w is not None
            featurizer_v = self.featurizer_w.annotator.featurizer
        return featurizer_v

    @overload
    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor,
                length: int = ...,
                strategy: Strategy = ...,
                **kwargs: Any) -> DecoderOutput:
        """Decode captions for the given top images and masks.

        Keyword arguments are forwarded to WordFeaturizer, if decoder has one.

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
                **kwargs):
        """Implement both overloads."""
        if isinstance(strategy, str) and strategy not in STRATEGIES:
            raise ValueError(f'unknown strategy: {strategy}')

        batch_size = len(images)

        # If necessary, obtain visual features.
        features_v: Optional[torch.Tensor] = None
        if self.featurizer_v is not None:
            if masks is not None:
                images = images.view(-1, 3, *images.shape[-2:])
                masks = masks.view(-1, 1, *masks.shape[-2:])
                with torch.no_grad():
                    features_v = self.featurizer_v(images, masks)
            else:
                features_v = images
            features_v = features_v.view(batch_size, -1, self.feature_v_size)

        # Obtain word features from word annotator or ground truth captions.
        features_w: Optional[torch.Tensor] = None
        words: Optional[Sequence[StrSequence]] = None
        if self.featurizer_w is not None:
            if features_v is not None:
                features_w, words = self.featurizer_w(features_v, **kwargs)
            else:
                features_w, words = self.featurizer_w(images, masks, **kwargs)

        # Prepare outputs.
        tokens = images.new_zeros(batch_size, length, dtype=torch.long)
        logprobs = images.new_zeros(batch_size, length, self.vocab_size)

        attention_vs = None
        if self.featurizer_v is not None:
            assert features_v is not None
            attention_vs = images.new_zeros(batch_size, length,
                                            features_v.shape[1])

        attention_ws = None
        if self.featurizer_w is not None:
            assert features_w is not None
            attention_ws = images.new_zeros(batch_size, length,
                                            features_w.shape[1])

        # Compute initial hidden state and cell value.
        pooled = torch.cat(
            [
                fs.mean(dim=1)
                for fs in (features_v, features_w)
                if fs is not None
            ],
            dim=-1,
        )
        h, c = self.init_h(pooled), self.init_c(pooled)

        # If necessary, compute LM initial hidden state and cell value.
        h_lm, c_lm = None, None
        if self.lm is not None:
            h_lm = h.new_zeros(batch_size, self.lm.hidden_size)
            c_lm = c.new_zeros(batch_size, self.lm.hidden_size)

        # Begin decoding.
        currents = tokens.new_empty(batch_size).fill_(self.indexer.start_index)
        for time in range(length):
            # Attend over visual features.
            attention_v, attenuated_v = None, None
            if self.featurizer_v is not None:
                assert features_v is not None
                assert attention_vs is not None
                assert self.attend_v is not None
                attention_v = self.attend_v(h, features_v)
                attention_vs[:, time] = attention_v
                attenuated_v = attention_v.unsqueeze(-1).mul(features_v).sum(1)

            # Attend over word featuers.
            attention_w, attenuated_w = None, None
            if self.featurizer_w is not None:
                assert features_w is not None
                assert attention_ws is not None
                assert self.attend_w is not None
                attention_w = self.attend_w(h, features_w)
                attention_ws[:, time] = attention_w
                attenuated_w = attention_w.unsqueeze(-1).mul(features_w).sum(1)

            # Concatenate and gate attenuated features.
            attenuated = torch.cat(
                [
                    attenuated for attenuated in (attenuated_v, attenuated_w)
                    if attenuated is not None
                ],
                dim=-1,
            )
            gate = self.feature_gate(h)
            gated = attenuated * gate

            # Prepare LSTM inputs and take a step.
            embeddings = self.embedding(currents)
            inputs = torch.cat((embeddings, gated), dim=-1)
            h, c = self.lstm(inputs, (h, c))
            logprobs[:, time] = log_p_w = self.output(h)

            if self.lm is not None:
                assert h_lm is not None and c_lm is not None
                inputs_lm = self.lm.embedding(currents)[None]
                _, (h_lm, c_lm) = self.lm.lstm(inputs_lm, (h_lm, c_lm))
                log_p_w_lm = self.lm.output(h_lm)
                logprobs[:, time] = log_p_w - log_p_w_lm

            # If copy mechanism is enabled, apply it.
            if self.copy_gate is not None:
                assert words is not None
                assert attention_w is not None
                word_idx = [self.indexer[w] for ws in words for w in ws]
                batch_idx = [
                    idx for idx, ws in enumerate(words) for _ in range(len(ws))
                ]

                p_copy = self.copy_gate(h)
                p_copy_w = torch.zeros_like(log_p_w)
                p_copy_w[batch_idx, word_idx] = torch.cat([
                    attention_w[idx, :len(words)]
                    for idx, words in enumerate(words)
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
            captions=self.indexer.reconstruct(tokens.tolist()),
            logprobs=logprobs,
            tokens=tokens,
            attention_vs=attention_vs,
            attention_ws=attention_ws,
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
        predictions = [pred.lower().strip('. ') for pred in predictions]

        references = []
        for index in range(len(predictions)):
            annotations = dataset[index][annotation_index]
            if isinstance(annotations, str):
                annotations = [annotations]
            # Preprocess target annotations in the same way as the predictions.
            annotations = [anno.lower().strip('. ') for anno in annotations]
            references.append(annotations)

        return sacrebleu.corpus_bleu(predictions, list(zip(*references)))

    def rouge(self,
              dataset: data.Dataset,
              annotation_index: int = 4,
              predictions: Optional[StrSequence] = None,
              **kwargs: Any) -> Mapping[str, Mapping[str, float]]:
        """Compute ROUGE score of this model on the given dataset.

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
            Mapping[str, Mapping[str, float]]: Average ROUGE (1, 2, l) scores.

        """
        if predictions is None:
            predictions = self.predict(dataset, **kwargs)

        hypotheses, references = [], []
        for index, prediction in enumerate(predictions):
            prediction = prediction.lower().strip('. ')

            annotations = dataset[index][annotation_index]
            if isinstance(annotations, str):
                annotations = [annotations]

            # Preprocess target annotations in the same way model was trained.
            for annotation in annotations:
                annotation = annotation.lower().strip('. ')

                # If annotation contains all unknown words, filter it.
                if not annotation:
                    continue

                hypotheses.append(prediction)
                references.append(annotation)

        scorer = rouge.Rouge()
        return scorer.get_scores(hypotheses,
                                 references,
                                 avg=True,
                                 ignore_empty=True)

    def predict(self,
                dataset: data.Dataset,
                image_index: int = 2,
                mask_index: int = 3,
                batch_size: int = 16,
                features: Optional[data.TensorDataset] = None,
                num_workers: int = 0,
                device: Optional[Device] = None,
                display_progress_as: Optional[str] = 'predict captions',
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
            num_workers (int, optional): Number of workers for loading data.
                Defaults to 0.
            device (Optional[Device], optional): Send model and data to this
                device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this key. Defaults to 'predict captions'.

        Returns:
            StrSequence: Captions for entire dataset.

        """
        if 'captions' in kwargs:
            raise ValueError('setting captions= not supported')
        if device is not None:
            self.to(device)
        if features is None:
            features = self.featurizer_v_.map(
                dataset,
                image_index=image_index,
                mask_index=mask_index,
                batch_size=batch_size,
                device=device,
                display_progress=display_progress_as and 'featurize dataset')

        loader = data.DataLoader(features,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        if display_progress_as is not None:
            loader = tqdm(loader, desc=display_progress_as)

        outputs = []
        for (inputs,) in loader:
            with torch.no_grad():
                output = self(inputs, **kwargs)
            outputs.append(output)

        captions = []
        for output in outputs:
            captions += output.captions

        return tuple(captions)

    def fit(self,
            dataset: data.Dataset,
            image_index: int = 2,
            mask_index: int = 3,
            annotation_index: int = 4,
            batch_size: int = 64,
            max_epochs: int = 100,
            patience: int = 4,
            hold_out: float = .1,
            regularization_weight: float = 1.,
            use_ground_truth_words: bool = True,
            optimizer_t: Type[optim.Optimizer] = optim.Adam,
            optimizer_kwargs: Optional[Mapping[str, Any]] = None,
            features: Optional[data.TensorDataset] = None,
            num_workers: int = 0,
            device: Optional[Device] = None,
            display_progress_as: Optional[str] = 'train decoder') -> None:
        """Train a new decoder on the given data.

        Args:
            dataset (data.Dataset): Dataset to train on.
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
                epochs, stop training. Defaults to 4.
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
            features (Optional[data.TensorDataset], optional): Precomputed
                visual features. By default, computed before training the
                captioner.
            num_workers (int, optional): Number of workers for loading data.
                Defaults to 0.
            device (Optional[Device], optional): Send all models and data
                to this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this key. Defaults to 'train captioner'.

        """
        if device is not None:
            self.to(device)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if features is None:
            features = self.featurizer_v_.map(dataset,
                                              image_index=image_index,
                                              mask_index=mask_index,
                                              batch_size=batch_size,
                                              device=device,
                                              progress=display_progress_as and
                                              'featurize dataset')

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
                                       num_workers=num_workers,
                                       batch_size=batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(WrapperDataset(val),
                                     num_workers=num_workers,
                                     batch_size=batch_size)

        # Prepare model and training tools.
        optimizer = optimizer_t(self.parameters(), **optimizer_kwargs)
        criterion = nn.NLLLoss(ignore_index=self.indexer.pad_index)
        stopper = training.EarlyStopping(patience=patience)

        progress = range(max_epochs)
        if display_progress_as is not None:
            progress = tqdm(progress, desc=display_progress_as)

        # Begin training!
        for _ in progress:
            self.train()
            self.featurizer_v_.eval()
            train_loss = 0.
            for features_v, captions in train_loader:
                targets = torch.tensor(self.indexer(captions),
                                       device=device)[:, 1:]
                _, length = targets.shape

                outputs: DecoderOutput = self(
                    features_v,
                    length=length,
                    strategy=targets,
                    captions=captions if use_ground_truth_words else None)

                loss = criterion(outputs.logprobs.permute(0, 2, 1), targets)

                attention_vs = outputs.attention_vs
                if attention_vs is not None:
                    regularizer = ((1 - attention_vs.sum(dim=1))**2).mean()
                    loss += regularization_weight * regularizer

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.eval()
            val_loss = 0.
            for features_v, captions in val_loader:
                targets = torch.tensor(self.indexer(captions),
                                       device=device)[:, 1:]
                _, length = targets.shape

                with torch.no_grad():
                    outputs = self(
                        features_v,
                        length=length,
                        strategy=targets,
                        captions=captions if use_ground_truth_words else None)
                    loss = criterion(outputs.logprobs.permute(0, 2, 1),
                                     targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)

            if display_progress_as is not None:
                assert not isinstance(progress, range)
                progress.set_description(f'{display_progress_as} '
                                         f'[train_loss={train_loss:.3f}, '
                                         f'val_loss={val_loss:.3f}]')

            if stopper(val_loss):
                break

    def properties(self, **kwargs):
        """Override `SerializableModule.properties`."""
        properties = super().properties(**kwargs)
        properties.update({
            'indexer': self.indexer,
            'copy': self.copy,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'attention_hidden_size': self.attention_hidden_size,
            'dropout': self.dropout,
        })

        state_dict = properties.get('state_dict', {})
        delete = []

        featurizer_v = self.featurizer_v
        if featurizer_v is not None and isinstance(
                featurizer_v, featurizers.MaskedPyramidFeaturizer):
            delete += [
                key for key in state_dict if key.startswith('featurizer_v.')
            ]
            properties['featurizer_v'] = featurizer_v

        featurizer_w = self.featurizer_w
        if featurizer_w is not None:
            properties['featurizer_w'] = featurizer_w
            if isinstance(featurizer_w.featurizer,
                          featurizers.MaskedPyramidFeaturizer):
                delete += [
                    key for key in state_dict
                    if key.startswith('featurizer_w.annotator.featurizer.')
                ]

        for key in delete:
            del state_dict[key]

        return properties

    @classmethod
    def recurse(cls):
        """Override `SerializableModule.recurse`."""
        return {
            'featurizer_w': WordFeaturizer,
            'featurizer_v': featurizers.MaskedPyramidFeaturizer,
            'indexer': lang.Indexer,
        }


def decoder(dataset: data.Dataset,
            featurizer: Optional[featurizers.Featurizer] = None,
            annotator: Optional[annotators.WordAnnotator] = None,
            lm: Optional[lms.LanguageModel] = None,
            annotation_index: int = 4,
            indexer_kwargs: Optional[Mapping[str, Any]] = None,
            word_featurizer_kwargs: Optional[Mapping[str, Any]] = None,
            **kwargs: Any) -> Decoder:
    """Instantiate a new decoder.

    Args:
        dataset (data.Dataset): Dataset to draw vocabulary from.
        featurizer (Optional[featurizers.Featurer], optional): Visual
            featurizer. Must set this or `annotator`, or both.
            Defaults to None.
        annotator (Optional[annotators.WordAnnotator], optional): Word
            annotation model. Must set this or `featurizer`, or both.
            Defaults to None.
        lm (Optional[lms.LanguageModel], optional): Language model.
            Defaults to None.
        annotation_index (int, optional): Index of language annotations in
            dataset samples. Defaults to 4 to be compatible with
            AnnotatedTopImagesDataset.
        indexer_kwargs (Optional[Mapping[str, Any]], optional): Indexer
            options. By default, indexer is configured to not ignore stop
            words and punctuation.
        word_featurizer_kwargs (Optional[Mapping[str, Any]], optional): Word
            featurizer options. Defaults to None.

    Raises:
        ValueError: If both `featurizer` and `annotator` are set.

    Returns:
        Decoder: The instantiated decoder.

    """
    if featurizer is None and annotator is None:
        raise ValueError('must set exactly one of featurizer and annotator')
    if indexer_kwargs is None:
        indexer_kwargs = {}
    if word_featurizer_kwargs is None:
        word_featurizer_kwargs = {}

    annotations = []
    for index in range(len(cast(Sized, dataset))):
        annotation = dataset[index][annotation_index]
        annotation = lang.join(annotation)
        annotations.append(annotation)

    indexer_kwargs = dict(indexer_kwargs)
    if 'tokenize' not in indexer_kwargs:
        copy = kwargs.get('copy', False)
        indexer_kwargs['tokenize'] = lang.tokenizer(lemmatize=copy,
                                                    ignore_stop=False,
                                                    ignore_punct=False)
    for key in ('start', 'stop', 'pad', 'unk'):
        indexer_kwargs.setdefault(key, True)
    indexer = lang.indexer(annotations, **indexer_kwargs)

    featurizer_v, featurizer_w = featurizer, None
    if annotator is not None:
        featurizer_w = WordFeaturizer(annotator, **word_featurizer_kwargs)

    return Decoder(indexer,
                   featurizer_v=featurizer_v,
                   featurizer_w=featurizer_w,
                   lm=lm,
                   **kwargs)
