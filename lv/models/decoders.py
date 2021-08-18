"""Models for decoding neuron captions."""
import warnings
from typing import (Any, Dict, Mapping, NamedTuple, Optional, Sized, Tuple,
                    Type, Union, cast, overload)

from lv.models import encoders, lms
from lv.utils import lang, serialize, training
from lv.utils.typing import Device, StrSequence

import bert_score
import rouge
import sacrebleu
import torch
from torch import nn, optim
from torch.distributions import categorical
from torch.utils import data
from tqdm.auto import tqdm


class Attention(serialize.SerializableModule):
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

    def properties(self) -> serialize.Properties:
        """Override `Serializable.properties`."""
        return {
            'query_size': self.query_size,
            'key_size': self.key_size,
            'hidden_size': self.hidden_size,
        }


class DecoderState(NamedTuple):
    """Wrapper for all decoder state.

    Fields:
        h (torch.Tensor): The decoder LSTM's hidden state after stepping.
        c (torch.Tensor): The decoder LSTM's cell state after stepping.
        h_lm (Optional[torch.Tensor]): The LM LSTM's hidden state after
            stepping.
        c_lm (Optional[torch.Tensor]): The LM LSTM's cell state after stepping.

    """

    h: torch.Tensor
    c: torch.Tensor
    h_lm: Optional[torch.Tensor]
    c_lm: Optional[torch.Tensor]


class DecoderStep(NamedTuple):
    """Wraps the outputs of one decoding step.

    Fields:
        scores (torch.Tensor): Shape (batch_size, vocab_size) tensor containing
            log probabilities (if using likelihood decoding) or mutual info (if
            using MI decoding) for each word.
        attentions (torch.Tensor): Shape (batch_size, n_features) tensor
            containing attention weights for each visual feature.
        state (DecoderState): Hidden states used by the decoder.

    """

    scores: torch.Tensor
    attentions: torch.Tensor
    state: DecoderState


class DecoderOutput(NamedTuple):
    """Wraps output of the caption decoder.

    Fields:
        captions (StrSequence): Fully decoded captions for each sample.
        scores (torch.Tensor): Shape (batch_size, length, vocab_size) tensor
            containing log probabilities (if using likelihood decoding) or
            mutual info (if using MI decoding) for each word.
        tokens (torch.Tensor): Shape (batch_size, length) integer tensor
            containing IDs of decoded tokens.
        attentions (torch.Tensor): Shape (batch_size, length, n_features)
            containing attention weights for each time step.

    """

    captions: StrSequence
    scores: torch.Tensor
    tokens: torch.Tensor
    attentions: torch.Tensor


Strategy = Union[torch.Tensor, str]

STRATEGY_GREEDY = 'greedy'
STRATEGY_SAMPLE = 'sample'
STRATEGY_BEAM = 'beam'
STRATEGIES = (STRATEGY_GREEDY, STRATEGY_SAMPLE, STRATEGY_BEAM)


class Decoder(serialize.SerializableModule):
    """Neuron caption decoder.

    Roughly mimics the architecture described in Show, Attend, and Tell
    [Xu et al., 2015]. The main difference is in how the features are encoded
    and that the decoder can also decode using mutual information instead of
    likelihood.
    """

    def __init__(self,
                 indexer: lang.Indexer,
                 encoder: encoders.Encoder,
                 lm: Optional[lms.LanguageModel] = None,
                 embedding_size: int = 128,
                 hidden_size: int = 512,
                 attention_hidden_size: Optional[int] = None,
                 dropout: float = .5,
                 length: int = 15,
                 strategy: str = STRATEGY_GREEDY,
                 temperature: float = .3,
                 beam_size: int = 5):
        """Initialize the decoder.

        Args:
            indexer (lang.Indexer): Indexer for captions.
            encoder (encoders.Encoder], optional): Visual encoder.
            lm (Optional[lms.LanguageModel], optional): Language model. Changes
                decoding to use mutual info [p(caption | image) / p(caption)]
                instead of likelihood [p(caption | image)]. Defaults to None,
                meaning decoding will always use likelihood.
            embedding_size (int, optional): Size of previous-word embeddings
                that are input to the LSTM. Defaults to 128.
            hidden_size (int, optional): Size of LSTM hidden states.
                Defaults to 512.
            attention_hidden_size (Optional[int], optional): Size of attention
                mechanism hidden layer. Defaults to minimum of visual feature
                size and `hidden_size`.
            dropout (float, optional): Dropout probability, applied before
                output layer of LSTM. Defaults to .5.
            length (int, optional): Default decoding length. Defaults to 15.
            strategy (str, optional): Default decoding strategy. Note that
                force decoding is not supported as a default.
                Defaults to 'greedy'.
            temperature (float, optional): Default temperature parameter to use
                when MI decoding. When not MI decoding, this parameter does
                nothing. Defaults to .3.
            beam_size (int, optional): Default beam size for beam search
                decoding. When not decoding with beam search, this parameter
                does nothing. Defaults to 5.

        Raises:
            ValueError: If LM is set but has a different vocabulary than the
                given indexer.

        """
        super().__init__()

        if lm is not None:
            my_vocab = indexer.vocab.unique
            lm_vocab = lm.indexer.vocab.unique
            if my_vocab != lm_vocab:
                raise ValueError('lm and decoder have different vocabs;'
                                 f'lm missing {my_vocab - lm_vocab} and '
                                 f'decoder missing {lm_vocab - my_vocab}')

        self.indexer = indexer
        self.encoder = encoder
        self.lm = lm
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.dropout = dropout
        self.length = length
        self.strategy = strategy
        self.temperature = temperature
        self.beam_size = beam_size

        self.init_h = nn.Sequential(nn.Linear(self.feature_size, hidden_size),
                                    nn.Tanh())
        self.init_c = nn.Sequential(nn.Linear(self.feature_size, hidden_size),
                                    nn.Tanh())
        self.embedding = nn.Embedding(self.vocab_size, embedding_size)

        self.attend = Attention(hidden_size,
                                self.feature_size,
                                hidden_size=attention_hidden_size)
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_size, self.feature_size),
            nn.Sigmoid(),
        )

        self.lstm = nn.LSTMCell(embedding_size + self.feature_size,
                                hidden_size)

        self.output = nn.Sequential(nn.Dropout(p=dropout),
                                    nn.Linear(hidden_size, self.vocab_size),
                                    nn.LogSoftmax(dim=-1))

    @property
    def feature_size(self) -> int:
        """Return the visual feature size."""
        return self.encoder.feature_shape[-1]

    @property
    def vocab_size(self) -> int:
        """Return the vocab size."""
        return len(self.indexer)

    @overload
    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor,
                length: Optional[int] = ...,
                strategy: Strategy = ...,
                mi: Optional[bool] = ...,
                temperature: Optional[float] = ...,
                **kwargs: Any) -> DecoderOutput:
        """Decode captions for the given top images and masks.

        Args:
            images (torch.Tensor): Top-k images for a neuron.
                Should have shape (batch_size, k, 3, height, width).
            masks (torch.Tensor): Top-k image masks for a neuron.
                Should have shape (batch_size, k, 1, height, width).
            length (Optional[int], optional): Decode for this many steps.
                Defaults to `self.length`.
            strategy (Strategy, optional): Decoding strategy. If a tensor,
                values will be used as inputs at each time step, so it should
                have shape (batch_size, length). Other options include 'greedy'
                and 'sample'. Defaults to 'greedy'.
            mi (bool, optional): If True, use MI decoding. If False, use
                likelihood decoding. By default, MI decoding is used if the
                decoder has an LM and is itself not in training mode.
            temperature (Optional[float], optional): Temperature parameter for
                MI decoding. Does nothing if not MI decoding.
                Defaults to `self.temperature`.

        Returns:
            DecoderOutput: Decoder outputs.

        """
        ...

    @overload
    def forward(self, images: torch.Tensor, **kwargs: Any) -> DecoderOutput:
        """Decode captions for the given visual features.

        Same as the other overload, but inputs are visual features.
        """
        ...

    def forward(self,
                images: torch.Tensor,
                masks: Optional[torch.Tensor] = None,
                length: Optional[int] = None,
                strategy: Optional[Strategy] = None,
                mi: Optional[bool] = None,
                temperature: Optional[float] = None,
                beam_size: Optional[int] = None,
                **_: Any) -> DecoderOutput:
        """Implement both overloads."""
        if length is None:
            length = self.length
        if strategy is None:
            strategy = self.strategy
        if mi is None:
            mi = self.lm is not None and not self.training
        if beam_size is None:
            beam_size = self.beam_size

        # Validate arguments.
        if mi and self.lm is None:
            raise ValueError('cannot use MI decoding without an LM')
        if mi and self.training:
            raise ValueError('cannot use MI decoding while training')

        if isinstance(strategy, str) and strategy not in STRATEGIES:
            raise ValueError(f'unknown strategy: {strategy}')
        if isinstance(strategy, torch.Tensor):
            if strategy.dim() != 2:
                raise ValueError(f'strategy must be 2D, got {strategy.dim()}')
            if strategy.shape[-1] != length:
                raise ValueError(f'strategy must have length {length}, '
                                 f'got {strategy.shape[-1]}')

        batch_size = len(images)

        # If necessary, obtain visual features.
        if masks is not None:
            images = images.view(-1, 3, *images.shape[-2:])
            masks = masks.view(-1, 1, *masks.shape[-2:])
            with torch.no_grad():
                features = self.encoder(images, masks)
        else:
            features = images.view(batch_size, -1, self.feature_size)

        # Compute initial decoder state and initial inputs.
        state = self.init_state(features, lm=mi)
        currents = images.new_empty(batch_size, dtype=torch.long)
        currents.fill_(self.indexer.start_index)

        # Begin decoding. If we're not doing beam search, it's easy!
        if strategy != STRATEGY_BEAM:
            tokens = currents.new_zeros(batch_size, length)
            scores = images.new_zeros(batch_size, length, self.vocab_size)
            attentions = images.new_zeros(batch_size, length,
                                          features.shape[1])
            for time in range(length):
                step = self.step(features,
                                 currents,
                                 state,
                                 temperature=temperature)

                # Pick next token by applying the decoding strategy.
                if isinstance(strategy, torch.Tensor):
                    currents = strategy[:, time]
                elif strategy == STRATEGY_GREEDY:
                    currents = step.scores.argmax(dim=1)
                else:
                    assert strategy == STRATEGY_SAMPLE
                    for index, logprobs in enumerate(step.scores):
                        probs = torch.exp(logprobs)
                        distribution = categorical.Categorical(probs=probs)
                        currents[index] = distribution.sample()

                # Record step results.
                scores[:, time] = step.scores
                attentions[:, time] = step.attentions
                tokens[:, time] = currents
                state = step.state

        # Otherwise, if we're doing beam search, life is hard.
        else:
            tokens = currents.new_zeros(batch_size, beam_size, length)
            scores = images.new_zeros(batch_size, beam_size, length,
                                      self.vocab_size)
            attentions = images.new_zeros(batch_size, beam_size, length,
                                          features.shape[1])
            totals = images.new_zeros(batch_size, beam_size, 1)

            # Take the first step, setting up the beam.
            step = self.step(features,
                             currents,
                             state,
                             temperature=temperature)
            topk = step.scores.topk(k=beam_size, dim=-1)
            tokens[:, :, 0] = topk.indices
            scores[:, :, 0] = step.scores
            attentions[:, :, 0] = step.attentions
            totals[:] = topk.values.view(batch_size, beam_size, 1)

            # Adjust the features and state to have the right shape.
            features = features.repeat(beam_size, dim=0)
            state = DecoderState(*(  # type: ignore
                tensor.repeat_interleave(beam_size, dim=0)  # Beamify state.
                if tensor is not None else None for tensor in state))

            # Take the remaining steps.
            for time in range(1, length):
                currents = tokens[:, :, time - 1].view(-1)
                step = self.step(features,
                                 currents,
                                 state,
                                 temperature=temperature)

                # Determine which sequences (s) in the beam, and which next
                # tokens (t) for those sequences, will comprise the next beam.
                topk_t = step.scores.topk(k=beam_size, dim=-1)
                topk_s = totals\
                    .add(topk_t.values.view(batch_size, beam_size, beam_size))\
                    .view(batch_size, beam_size**2)\
                    .topk(k=beam_size, dim=-1)
                idx_s = (topk_s.indices // beam_size).view(-1)
                idx_t = (topk_s.indices % beam_size).view(-1)
                idx_b = torch.arange(batch_size).repeat_interleave(beam_size)

                # Update the beam. The fancy indexing here allows us to forgo
                # for loops, which generally impose a big performance penalty.
                tokens[:, :, time] = topk_t.indices\
                    .view(batch_size, beam_size, beam_size)[
                        idx_b, idx_s, idx_t]\
                    .view(batch_size, beam_size)
                scores[:, :, time] = step.scores\
                    .view(batch_size, beam_size, self.vocab_size)[
                        idx_b, idx_s]\
                    .view(batch_size, beam_size, self.vocab_size)
                attentions[:, :, time] = step.attentions\
                    .view(batch_size, beam_size, step.attentions.shape[-1])[
                        idx_b, idx_s]\
                    .view(batch_size, beam_size, step.attentions.shape[-1])
                totals[:] = topk_s.values.view(batch_size, beam_size, 1)

                # Don't forget to update RNN state as well!
                state = DecoderState(*(  # type: ignore
                    tensor.view(batch_size, beam_size, -1)[
                        idx_b, idx_s] if tensor is not None else None
                    for tensor in step.state))

            # Throw away everything but the best.
            tokens = tokens[:, 0].clone()
            scores = scores[:, 0].clone()
            attentions = attentions[:, 0].clone()

        return DecoderOutput(
            captions=self.indexer.reconstruct(tokens.tolist()),
            scores=scores,
            tokens=tokens,
            attentions=attentions,
        )

    def init_state(self,
                   features: torch.Tensor,
                   lm: bool = True) -> DecoderState:
        """Initialize decoder state for a fresh decoding.

        Args:
            features (torch.Tensor): Visualf features. Should have shape
                (batch_size, num_features, feature_size).
            lm (bool, optional): Initialize LM hidden state as well, if
                possible. Defaults to True.

        Returns:
            DecoderState: The freshly initialized decoder state.

        """
        # Compute initial hidden state and cell value.
        pooled = features.mean(dim=1)
        h, c = self.init_h(pooled), self.init_c(pooled)

        # If necessary, compute LM initial hidden state and cell value.
        h_lm, c_lm = None, None
        if self.lm is not None and lm:
            batch_size = len(features)
            h_lm = h.new_zeros(self.lm.layers, batch_size, self.lm.hidden_size)
            c_lm = c.new_zeros(self.lm.layers, batch_size, self.lm.hidden_size)

        return DecoderState(h, c, h_lm, c_lm)

    def step(self,
             features: torch.Tensor,
             tokens: torch.Tensor,
             state: DecoderState,
             temperature: Optional[float] = None) -> DecoderStep:
        """Take one decoding step.

        This does everything *except* choose the next token, which depends
        on the decoding strategy being used.

        Args:
            features (torch.Tensor): The visual features. Should have shape
                (batch_size, n_features, feature_size).
            tokens (torch.Tensor): The current token inputs for the LSTM.
                Should be an integer tensor of shape (batch_size,).
            state (DecoderState): The current decoder state.
            temperature (Optional[float], optional): Temperature to use when MI
                decoding. If not MI decoding, does nothing. Defaults to
                `self.temperature`.

        Raises:
            ValueError: If one of {h_lm, c_lm} is set but not the other, or
                if either is set but decoder has no lm.

        Returns:
            DecoderStep: Result of taking the step.

        """
        h, c, h_lm, c_lm = state
        if (h_lm is None) != (c_lm is None):
            raise ValueError('state must have both h_lm and c_lm or neither')
        if h_lm is not None and self.lm is None:
            raise ValueError('state has h_lm or c_lm, but decoder has no lm')
        temperature = self.temperature if temperature is None else temperature

        # Attend over visual features and gate them.
        attentions = self.attend(h, features)
        attenuated = attentions.unsqueeze(-1).mul(features).sum(dim=1)
        gate = self.feature_gate(h)
        gated = attenuated * gate

        # Prepare LSTM inputs and take a step.
        embeddings = self.embedding(tokens)
        inputs = torch.cat((embeddings, gated), dim=-1)
        h, c = self.lstm(inputs, (h, c))
        scores = log_p_w = self.output(h)

        # If MI decoding, convert likelihood into mutual information.
        if self.lm is not None and h_lm is not None and c_lm is not None:
            with torch.no_grad():
                inputs_lm = self.lm.embedding(tokens)[:, None]
                _, (h_lm, c_lm) = self.lm.lstm(inputs_lm, (h_lm, c_lm))
                assert h_lm is not None and c_lm is not None
                log_p_w_lm = self.lm.output(h_lm[-1])
            scores = log_p_w - temperature * log_p_w_lm

        return DecoderStep(scores=scores,
                           attentions=attentions,
                           state=DecoderState(h=h, c=c, h_lm=h_lm, c_lm=c_lm))

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

    def bert_score(self,
                   dataset: data.Dataset,
                   annotation_index: int = 4,
                   batch_size: int = 32,
                   predictions: Optional[StrSequence] = None,
                   device: Optional[Device] = None,
                   bert_scorer: Optional[bert_score.BERTScorer] = None,
                   **kwargs: Any) -> Mapping[str, float]:
        """Return average BERTScore P/R/F.

        Args:
            dataset (data.Dataset): The test dataset.
            annotation_index (int, optional): Index of language annotations in
                dataset samples. Defaults to 4 to be compatible with
                AnnotatedTopImagesDataset.
            batch_size (int, optional): Batch size to use when computing
                BERTScore. Defaults to 32.
            predictions (Optional[StrSequence], optional): Precomputed
                predicted captions for all images in the dataset.
                By default, computed from the dataset using `Decoder.predict`.
            bert_scorer (Optional[bert_score.BERTScorer], optional): Pre-
                instantiated BERTScorer object. Defaults to none.
            device (Optional[Device], optional): Run BERT on this device.
                Defaults to torch default.

        Returns:
            Mapping[str, float]: Average BERTScore precision/recall/F1.

        """
        if bert_scorer is None:
            bert_scorer = bert_score.BERTScorer(idf=True,
                                                lang='en',
                                                rescale_with_baseline=True,
                                                device=device)
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

        if bert_scorer.idf:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message=r'.*Overwriting.*')
                bert_scorer.compute_idf([r for rs in references for r in rs])

        prf = bert_scorer.score(predictions, references, batch_size=batch_size)
        return {
            key: scores.mean().item()
            for key, scores in zip(('p', 'r', 'f'), prf)
        }

    def predict(self,
                dataset: data.Dataset,
                mask: bool = True,
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
            mask (bool, optional): Use masks when computing features. Exact
                behavior depends on the featurizer. Defaults to True.
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
            features = self.encoder.map(dataset,
                                        mask=mask,
                                        image_index=image_index,
                                        mask_index=mask_index,
                                        batch_size=batch_size,
                                        device=device,
                                        display_progress_as=display_progress_as
                                        is not None)

        loader = data.DataLoader(features,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        if display_progress_as is not None:
            loader = tqdm(loader, desc=display_progress_as)

        outputs = []
        for (inputs,) in loader:
            with torch.no_grad():
                output = self(inputs.to(device), **kwargs)
            outputs.append(output)

        captions = []
        for output in outputs:
            captions += output.captions

        return tuple(captions)

    def fit(self,
            dataset: data.Dataset,
            mask: bool = True,
            image_index: int = 2,
            mask_index: int = 3,
            annotation_index: int = 4,
            batch_size: int = 64,
            max_epochs: int = 100,
            patience: int = 4,
            hold_out: float = .1,
            regularization_weight: float = 1.,
            optimizer_t: Type[optim.Optimizer] = optim.Adam,
            optimizer_kwargs: Optional[Mapping[str, Any]] = None,
            features: Optional[data.TensorDataset] = None,
            num_workers: int = 0,
            device: Optional[Device] = None,
            display_progress_as: Optional[str] = 'train decoder') -> None:
        """Train a new decoder on the given data.

        Args:
            dataset (data.Dataset): Dataset to train on.
            mask (bool, optional): Use masks when computing features. Exact
                behavior depends on the featurizer. Defaults to True.
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
            features = self.encoder.map(dataset,
                                        mask=mask,
                                        image_index=image_index,
                                        mask_index=mask_index,
                                        batch_size=batch_size,
                                        device=device,
                                        display_progress_as=display_progress_as
                                        is not None)

        # Prepare dataset and data loader. Use an anonymous dataset class to
        # make this easier, since we want to split train/val by neuron, but
        # iterate by annotation (for which there are ~3x per neuron).
        class WrapperDataset(data.Dataset):

            def __init__(self, subset: data.Subset):
                self.samples = []
                for index in subset.indices:
                    assert features is not None
                    feature, = features[index]

                    annotations = dataset[index][annotation_index]
                    if isinstance(annotations, str):
                        sample = (feature, annotations)
                        self.samples.append(sample)
                        continue

                    for annotation in annotations:
                        sample = (feature, annotation)
                        self.samples.append(sample)

            def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
                return self.samples[index]

            def __len__(self) -> int:
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
            self.encoder.eval()
            train_loss = 0.
            for features_v, captions in train_loader:
                targets = torch.tensor(self.indexer(captions),
                                       device=device)[:, 1:]
                _, length = targets.shape

                outputs: DecoderOutput = self(features_v.to(device),
                                              length=length,
                                              strategy=targets,
                                              mi=False)

                loss = criterion(outputs.scores.permute(0, 2, 1), targets)

                attentions = outputs.attentions
                regularizer = ((1 - attentions.sum(dim=1))**2).mean()
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
                    outputs = self(features_v,
                                   length=length,
                                   strategy=targets,
                                   mi=False)
                    loss = criterion(outputs.scores.permute(0, 2, 1), targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)

            if display_progress_as is not None:
                assert not isinstance(progress, range)
                progress.set_description(f'{display_progress_as} '
                                         f'[train_loss={train_loss:.3f}, '
                                         f'val_loss={val_loss:.3f}]')

            if stopper(val_loss):
                break

    def properties(self) -> serialize.Properties:
        """Override `Serializable.properties`."""
        return {
            'indexer': self.indexer,
            'encoder': self.encoder,
            'lm': self.lm,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'attention_hidden_size': self.attention_hidden_size,
            'dropout': self.dropout,
            'length': self.length,
            'strategy': self.strategy,
            'temperature': self.temperature,
            'beam_size': self.beam_size,
        }

    def serializable(self) -> serialize.Children:
        """Override `Serializable.serializable`."""
        serializable = {}
        if self.encoder is not None:
            serializable['encoder'] = encoders.key(self.encoder)
        return serializable

    @classmethod
    def resolve(cls, children: serialize.Children) -> serialize.Resolved:
        """Override `Serializable.resolve`."""
        resolved: Dict[str, Type[serialize.Serializable]]
        resolved = {
            'indexer': lang.Indexer,
            'lm': lms.LanguageModel,
        }

        encoder_key = children.get('encoder')
        if encoder_key is None:
            raise ValueError('serialized decoder missing encoder')
        resolved['encoder'] = encoders.parse(encoder_key)

        return resolved


def decoder(dataset: data.Dataset,
            encoder: encoders.Encoder,
            annotation_index: int = 4,
            indexer_kwargs: Optional[Mapping[str, Any]] = None,
            **kwargs: Any) -> Decoder:
    """Instantiate a new decoder.

    Args:
        dataset (data.Dataset): Dataset to draw vocabulary from.
        encoder (encoders.Encoder): Visual encoder.
        annotation_index (int, optional): Index of language annotations in
            dataset samples. Defaults to 4 to be compatible with
            AnnotatedTopImagesDataset.
        indexer_kwargs (Optional[Mapping[str, Any]], optional): Indexer
            options. By default, indexer is configured to not ignore stop
            words and punctuation.

    Returns:
        Decoder: The instantiated decoder.

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

    return Decoder(indexer, encoder, **kwargs)
