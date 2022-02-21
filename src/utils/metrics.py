"""Utilities for computing standard performance metrics."""
import warnings
from typing import Mapping, Optional

from src.deps.ext import bert_score as bert_score_lib
from src.utils.typing import Device, StrSequence

import rouge as rouge_lib
import sacrebleu
from torch.utils import data

# TODO(evandez): Move accuracy fns here?
# TODO(evandez): Commonize string normalization.


def bleu(dataset: data.Dataset,
         predictions: StrSequence,
         annotation_index: int = 4) -> sacrebleu.BLEUScore:
    """Compute BLEU score of this model on the given dataset.

    Keyword arguments forwarded to `Decoder.predict` if `predictions` not
    provided.

    Args:
        dataset (data.Dataset): The test dataset.
        predictions (StrSequence): Predictions to score.
        annotation_index (int, optional): Index of language annotations in
            dataset samples. Defaults to 4 to be compatible with
            AnnotatedTopImagesDataset.

    Returns:
        sacrebleu.BLEUScore: Corpus BLEU score.

    """
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


def rouge(dataset: data.Dataset,
          predictions: StrSequence,
          annotation_index: int = 4) -> Mapping[str, Mapping[str, float]]:
    """Compute ROUGE score of this model on the given dataset.

    Keyword arguments forwarded to `Decoder.predict` if `predictions` not
    provided.

    Args:
        dataset (data.Dataset): The test dataset.
        predictions (StrSequence): Predictions to score.
        annotation_index (int, optional): Index of language annotations in
            dataset samples. Defaults to 4 to be compatible with
            AnnotatedTopImagesDataset.

    Returns:
        Mapping[str, Mapping[str, float]]: Average ROUGE (1, 2, l) scores.

    """
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

    scorer = rouge_lib.Rouge()
    return scorer.get_scores(hypotheses,
                             references,
                             avg=True,
                             ignore_empty=True)


def bert_score(
    dataset: data.Dataset,
    predictions: StrSequence,
    annotation_index: int = 4,
    batch_size: int = 16,
    device: Optional[Device] = None,
    bert_scorer: Optional[bert_score_lib.BERTScorer] = None,
) -> Mapping[str, float]:
    """Return average BERTScore P/R/F.

    Args:
        dataset (data.Dataset): The test dataset.
        predictions (StrSequence): Predictions to score.
        annotation_index (int, optional): Index of language annotations in
            dataset samples. Defaults to 4 to be compatible with
            AnnotatedTopImagesDataset.
        batch_size (int, optional): Batch size to use when computing
            BERTScore. Defaults to 16.
        predictions (Optional[StrSequence], optional): Precomputed
            predicted captions for all images in the dataset.
            By default, computed from the dataset using `Decoder.predict`.
        bert_scorer (Optional[bert_score.BERTScorer], optional): A
            pre-instantiated BERTScorer object. Defaults to None.
        device (Optional[Device], optional): Run BERT on this device.
            Defaults to torch default.

    Returns:
        Mapping[str, float]: Average BERTScore precision/recall/F1.

    """
    if bert_scorer is None:
        bert_scorer = bert_score_lib.BERTScorer(idf=True,
                                                lang='en',
                                                rescale_with_baseline=True,
                                                use_fast_tokenizer=True,
                                                device=device)

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
        key: scores.mean().item() for key, scores in zip(('p', 'r', 'f'), prf)
    }
