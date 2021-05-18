"""Run CNN editing experiments."""
import argparse
import collections
import pathlib
import random
from typing import Callable, Optional, Sequence, Sized, Union, cast

import lv.dissection.zoo
import lv.zoo
from lv import datasets
from lv.models import captioners, featurizers
from lv.utils.typing import Device, StrSequence
from third_party.netdissect import nethook

import nltk
import spacy
import torch
import wandb
from nltk.corpus import wordnet
from spacy import language
from spacy_wordnet import wordnet_annotator
from torch import nn
from torch.utils import data
from tqdm.auto import tqdm


def zero(units: Sequence[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Zero the given units.

    Args:
        units (Sequence[int]): The units to zero.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Function that takes layer
            features and zeros the given units, returning the result.

    """

    def fn(features: torch.Tensor) -> torch.Tensor:
        assert features.dim() == 4
        features[:, units] = 0
        return features

    return fn


AnyTopImagesDataset = Union[datasets.TopImagesDataset,
                            datasets.AnnotatedTopImagesDataset]


def ablate_and_test(model: nn.Module,
                    dataset: data.Dataset,
                    dissected: AnyTopImagesDataset,
                    indices: Sequence[int],
                    batch_size: int = 64,
                    num_workers: int = 0,
                    display_progress_as: str = 'ablate and test',
                    device: Optional[Device] = None) -> float:
    """Ablate the given neurons and test the model on the given dataset.

    Args:
        model (nn.Module): The model, which should take images as input.
        dataset (data.Dataset): The test dataset.
        dissected (AnyTopImagesDataset): Dissection results for the model.
        indices (Sequence[int]): The indices of neurons to ablate.
        batch_size (int, optional): Batch size for evaluation.
            Defaults to 64.
        num_workers (int, optional): Number of workers for loading test data.
            Defaults to 0.
        display_progress_as (str, optional): String to show on progress bar.
            Defaults to 'ablate and test'.
        device (Optional[Device], optional): Send images to this device.
            Defaults to None.

    Returns:
        float: Test accuracy.

    """
    neurons = [
        (dissected[index].layer, dissected[index].unit) for index in indices
    ]

    edits = collections.defaultdict(list)
    for layer, unit in neurons:
        edits[layer].append(unit)

    with nethook.InstrumentedModel(model) as instrumented:
        for layer, units in edits.items():
            instrumented.edit_layer(layer, rule=zero(sorted(units)))

        correct = 0
        loader = data.DataLoader(dataset, batch_size=batch_size)
        for batch in tqdm(loader, desc=display_progress_as):
            assert len(batch) == 2, 'weird dataset batch?'
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                predictions = instrumented(images)
            correct += predictions.argmax(dim=-1).eq(targets).sum().item()
    return correct / len(cast(Sized, dataset))


def create_wandb_images(dissected: AnyTopImagesDataset,
                        captions: StrSequence,
                        indices: Sequence[int],
                        condition: str = '?',
                        k: int = 25) -> Sequence[wandb.Image]:
    """Create some image samples of ablated neurons for wandb.

    Args:
        dissected (AnyTopImagesDataset): Top images dataset for neurons.
        captions (StrSequence): Captions for each neuron.
        indices (Sequence[int]): Indices of samples to choose from.
        k (int, optional): Max samples to draw. Defaults to 25.

    Returns:
        Sequence[wandb.Image]: Wandb images to upload.

    """
    chosen = random.sample(indices, k=min(len(indices), k))
    images = []
    for index in chosen:
        sample = dissected[index]
        image = wandb.Image(
            sample.as_pil_image_grid(),
            caption=f'(cond={condition}, l={sample.layer}, u={sample.unit}) '
            f'{captions[index]}')
        images.append(image)
    return images


EXPERIMENT_OBJECT_WORDS = 'object-words'
EXPERIMENT_SPATIAL_RELATION = 'spatial-relation'
EXPERIMENT_MANY_WORDS = 'many-words'
EXPERIMENT_LARGE_EMBEDDING_DIFFERENCE = 'experiment-large-embedding-difference'
EXPERIMENTS = (EXPERIMENT_OBJECT_WORDS, EXPERIMENT_SPATIAL_RELATION,
               EXPERIMENT_MANY_WORDS, EXPERIMENT_LARGE_EMBEDDING_DIFFERENCE)

MODELS = (lv.zoo.KEY_ALEXNET, lv.zoo.KEY_RESNET152)
DATASETS = (lv.zoo.KEY_IMAGENET, lv.zoo.KEY_PLACES365)
TRAIN = {
    lv.zoo.KEY_ALEXNET: (
        # TOOD(evandez): Uncomment once ready.
        lv.zoo.KEY_RESNET152_IMAGENET,
        # lv.zoo.KEY_RESNET152_PLACES365,
        # lv.zoo.KEY_BIGGAN_IMAGENET,
        # lv.zoo.KEY_BIGGAN_PLACES365,
    ),
    lv.zoo.KEY_RESNET152: (
        # TOOD(evandez): Uncomment once ready.
        lv.zoo.KEY_ALEXNET_IMAGENET,
        # lv.zoo.KEY_ALEXNET_PLACES365,
        # lv.zoo.KEY_BIGGAN_IMAGENET,
        # lv.zoo.KEY_BIGGAN_PLACES365,
    ),
}


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(description='run cnn editing experiments')
    parser.add_argument('--models',
                        choices=MODELS,
                        default=MODELS,
                        help='models to ablate (default: alexnet, resnet152)')
    parser.add_argument(
        '--datasets',
        choices=DATASETS,
        default=DATASETS,
        help='dataset model(s) trained on (default: imagenet, places365)')
    parser.add_argument('--experiments',
                        nargs='+',
                        choices=EXPERIMENTS,
                        default=EXPERIMENTS,
                        help='experiments to run (default: all)')
    parser.add_argument('--datasets-root',
                        type=pathlib.Path,
                        default='.zoo/datasets',
                        help='root dir for datasets (default: .zoo/datasets)')
    parser.add_argument('--max-ablation',
                        type=float,
                        default=.1,
                        help='max fraction of neurons to ablate (default: .1)')
    parser.add_argument(
        '--n-random-trials',
        type=int,
        default=5,
        help='for each experiment, delete an equal number of random '
        'neurons and retest this many times (default: 5)')
    parser.add_argument('--ground-truth',
                        action='store_true',
                        help='use ground truth captions')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='number of worker threads to load data with (default: 16)')
    parser.add_argument('--cuda', action='store_true', help='use cuda device')
    parser.add_argument('--wandb-project',
                        default='lv',
                        help='wandb project name (default: lv)')
    parser.add_argument('--wandb-name',
                        default='cnn-ablations',
                        help='wandb run name (default: cnn-ablations)')
    parser.add_argument('--wandb-group',
                        default='applications',
                        help='wandb group name (default: applications)')
    parser.add_argument('--wandb-entity', help='wandb user or team')
    parser.add_argument('--wandb-dir', metavar='PATH', help='wandb directory')
    parser.add_argument('--wandb-n-samples',
                        type=int,
                        default=25,
                        help='number of samples to upload for each model')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project,
               name=args.wandb_name,
               entity=args.wandb_entity,
               group=args.wandb_group,
               dir=args.wandb_dir)

    device = 'cuda' if args.cuda else 'cpu'

    nlp = spacy.load('en_core_web_lg')
    object_synset = None
    if EXPERIMENT_OBJECT_WORDS in args.experiments:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw', quiet=True)

        # Redefine spacy factory.
        # TODO(evandez): Figure out a way to not do this.
        @language.Language.factory('spacy_wordnet',
                                   default_config={'lang': 'en'})
        def _(nlp, name, lang):
            return wordnet_annotator.WordnetAnnotator(lang=lang)

        nlp.add_pipe('spacy_wordnet', after='tagger')
        object_synset = wordnet.synset('object.n.01')

    for dataset_name in args.datasets:
        dataset = lv.dissection.zoo.dataset(dataset_name,
                                            path=args.datasets_root /
                                            dataset_name / 'val')
        for model_name in args.models:
            model, *_ = lv.dissection.zoo.model(model_name, dataset_name)
            model.eval()

            annotations = lv.zoo.datasets(f'{model_name}/{dataset_name}',
                                          path=args.datasets_root)
            assert isinstance(annotations, datasets.AnnotatedTopImagesDataset)
            ablatable = int(args.max_ablation * len(annotations))

            if args.ground_truth:
                captions: StrSequence = []
                for index in range(len(cast(Sized, annotations))):
                    caption = random.choice(annotations[index].annotations)
                    assert isinstance(captions, list)
                    captions.append(caption)
            else:
                train = lv.zoo.datasets(*TRAIN[model_name],
                                        path=args.datasets_root)
                featurizer = featurizers.MaskedPyramidFeaturizer().to(device)
                captioner = captioners.decoder(train, featurizer=featurizer)
                captioner.fit(train, device=device)
                captioner.eval()
                captions = captioner.predict(annotations, device=device)
            tokenized = tuple(nlp.pipe(captions))

            for experiment in args.experiments:
                print('\n-------- BEGIN EXPERIMENT: '
                      f'{model_name}/{dataset_name}/{experiment} '
                      '--------')
                if experiment == EXPERIMENT_OBJECT_WORDS:
                    assert object_synset is not None
                    indices = [
                        index for index, tokens in enumerate(tokenized)
                        if any(object_synset in synset.lowest_common_hypernyms(
                            object_synset)
                               for token in tokens
                               for synset in token.wordnet.synsets())
                    ]
                elif experiment == EXPERIMENT_SPATIAL_RELATION:
                    indices = [
                        index for index, tokens in enumerate(tokenized)
                        if any(token.lemma_.lower() in
                               {'left', 'right', 'above', 'under', 'around'}
                               for token in tokens)
                    ]
                elif experiment == EXPERIMENT_MANY_WORDS:
                    indices = sorted(range(len(captions)),
                                     key=lambda index: len(tokenized[index]),
                                     reverse=True)
                    indices = indices[:ablatable]
                else:
                    assert experiment == EXPERIMENT_LARGE_EMBEDDING_DIFFERENCE
                    scores = torch.zeros(len(captions))
                    for index, tokens in enumerate(tokenized):
                        vectors = torch.stack([
                            torch.from_numpy(token.vector) for token in tokens
                        ])
                        distances = vectors[:, None] - vectors[None, :]
                        distances = (distances**2).sum(dim=-1)
                        scores[index] = distances.max()
                    indices = scores.topk(k=ablatable).indices.tolist()

                if len(indices) > ablatable:
                    indices = random.sample(indices, k=ablatable)

                accuracy = ablate_and_test(
                    model,
                    dataset,
                    annotations,
                    indices,
                    num_workers=args.num_workers,
                    display_progress_as=f'ablate {experiment}',
                    device=device)
                key = f'{experiment}/{model_name}/{dataset_name}'
                samples = create_wandb_images(annotations,
                                              captions,
                                              indices,
                                              condition=f'{key}/text',
                                              k=args.wandb_n_samples)
                wandb.log({
                    'model': model_name,
                    'dataset': dataset_name,
                    'experiment': experiment,
                    'condition': 'text',
                    'trial': 1,
                    'neurons': len(indices),
                    'accuracy': accuracy,
                    'samples': samples,
                })

                for trial in range(args.n_random_trials):
                    indices = random.sample(range(len(annotations)),
                                            k=len(indices))
                    accuracy = ablate_and_test(
                        model,
                        dataset,
                        annotations,
                        indices,
                        num_workers=args.num_workers,
                        display_progress_as=f'ablate random (t={trial + 1})',
                        device=device)
                    samples = create_wandb_images(annotations,
                                                  captions,
                                                  indices,
                                                  condition=f'{key}/random',
                                                  k=args.wandb_n_samples)
                    wandb.log({
                        'model': model_name,
                        'dataset': dataset_name,
                        'experiment': experiment,
                        'condition': 'random',
                        'trial': trial + 1,
                        'neurons': len(indices),
                        'accuracy': accuracy,
                        'samples': samples,
                    })


# Guard this script behind whether it's invoked, since it is imported.
# TODO(evandez): Ew, don't do this.
if __name__ == '__main__':
    main()
