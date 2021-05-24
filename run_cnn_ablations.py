"""Run CNN ablation experiments."""
import argparse
import pathlib
import random

import lv.dissection.zoo
import lv.zoo
from lv import datasets
from lv.models import annotators, captioners, classifiers, featurizers
from lv.utils import logging, training
from lv.utils.typing import StrSequence

import nltk
import numpy as np
import spacy
import torch
import wandb
from nltk.corpus import wordnet
from spacy import language
from spacy_wordnet import wordnet_annotator

EXPERIMENT_RANDOM = 'random'
EXPERIMENT_N_OBJECT_WORDS = 'n-object-words'
EXPERIMENT_N_ABSTRACT_WORDS = 'n-abstract-words'
EXPERIMENT_N_SPATIAL_RELATIONS = 'n-spatial-relations'
EXPERIMENT_N_NOUNS = 'n-nouns'
EXPERIMENT_N_VERBS = 'n-verbs'
EXPERIMENT_N_ADPS = 'n-adps'
EXPERIMENT_N_ADJECTIVES = 'n-adjectives'
EXPERIMENT_CAPTION_LENGTH = 'caption-length'
EXPERIMENT_MAX_WORD_DIFFERENCE = 'max-word-difference'
EXPERIMENTS = (EXPERIMENT_RANDOM, EXPERIMENT_N_OBJECT_WORDS,
               EXPERIMENT_N_ABSTRACT_WORDS, EXPERIMENT_N_SPATIAL_RELATIONS,
               EXPERIMENT_N_NOUNS, EXPERIMENT_N_VERBS, EXPERIMENT_N_ADPS,
               EXPERIMENT_N_ADJECTIVES, EXPERIMENT_CAPTION_LENGTH,
               EXPERIMENT_MAX_WORD_DIFFERENCE)

ORDER_INCREASING = 'increasing'
ORDER_DECREASING = 'decreasing'
ORDERS = (ORDER_DECREASING, ORDER_INCREASING)

SPATIAL_RELATIONS = frozenset({
    'left',
    'right',
    'above',
    'under',
    'around',
    'center',
})

CNNS = (
    lv.zoo.KEY_ALEXNET,
    lv.zoo.KEY_RESNET152,
)
DATASETS = (
    lv.zoo.KEY_IMAGENET,
    # TODO(evandez): Figure out why this crashes.
    # lv.zoo.KEY_PLACES365,
)
TRAIN = {
    lv.zoo.KEY_ALEXNET: (
        lv.zoo.KEY_RESNET152_IMAGENET,
        lv.zoo.KEY_RESNET152_PLACES365,
        lv.zoo.KEY_BIGGAN_IMAGENET,
        lv.zoo.KEY_BIGGAN_PLACES365,
    ),
    lv.zoo.KEY_RESNET152: (
        lv.zoo.KEY_ALEXNET_IMAGENET,
        lv.zoo.KEY_ALEXNET_PLACES365,
        lv.zoo.KEY_BIGGAN_IMAGENET,
        lv.zoo.KEY_BIGGAN_PLACES365,
    ),
}

CAPTIONER_GT = 'gt'
CAPTIONER_SAT = 'sat'
CAPTIONER_SAT_MF = 'sat+mf'
CAPTIONER_SAT_WF = 'sat+wf'
CAPTIONER_SAT_MF_WF = 'sat+mf+wf'
CAPTIONERS = (CAPTIONER_GT, CAPTIONER_SAT, CAPTIONER_SAT_MF, CAPTIONER_SAT_WF,
              CAPTIONER_SAT_MF_WF)

parser = argparse.ArgumentParser(description='run cnn ablation experiments')
parser.add_argument('--cnns',
                    choices=CNNS,
                    default=CNNS,
                    help='cnns to ablate (default: alexnet, resnet152)')
parser.add_argument('--captioner',
                    choices=CAPTIONERS,
                    default=CAPTIONER_SAT_MF,
                    help='captioning model to use (default: sat+mf)')
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
parser.add_argument('--orders',
                    nargs='+',
                    choices=ORDERS,
                    default=ORDERS,
                    help='ablation orders to try (default: all)')
parser.add_argument('--datasets-root',
                    type=pathlib.Path,
                    default='.zoo/datasets',
                    help='root dir for datasets (default: .zoo/datasets)')
parser.add_argument('--ablation-min',
                    type=float,
                    default=0,
                    help='min fraction of neurons to ablate')
parser.add_argument('--ablation-max',
                    type=float,
                    default=.2,
                    help='max fraction of neurons to ablate')
parser.add_argument(
    '--ablation-step-size',
    type=float,
    default=.02,
    help='fraction of neurons to delete at each step (default: .05)')
parser.add_argument(
    '--n-random-trials',
    type=int,
    default=5,
    help='for each experiment, delete an equal number of random '
    'neurons and retest this many times (default: 5)')
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
           config={
               'captioner': args.captioner,
               'ablation_step_size': args.ablation_step_size,
               'n_random_trials': args.n_random_trials,
           },
           dir=args.wandb_dir)

device = 'cuda' if args.cuda else 'cpu'

nlp = spacy.load('en_core_web_lg')
object_synset, abstract_synset = None, None
if (EXPERIMENT_N_OBJECT_WORDS in args.experiments or
        EXPERIMENT_N_ABSTRACT_WORDS in args.experiments):
    nltk.download('wordnet', quiet=True)
    nltk.download('omw', quiet=True)

    # Redefine spacy factory.
    # TODO(evandez): Figure out a way to not do this.
    @language.Language.factory('spacy_wordnet', default_config={'lang': 'en'})
    def _(nlp, name, lang):
        return wordnet_annotator.WordnetAnnotator(lang=lang)

    nlp.add_pipe('spacy_wordnet', after='tagger')
    object_synset = wordnet.synset('object.n.01')
    abstract_synset = wordnet.synset('abstraction.n.01')

for dataset_name in args.datasets:
    dataset = lv.dissection.zoo.dataset(dataset_name,
                                        path=args.datasets_root /
                                        dataset_name / 'val',
                                        factory=training.PreloadedImageFolder)
    for cnn_name in args.cnns:
        cnn, *_ = lv.dissection.zoo.model(cnn_name, dataset_name)
        cnn = classifiers.ImageClassifier(cnn).to(device).eval()

        annotations = lv.zoo.datasets(f'{cnn_name}/{dataset_name}',
                                      path=args.datasets_root)
        assert isinstance(annotations, datasets.AnnotatedTopImagesDataset)

        # Obtain captions for every neuron in the CNN.
        if args.captioner == CAPTIONER_GT:
            captions: StrSequence = []
            for index in range(len(annotations)):
                caption = random.choice(annotations[index].annotations)
                assert isinstance(captions, list)
                captions.append(caption)
        else:
            train = lv.zoo.datasets(*TRAIN[cnn_name], path=args.datasets_root)

            if args.captioner != CAPTIONER_SAT:
                featurizer = featurizers.MaskedPyramidFeaturizer()
            else:
                featurizer = featurizers.MaskedImagePyramidFeaturizer()
            featurizer.to(device)

            features = None
            if args.captioner != CAPTIONER_SAT_WF:
                features = featurizer.map(train, device=device)

            annotator = None
            if args.captioner in (CAPTIONER_SAT_WF, CAPTIONER_SAT_MF_WF):
                annotator = annotators.WordAnnotator.fit(train,
                                                         featurizer=featurizer,
                                                         features=features,
                                                         device=device)

            if args.captioner in (CAPTIONER_SAT, CAPTIONER_SAT_MF):
                captioner = captioners.decoder(train, featurizer=featurizer)
            else:
                captioner = captioners.decoder(
                    train,
                    annotator=annotator,
                    featurizer=None
                    if args.captioner == CAPTIONER_SAT_WF else featurizer)
            captioner.fit(train, device=device)
            captioner.eval()
            captions = captioner.predict(annotations, device=device)

        # Pretokenize the captions for efficiency.
        tokenized = tuple(nlp.pipe(captions))

        # Begin the experiments! For each one, we will ablate neurons in
        # order of some criterion and measure drops in validation accuracy.
        for experiment in args.experiments:
            print('\n-------- BEGIN EXPERIMENT: '
                  f'{cnn_name}/{dataset_name}/{experiment} '
                  '--------')

            # When ablating random neurons, do it a few times to denoise.
            if experiment == EXPERIMENT_RANDOM:
                trials = args.n_random_trials
            else:
                trials = 1

            for trial in range(trials):
                if experiment == EXPERIMENT_RANDOM:
                    scores = torch.rand(len(captions)).tolist()
                elif experiment == EXPERIMENT_N_OBJECT_WORDS:
                    assert object_synset is not None
                    scores = [
                        sum(object_synset in synset.lowest_common_hypernyms(
                            object_synset)
                            for token in tokens
                            for synset in token._.wordnet.synsets())
                        for tokens in tokenized
                    ]
                elif experiment == EXPERIMENT_N_ABSTRACT_WORDS:
                    assert abstract_synset is not None
                    scores = [
                        sum(abstract_synset in synset.lowest_common_hypernyms(
                            abstract_synset)
                            for token in tokens
                            for synset in token._.wordnet.synsets())
                        for tokens in tokenized
                    ]
                elif experiment == EXPERIMENT_N_SPATIAL_RELATIONS:
                    scores = [
                        sum(token.lemma_.lower() in SPATIAL_RELATIONS
                            for token in tokens)
                        for tokens in tokenized
                    ]
                elif experiment == EXPERIMENT_N_NOUNS:
                    scores = [
                        sum(token.pos_ == 'NOUN'
                            for token in tokens)
                        for tokens in tokenized
                    ]
                elif experiment == EXPERIMENT_N_VERBS:
                    scores = [
                        sum(token.pos_ == 'VERB'
                            for token in tokens)
                        for tokens in tokenized
                    ]
                elif experiment == EXPERIMENT_N_ADPS:
                    scores = [
                        sum(token.pos_ == 'ADP'
                            for token in tokens)
                        for tokens in tokenized
                    ]
                elif experiment == EXPERIMENT_N_ADJECTIVES:
                    scores = [
                        sum(token.pos_ == 'ADJ'
                            for token in tokens)
                        for tokens in tokenized
                    ]
                elif experiment == EXPERIMENT_CAPTION_LENGTH:
                    scores = [len(tokens) for tokens in tokenized]
                else:
                    assert experiment == EXPERIMENT_MAX_WORD_DIFFERENCE
                    scores = []
                    for index, tokens in enumerate(tokenized):
                        vectors = torch.stack([
                            torch.from_numpy(token.vector) for token in tokens
                        ])
                        distances = vectors[:, None] - vectors[None, :]
                        distances = (distances**2).sum(dim=-1)
                        score = distances.max().item()
                        scores.append(score)

                for order in args.orders:
                    indices = sorted(range(len(captions)),
                                     key=lambda i: scores[i],
                                     reverse=order == ORDER_DECREASING)
                    fractions = np.arange(args.ablation_min, args.ablation_max,
                                          args.ablation_step_size)
                    for fraction in fractions:
                        ablated = indices[:int(fraction * len(indices))]
                        accuracy = cnn.accuracy(
                            dataset,
                            ablate=annotations.units(ablated),
                            display_progress_as='test ablated cnn '
                            f'(cond={experiment}, '
                            f'trial={trial + 1}, '
                            f'order={order}, '
                            f'frac={fraction:.2f})',
                            device=device)
                        samples = logging.random_neuron_wandb_images(
                            annotations,
                            captions,
                            indices=ablated,
                            k=args.wandb_n_samples,
                            cnn=cnn_name,
                            data=dataset_name,
                            exp=experiment,
                            order=order,
                            frac=fraction)
                        wandb.log({
                            'cnn': cnn_name,
                            'dataset': dataset_name,
                            'experiment': experiment,
                            'trial': trial + 1,
                            'order': order,
                            'frac_ablated': fraction,
                            'n_ablated': len(ablated),
                            'accuracy': accuracy,
                            'samples': samples,
                        })
