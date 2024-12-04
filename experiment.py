# Import packages
import numpy as np
from skimage.segmentation import slic
from skimage.filters import gaussian
import torch
import torchvision
import torchvision.transforms as v1
from torch.utils.data import DataLoader, Subset
import os
import csv
from datetime import datetime
import argparse
import itertools
import random

# Import samplers
from post_hoc.samplers import (
    RandomSampler, SingleFeatureSampler, ShapSampler, SampleProbabilitySampler
)
# Import explainers
from post_hoc.explainers import (
    OriginalCIUAttributer, CIUAttributer, SHAPAttributer, RISEAttributer,
    LinearLIMEAttributer, PDAAttributer, CIUPlusAttributer1, ExplainerAttributer
)
# Import segmenters
from post_hoc.image_segmenters import (
    GridSegmenter, WrapperSegmenter, FadeMaskSegmenter
)
# Import segmentation and perturbation utils
from post_hoc.image_segmenters import perturbation_masks
# Import image perturbers
from post_hoc.image_perturbers import (
    SingleColorPerturber, ReplaceImagePerturber, TransformPerturber,
    RandomColorPerturber, ColorHistogramPerturber, Cv2InpaintPerturber
)
# Import connectors
from post_hoc.connectors import SegmentationPredictionPipeline, SegmentationAttribuitionPipeline
# Import PyTorch utils
from post_hoc.torch_utils import TorchModelWrapper, ImageToTorch
# Import explanation evaluators
from post_hoc.evaluation import ImageAUCEvaluator
# Import dataset handler
from dataset_collector import dataset_collector

# Set logging folder
from workspace_path import home_path
log_dir = home_path / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

def main():
    '''
    Collects parameters from when the file is run and evaluates all
    combinations of those parameters on the given benchmark.
    '''
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evaluations', type=str, nargs='+', default=['auc'], choices=['auc'],
        help='The evaluations to perform (auc: LIF,MIF,SRG)'
    )
    parser.add_argument(
        '--dataset', type=str, default='imagenet', choices=['imagenet'],
        help='The dataset to evaluate on (imagenet: ImageNet validation set)'
    )
    parser.add_argument(
        '--nets', type=str, nargs='+', default=['alexnet'],
        choices=['alexnet','vgg16','resnet50'],
        help='Which models to use for evaluation'
    )
    parser.add_argument(
        '--softmax', action='store_true',
        help='Apply softmax to model output to normalize it'
    )
    parser.add_argument(
        '--segmenters', type=str, nargs='+', default=['grid'],
        choices=['grid', 'slic'], help='Which segmentation methods to use'
    )
    parser.add_argument(
        '--n_segments', type=int, nargs='+', default=[49], metavar='Nr',
        help='The approximate number of segments to divide the images into'
    )
    parser.add_argument(
        '--samplers', type=str, nargs='+', default=['single_feature'],
        choices=[
            'single_feature', 'random', 'shap', 'normal', 'inverse_feature'
        ],
        help='The methods to use for sampling the segments to be perturbed'
    )
    parser.add_argument(
        '--blur', action='store_true',
        help='Use Gaussian filter to blur the perturbation masks'
    )
    parser.add_argument(
        '--perturbers', type=str, nargs='+', default=['black'],
        choices=[
            'black', 'gray', 'norm_zero', 'mean', 'histogram', 'blur', 'cv2'
        ],
        help='The methods to use for perturbing the images'
    )
    parser.add_argument(
        '--explainers', type=str, nargs='+', default=['rise'],
        #choices=['rise', 'ciu', 'lime', 'shap', 'pda', 'inverse_ciu'],
        help='Which methods to use to calculate attribution of the segments'
    )
    parser.add_argument(
        '--sample_sizes', type=int, nargs='+', default=[None], metavar='Nr',
        help='The sample sizes (if applicable) used when sampling'
    )
    parser.add_argument(
        '--attribution_types', type=str, nargs='+', default=['segments'],
        choices=['segments', 'pixels'],
        help='How to calculate the per pixel attribution'
    )
    parser.add_argument(
        '--auc_samples', type=int, default=10, metavar='Nr',
        help='The number of samples used in auc evaluation'
    )
    parser.add_argument(
        '--use_true_class', action='store_true',
        help='Use actual class instead of predicted class for auc evaluation'
    )
    parser.add_argument(
        '--per_input', action='store_true',
        help='Logs the results of each image instead of averaging over dataset'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, metavar='Nr',
        help='Maximum number of inputs to feed through a model at once'
    )
    parser.add_argument(
        '--fraction', type=float, default=1, metavar='X',
        help='What fraction of the dataset to use in the evaluation'
    )
    parser.add_argument(
        '--versions', type=int, default=1, metavar='Nr',
        help='How many times to run non-deteministic evaluations'
    )
    parser.add_argument(
        '--compile', action='store_true',
        help='Use torch.compile() to attempt to speed up evaluation'
    )
    parser.add_argument(
        '--segmenter_kw', type=str, nargs='*', metavar='key=value', default=[],
        help='Additional arguments for the segmenters formated as key=value'
    )
    parser.add_argument(
        '--sampler_kw', type=str, nargs='*', metavar='key=value', default=[],
        help='Additional arguments for the samplers formated as key=value'
    )
    parser.add_argument(
        '--perturber_kw', type=str, nargs='*', metavar='key=value', default=[],
        help='Additional arguments for the perturbers formated as key=value'
    )
    parser.add_argument(
        '--explainer_kw', type=str, nargs='*', metavar='key=value', default=[],
        help='Additional arguments for the explainers formated as key=value'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Prevent printing by the evaluation'
    )
    parser.add_argument(
        '--run_rise_plus', action='store_true',
        help='Ignore (most) arguments to run predefined rise+ experiments'
    )
    args = parser.parse_args()

    # Parse any additional key=value arguments
    segmenter_kw, sampler_kw, perturber_kw, explainer_kw = {}, {}, {}, {}
    kw_dicts = [segmenter_kw, sampler_kw, perturber_kw, explainer_kw]
    for kw_dict, kw_strs in zip((kw_dicts) ,[
        args.segmenter_kw,args.sampler_kw,args.perturber_kw,args.explainer_kw
    ]):
        for key_value in kw_strs:
            key, value = key_value.split('=')
            value = cancast(value)
            kw_dict[key] = value

    if args.run_rise_plus:
        rise_plus_dict = {
            'evaluators': ['auc'], 'net':None, 'segmenter':'grid', 'n_seg':49,
            'sampler':'random', 'perturber':'norm_zero', 'sample_size':None,
            'softmax':True, 'blur':False,
            'explainers':['rise', 'lime', 'shap', 'pda', 'ciu', 'inverse_ciu'],
            'attribution_types':['segments', 'pixels'], 'fraction':1,
            'segmenter_kw':{'bilinear':True}, 'sampler_kw':{},
            'perturber_kw':{}, 'explainer_kw':{},
        }
        net_dicts = [
            {'net':'alexnet', 'sample_size':4000},
            {'net':'vgg16', 'sample_size':4000},
            {'net':'resnet50', 'sample_size':8000}
        ]

        experiment_dicts = [
            # Test RISE setup with different samples sizes and samplers
            {'fraction':0.02},
            {'sample_size':None, 'fraction':0.02},
            {'sample_size':400, 'fraction':0.02},
            {'fraction':0.02, 'sampler':'shap'},
            {'fraction':0.02, 'sampler':'shap', 'sample_size':None},
            {'fraction':0.02, 'sampler':'shap', 'sample_size':400},
            {
                'fraction':0.02, 'sampler':'single_feature',
                'sample_size':None, 'sampler_kw':{'add_none':True}
            },
            {
                'fraction':0.02, 'sampler':'inverse_feature',
                'sample_size':None, 'sampler_kw':{'add_none':True}
            },
            # Same experiments but use gaussian blur instead of bilinear
            {'fraction':0.02, 'blur':True, 'segmenter_kw':{}},
            {
                'sample_size':None, 'fraction':0.02, 'blur':True,
                'segmenter_kw':{}
            },
            {
                'sample_size':400, 'fraction':0.02, 'blur':True,
                'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'shap', 'blur':True,
                'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'shap', 'sample_size':None,
                'blur':True, 'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'shap', 'sample_size':400,
                'blur':True, 'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'single_feature',
                'sample_size':None, 'sampler_kw':{'add_none':True},
                'blur':True, 'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'inverse_feature',
                'sample_size':None, 'sampler_kw':{'add_none':True},
                'blur':True, 'segmenter_kw':{}
            },
            # Same experiments but use gaussian blur and slic
            {
                'fraction':0.02, 'blur':True, 'segmenter':'slic',
                'segmenter_kw':{}
            },
            {
                'sample_size':None, 'fraction':0.02, 'blur':True,
                'segmenter':'slic', 'segmenter_kw':{}
            },
            {
                'sample_size':400, 'fraction':0.02, 'blur':True,
                'segmenter':'slic', 'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'shap', 'blur':True,
                'segmenter':'slic', 'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'shap', 'sample_size':None,
                'blur':True, 'segmenter':'slic', 'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'shap', 'sample_size':400,
                'blur':True, 'segmenter':'slic', 'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'single_feature',
                'sample_size':None, 'sampler_kw':{'add_none':True},
                'blur':True, 'segmenter':'slic', 'segmenter_kw':{}
            },
            {
                'fraction':0.02, 'sampler':'inverse_feature',
                'sample_size':None, 'sampler_kw':{'add_none':True},
                'blur':True, 'segmenter':'slic', 'segmenter_kw':{}
            },
            # Run the original experiment once with full validation set
            {}
        ]
        runs = [
            rise_plus_dict | net_dict | run_dict for run_dict, net_dict in
            itertools.product(experiment_dicts, net_dicts)
        ]
        keys = [
            'evaluators', 'net', 'segmenter', 'n_seg', 'sampler', 'perturber',
            'sample_size', 'softmax', 'blur', 'explainers', 'attribution_types',
            'fraction', 'segmenter_kw', 'sampler_kw', 'perturber_kw',
            'explainer_kw'
        ]
        runs = [[d[x] for x in keys] for d in runs]
    else:
        runs = itertools.product(
            [args.evaluations], args.nets, args.segmenters, args.n_segments,
            args.samplers, args.perturbers, args.sample_sizes, [args.softmax],
            [args.blur], [args.explainers], [args.attribution_types],
            [args.fraction], [segmenter_kw], [sampler_kw], [perturber_kw],
            [explainer_kw]
        )

    # For each combination of net, segmenter, n_seg, perturber, and sample_size
    for (
        evaluations, net, segmenter, n_seg, sampler, perturber, sample_size,
        softmax, blur, explainers, attribution_types, fraction, segmenter_kw,
        sampler_kw, perturber_kw, explainer_kw
    ) in runs:
        print(evaluations, net, segmenter, n_seg, sampler, perturber,
            sample_size, softmax, blur, explainers, attribution_types, fraction,
            segmenter_kw, sampler_kw, perturber_kw, explainer_kw
        )

        # Get model and weights
        weights = 'IMAGENET1K_V2' if net == 'resnet50' else 'IMAGENET1K_V1'
        net = torchvision.models.__dict__[net](weights=weights)
        net.eval()

        # Create a transform to prepare images for Torchvision
        transforms = v1.Compose([
            ImageToTorch(),
            v1.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            v1.Resize((224,224), antialias=True)
        ])

        # Wrap model and check if it can run on the GPU
        gpu = torch.cuda.is_available()
        model = TorchModelWrapper(
            net, transforms, gpu=gpu, softmax_out=softmax
        )

        # Prepare the dataset
        if args.dataset == 'imagenet':
            dataset_transform = v1.Compose([
                v1.ToTensor(),
                v1.Resize((224,224), antialias=True)
            ])
            dataset = dataset_collector(
                'IMAGENET1K2012', split='val', download=False,
                transform=dataset_transform
            )
            dataset_name = 'imagenet_val'
            image_idx=0
            label_idx=1

        # Create the segmenter
        hw = round(np.sqrt(n_seg))
        if segmenter == 'grid':
            segmenter = GridSegmenter(
                hw, hw, segmenter_kw.get('bilinear', False)
            )
        elif segmenter == 'slic':
            segmenter = WrapperSegmenter(
                slic, n_segments=n_seg, start_label=0, **segmenter_kw
            )

        if blur:
            sigma = 70/hw # To get similar blur to bilinear in GridSegmenter
            segmenter = FadeMaskSegmenter(segmenter, sigma=sigma)

        # Create the sampler
        if sampler == 'single_feature':
            sampler = SingleFeatureSampler(**sampler_kw)
            sample_size = None # This sampler has a fixed sample size
        elif sampler == 'random':
            sampler = RandomSampler(sampler_kw.get('p', 0.5))
        elif sampler == 'shap':
            sampler = ShapSampler(sampler_kw.get('ignore_warnings', True))
        elif sampler == 'normal':
            sampler = SampleProbabilitySampler(
                distribution='normal', **sampler_kw
            )
        elif sampler == 'inverse_feature':
            sampler = SingleFeatureSampler(inverse=True, **sampler_kw)
            sample_size = None # This sampler has a fixed sample size

        # Create the perturber
        if perturber == 'black':
            perturber = SingleColorPerturber((0,0,0))
        elif perturber == 'gray':
            perturber = SingleColorPerturber((0.75,0.75,0.75))
        elif perturber == 'norm_zero':
            perturber = SingleColorPerturber((0.485, 0.456, 0.406))
        elif perturber == 'mean':
            perturber = SingleColorPerturber('mean')
        elif perturber == 'histogram':
            perturber = ColorHistogramPerturber(**perturber_kw)
        elif perturber == 'blur':
            perturber = TransformPerturber(gaussian, **perturber_kw)
        elif perturber == 'cv2':
            perturber = Cv2InpaintPerturber(**perturber_kw)

        # Create the explainers
        new_explainers = []
        if 'rise' in explainers:
            new_explainers.append(RISEAttributer())
        if 'ciu' in explainers:
            new_explainers.append(OriginalCIUAttributer(**explainer_kw))
        if 'lime' in explainers:
            new_explainers.append(LinearLIMEAttributer())
        if 'shap' in explainers:
            new_explainers.append(SHAPAttributer())
        if 'pda' in explainers:
            new_explainers.append(PDAAttributer(**explainer_kw))
        if 'inverse_ciu' in explainers:
            new_explainers.append(OriginalCIUAttributer(
                inverse=True, **explainer_kw
            ))
        #### TESTING BELOW ####
        if 'ciu_plus' in explainers:
            new_explainers.append(CIUPlusAttributer1())
        if 'ciu_shap' in explainers:
            new_explainers.append(
                ExplainerAttributer(CIUAttributer(), SHAPAttributer())
            )
        #### TESTING ABOVE ####
        explainers = new_explainers

        # Do not use both attribution types if they are equivalent
        if not (blur or segmenter_kw.get('bilinear', False)):
            attribution_types = ['segments']

        # Create the evaluators
        evaluators = []
        if 'auc' in evaluations:
            evaluators.append(ImageAUCEvaluator(
                mode='srg', perturber=SingleColorPerturber(color='mean'),
                normalize=True
            ))

        pipeline = SegmentationAttribuitionPipeline(
            segmenter, sampler, perturber, explainers,
            explanans=['segment_map', 'pixel_map'], prune=True, 
            batch_size=args.batch_size
        )

        experiment = run_evaluation
        if args.compile:
            os.environ['TORCHDYNAMO_VERBOSE'] = '1'
            torch._dynamo.config.suppress_errors = True
            experiment = torch.compile(run_evaluation)

        for version in range(args.versions):
            # Only need to run non-deterministic experiments once
            if version>0 and sampler.deterministic and perturber.deterministic:
                continue
            experiment(
                pipeline, dataset, dataset_name, model, evaluators,
                sample_size=sample_size, attribution_types=attribution_types,
                explain='label' if args.use_true_class else 'top_class',
                fraction=fraction, image_idx=image_idx, label_idx=label_idx,
                per_input=args.per_input, version=version, num_workers=10,
                compile=args.compile, verbose=not args.quiet
            )


def cancast(val):
    '''
    Casts numbers as floats and bool strings as bools, else returns input as is
    Args:
        val (any): Input to parse
    Returns (any): The input cast to float, bool, or without change.
    '''
    try:
        return float(val)
    except ValueError:
        if val.lower() == 'true': return True
        if val.lower() == 'false': return False
    return val


class Logger():
    '''
    '''
    def __init__(self, file_name, key_header, value_header):
        self.file_path = log_dir / file_name
        self.key_header = key_header
        self.value_header = value_header
        if not self.file_path.is_file():
            with open(self.file_path, 'w', newline='') as file:
                writer = csv.writer(file, delimiter='\t', quotechar='|')
                writer.writerow(self.key_header+self.value_header)

    def exists(self, keys):
        in_file = np.full(len(keys), False)
        with open(self.file_path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for i, key in enumerate(keys):
                key = [str(entry) for entry in key]
                for row in reader:
                    if key == row[:len(self.key_header)]:
                        in_file[i] = True
                        break
        return in_file

    def write(self, keys, results):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar='|')
            for key, result in zip(keys, results):
                row = [str(a) for a in key] + [str(a) for a in result]
                writer.writerow(row)


def run_evaluation(
    pipeline, dataset, dataset_name, model, evaluators, sample_size=None,
    attribution_types=['segments', 'pixels'], explain='top_class', fraction=1,
    image_idx=0, label_idx=None, per_input=False, version=0, num_workers=10,
    compile=False, verbose=False
):
    '''
    Initializes an evaluation of an image attribution pipeline with the given
    parameters using the given evaluators for a given dataset and model.
    Excludes any parameter combination for which results already exists.
    Records the paramters and results using the given logger in files per each
    dataset and evaluator.

    Args:
        pipeline (callable): Return [N,O], [N,S] predictions and samples
        dataset (iterable): Iterable with [1,H,W,C] input images for the model
        dataset_name (str): Name for the dataset used for logging results
        model (callable): Return [J,O] predictions for image [J,H,W,C]
        evaluators [callable]: Returns [1,1,H,W], [1,1,H,W] attribution maps
        sample_size (int): The nr N of samples to use to perturb
        attribution_types [str]: List of 'segment' and/or 'pixel' attribution
        explain (str): What output to explain ("top_class", "label", or "all")
        fraction (int): Fraction of the data to use for evaluation
        image_idx (int): Index of images in the entries of the dataset
        label_idx (int: Index of the image label (None if no labels)
        per_input (bool): If True, logs evaluation per input instead of aggregate
        version (int): Version used to separate runs with the same parameters
        num_workers (int): Nr of worker processes used to load data
        compile (bool): Whether to use torch.compile to improve computation
        verbose (bool): Whether to print the evaluation progress
    '''
    # Start timer to check speed of evaluation
    if verbose:
        start = datetime.now()

    # Check for incompatible arguments
    if explain == 'label' and label_idx is None:
        raise ValueError('explain cannot be "label" when label_idx is None')

    # If compile is set, use torch.compile to attempt to speed up experiment
    if compile:
        model.__call__ = torch.compile(model.__call__)
        pipeline.__call__ = torch.compile(
            pipeline.__call__
        )
        for evaluator in evaluators:
            evaluator.__call__ = torch.compile(evaluator.__call__)

    # Create general headers for logging
    parameter_header = [
        'segmenter', 'sampler', 'perturber', 'sample_size', 'explained',
        'model', 'fraction', 'version', 'explainer', 'attribution_type',
        'evaluator'
    ]
    general_result_header = ['timestamp', 'nr_model_calls', 'model_accuracy']

    # Create loggers for each evaluator
    loggers = []
    for evaluator in evaluators:
        log_file = f'{dataset_name}_{evaluator.title()}'
        result_header = evaluator.header
        if per_input:
            log_file = log_file + '_per_input'
            result_header = [','.join(result_header)+' ...']
        else:
            result_header = result_header + ['var_'+a for a in result_header]

        log_file = log_file + '.csv'
        result_header = general_result_header + result_header
        loggers.append(Logger(log_file, parameter_header, result_header))

    # Create IDs for each part of the evaluation process
    general_id = [str(a) for a in [
        pipeline.segmenter, pipeline.sampler, pipeline.perturber, sample_size,
        explain, model, fraction, version
    ]]

    # Create holders for each experiment
    experiments = []
    for explainer in pipeline.explainers:
        experiments.append((explainer, [
            (attribution, [
                (eva, log, []) for eva, log in zip(evaluators, loggers)])
            for attribution in attribution_types
        ]))

    # Prune already run experiments
    for i, (explainer, rest) in enumerate(experiments):
        for j, (attribution, evaluations) in enumerate(rest):
            for k, (evaluator, logger, results) in enumerate(evaluations):
                key = general_id + [
                    str(a) for a in [explainer, attribution, evaluator]
                ]
                if logger.exists([key])[0]:
                    evaluations[k] = None
            rest[j] = (attribution, [a for a in evaluations if a != None])
        experiments[i] = (explainer, [a for a in rest if len(a[1])>0])
    experiments = [a for a in experiments if len(a[1])>0]
    pipeline.explainers = [explainer for (explainer, rest) in experiments]
    if len(experiments) == 0:
        return

    # Set the seeds for consistent experiments
    np.random.seed(version)
    random.seed(version)

    # Cut dataset into the given fraction and prepare a loader
    dataset_fraction = Subset(
        dataset, [int(x) for x in np.linspace(
            0, len(dataset), int(len(dataset)*fraction), endpoint=False
        )]
    )
    batch_size = 1
    data_loader = DataLoader(
        dataset_fraction, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Iterate over all images, perturb them, expain them, and calculate AUC
    correct = 0
    nr_model_calls = 0
    for i, data in enumerate(data_loader):

        # If verbose, print info
        if verbose:
            print(
                f'\rEvaluating on {fraction*100}% of {dataset_name} '
                f'[{i+1}/{len(dataset_fraction)}] '
                f'{datetime.now().strftime("%y-%m-%d_%Hh%M")}', end=''
            )

        # Extract the image, label (if applicable) and make a prediction
        image = data[image_idx].permute((0,2,3,1)).numpy(force=True)[0]
        label = None if label_idx is None else data[label_idx]
        predicted = np.argmax(model(image), axis=1)

        # If there is a label, check if it is correctly predicted
        if not label_idx is None and label == predicted[0]:
            correct += 1

        # Set the target to explain
        if explain == 'top_class':
            target = predicted[0]
        elif explain == 'label':
            target = label

        # Get the attribution maps (Return silently if there is pruning error)
        try:
            segment_maps, pixel_maps = pipeline(
                image, model, sample_size=sample_size, output_idxs=target
            )
        except RuntimeError as e:
            if 'All explainers have been pruned raising the' in str(e):
                return
            else:
                raise e

        # Exclude experiments that raises errors (if pipeline.prune=True)
        if len(experiments) != len(pipeline.explainers):
            for j, (explainer, rest) in enumerate(experiments):
                if not explainer in pipeline.explainers:
                    experiments[j] = None
            experiments = [a for a in experiments if a != None]
            if len(experiments) == 0:
                return

        # Get the number of times the model was called
        nr_model_calls += pipeline.nr_model_calls

        # Evaluate each attribution map using each applicable evaluator
        for segment_map, pixel_map, (explainer, rest) in zip(
            segment_maps, pixel_maps, experiments
        ):
            for attribution, evaluations in rest:
                if attribution == 'segments':
                    map = segment_map[0]
                elif attribution == 'pixels':
                    map = pixel_map[0]
                for evaluator, logger, results in evaluations:
                    score = evaluator(
                        image=image, model=model, vals=map, label=label,
                        model_idxs=(...,target)
                    )
                    if per_input:
                        # Store every individual score
                        results.append(score)
                    else:
                        if len(results) == 0:
                            results[:] = [0]*(2*len(evaluator.header))
                        # Store the sum of and sum of squares of each score
                        results[:len(score)] = [
                            r+s for r,s in zip(results[:len(score)], score)
                        ]
                        results[len(score):] = [
                            r+s**2 for r,s in zip(results[len(score):], score)
                        ]

    # Prepare general results (timestamp, nr_model_calls, and maybe accuracy)
    total_num = len(data_loader)
    general_results = []
    general_results.append(datetime.now().strftime('%y-%m-%d_%Hh%M'))
    general_results.append(nr_model_calls)
    if label_idx is None:
        general_results.append('N/A')
    else:
        general_results.append(correct/total_num)

    # Postprocess results and log them to file
    for explainer, rest in experiments:
        for attribution, evaluations in rest:
            for evaluator, logger, results in evaluations:
                if per_input:
                    # Make a strings of all individual results
                    results = [
                        ','.join([f'{r:.10f}' for r in res]) for res in results
                    ]
                else:
                    # Get the variance of results
                    results[int(len(results)/2):] = [
                        (s2-(s*s)/total_num)/(total_num-1) for s, s2 in zip(
                            results[:int(len(results)/2)],
                            results[int(len(results)/2):]
                        )
                    ]
                    # Get mean of results
                    results[:int(len(results)/2)] = [
                        r/total_num for r in results[:int(len(results)/2)]
                    ]
                    # Make results into nice uniform readable strings
                    results = [f'{r:.10f}' for r in results]
                key = general_id + [
                    str(a) for a in [explainer, attribution, evaluator]
                ]
                logger.write([key], [general_results+results])

    if verbose:
        eval_time = (datetime.now()-start).total_seconds()
        print(
            f'\rEvaluation on {fraction*100}% of {dataset_name} ('
            f'{len(dataset_fraction)} images) took {eval_time:.1f} seconds.'
        )
    return


# When this file is executed independently, execute the main function
if __name__ == "__main__":
    main()