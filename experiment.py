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

# Import samplers
from samplers import (
    RandomSampler, SingleFeatureSampler, ShapSampler, SampleProbabilitySampler
)
# Import explainers
from explainers import (
    OriginalCIUAttributer, SHAPAttributer, RISEAttributer,
    LinearLIMEAttributer, PDAAttributer
)
# Import segmenters
from image_segmenters import GridSegmenter, WrapperSegmenter, FadeMaskSegmenter
# Import segmentation and perturbation utils
from image_segmenters import perturbation_masks
# Import image perturbers
from image_perturbers import (
    SingleColorPerturber, ReplaceImagePerturber, TransformPerturber,
    RandomColorPerturber, ColorHistogramPerturber, Cv2InpaintPerturber
)
# Import connectors
from connectors import SegmentationPredictionPipeline
# Import PyTorch utils
from torch_utils import TorchModelWrapper, ImageToTorch
# Import explanation evaluators
from evaluation import ImageAUCEvaluator, PointingGameEvaluator
# Import dataset handler
from dataset_collector import dataset_collector

# Set logging folder
from workspace_path import home_path
log_dir = home_path / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)


# Headers of result files as divided into parameters and results
general_parameters = [
    'segmenter', 'sampler', 'perturber', 'model', 'explainer', 'sample_size',
    'attribution_type', 'fraction', 'version'
]
auc_parameters = ['auc_perturber', 'auc_sample_size', 'use_true_class',]
auc_results = [
    'timestamp', 'nr_model_calls', 'model_accuracy', 'lif', 'mif', 'srg',
    'norm_lif', 'norm_mif', 'norm_srg', 'var_lif', 'var_mif', 'var_srg',
    'var_norm_lif', 'var_norm_mif', 'var_norm_srg'
]


def run_auc_experiment(
    segmenter, sampler, perturber, model, explainers, sample_size=None,
    attribution_types=['segments', 'pixels'], softmax=False,
    auc_perturber = None, auc_sample_size=10, use_true_class=False,
    fraction=1, version=0, num_workers=10, compile=False, verbose=False
):
    # Create auc log file if it does not already exist
    log_file = log_dir / 'imagenet_auc_results.csv'
    if not log_file.is_file():
        with open(log_file, 'w', newline='') as results:
            results_writer = csv.writer(results, delimiter='\t', quotechar='|')
            results_writer.writerow(
                general_parameters+auc_parameters+auc_results
            )

    # Only need to run non-deterministic experiments once
    if version>0 and sampler.deterministic and perturber.deterministic:
        return

    # Combine explainers and attributions types (and copy explainers container)
    tests = [(exp,att) for att in attribution_types for exp in explainers]

    # Create evaluator
    if auc_perturber is None:
        SingleColorPerturber(color='mean')
    auc_evaluator = ImageAUCEvaluator(
        mode='srg', perturber=auc_perturber, return_curves=True
    )

    # Create a transform to prepare images for Torchvision
    transforms = v1.Compose([
        ImageToTorch(),
        v1.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v1.Resize((224,224), antialias=True)
    ])
    transforms2 = v1.Compose([
        v1.ToTensor(),
        v1.Resize((224,224), antialias=True)
    ])

    # Run on GPU?
    gpu = torch.cuda.is_available()

    # Wrap model
    model = TorchModelWrapper(model, transforms, gpu=gpu, softmax_out=softmax)

    # Create experiment IDs
    ids = []
    for explainer, attribution_type in tests:
        ids.append([str(y) for y  in [
            segmenter, sampler, perturber, model, explainer, sample_size,
            attribution_type, fraction, version, auc_perturber,
            auc_sample_size, use_true_class
        ]])
    
    # Prune already run experiments
    new_ids = []
    new_tests = []
    for id, test in zip(ids, tests):
        ok = True
        with open(log_file, newline='') as results:
            results_reader = csv.reader(results, delimiter='\t', quotechar='|')
            for row in results_reader:
                if id == row[:len(general_parameters+auc_parameters)]:
                    ok = False
                    break
            if ok:
                new_ids.append(id)
                new_tests.append(test)
    if len(new_tests) == 0:
        return
    tests = new_tests
    ids = new_ids

    # Set the seeds for consistent experiments
    np.random.seed(version)

    # Create pipeline
    prediction_pipeline = SegmentationPredictionPipeline(
        segmenter, sampler, perturber, batch_size=128
    )

    # If compile is set, use torch.compile to attempt to speed up experiment
    if compile:
        model.__call__ = torch.compile(model.__call__)
        prediction_pipeline.__call__ = torch.compile(
            prediction_pipeline.__call__
        )
        auc_evaluator.__call__ = torch.compile(auc_evaluator.__call__)
        maskify = torch.compile(perturbation_masks)
    else:
        maskify = perturbation_masks

    # Collect dataset
    imagenet = dataset_collector(
        'IMAGENET1K2012', split='val', download=False, transform=transforms2
    )
    imagenet_fraction = Subset(
        imagenet, [int() for x in np.linspace(
            0, 50000, int(50000*fraction), endpoint=False
        )]
    )
    batch_size = 1
    imagenet_loader = DataLoader(
        imagenet_fraction, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Iterate over all images, perturb them, expain them, and calculate AUC
    total_num = 0
    correct = 0
    nr_model_calls = 0
    for i, (image, target) in enumerate(imagenet_loader):
        image = image.permute((0,2,3,1)).numpy(force=True)
        predicted = np.argmax(model(image), axis=1)
        for k, (image, target) in enumerate(zip(image, target)):
            total_num += 1
            step = i*batch_size+k

            # Check if true and predicted class are the same for verification
            if target == predicted[k]:
                correct += 1

            # If not using the true classes, get the top predicted classes
            if not use_true_class:
                target = predicted[k]

            # Get the y values for each prediction
            ys, samples = prediction_pipeline(
                image, model, sample_size=sample_size
            )

            # Get the number of times the model was called
            nr_model_calls += prediction_pipeline.nr_model_calls

            # Exclude explainers that cannot handle the current pipeline
            if step == 0:
                new_tests = []
                new_ids = []
                ret = []
                for id, (explainer, attribution_type) in zip(ids, tests):
                    try:
                        explainer(ys[:, target], samples)
                        new_tests.append((explainer, attribution_type))
                        new_ids.append(id)
                        # Create a return value container the explainer
                        ret.append([0 for _ in range(12)])
                    except Exception:
                        pass
                # Stop experiment if there are no suitable explainers
                if len(new_tests) == 0:
                    return
                tests = new_tests
                ids = new_ids
            
            # If verbose, print info
            if verbose and step%10 == 9:
                print(
                    f'\rEvaluating on ImageNet validation set '
                    f'({step+1}/{len(imagenet_fraction)}])', end=''
                )

            # Iterate over explainers and calculate their AUC
            for j, (explainer, attribution_type) in enumerate(tests):
                # Get explanations and calculate the pixel-wise attribution
                attribution = explainer(ys[:, target], samples)[-2]
                if attribution_type == 'segments':
                    masks = prediction_pipeline.masks
                else:
                    masks = prediction_pipeline.transformed_masks
                pixel_map = maskify(
                    masks, attribution.reshape((1,-1))
                )

                # Calculate the AUC scores
                scores, curves = auc_evaluator(
                    image, model, pixel_map, sample_size=auc_sample_size,
                    model_idxs=(...,target)
                )

                # AUC mean and variance
                for k in range(0,3):
                    ret[j][k] += scores[k]
                for k in range(0,3):
                    ret[j][k+6] += scores[k]**2

                # Normalized AUC mean and variance
                scores, curves = auc_evaluator.get_normalized()
                for k in range(0,3):
                    ret[j][k+3] += scores[k]
                for k in range(0,3):
                    ret[j][k+9] += scores[k]**2

    for j, (id, (explainer, attribution_type)) in enumerate(zip(ids, tests)):
        # AUC mean
        for k in range(0,6):
            ret[j][k] = ret[j][k][0] / total_num
        # AUC variance
        for k in range(6,12):
            ret[j][k] = np.sqrt((
                    ret[j][k][0]*total_num - ret[j][k-3]**2
                )/(total_num*(total_num-1))
            )
        # Timestamp
        timestamp = datetime.now().strftime('%y-%m-%d_%Hh%M')
        # Print experiment parameters and results to log file
        with open(log_file, 'a', newline='') as results:
            results_writer = csv.writer(results, delimiter='\t', quotechar='|')
            results_writer.writerow(
                id+[timestamp]+[nr_model_calls]+[correct/total_num]+ret[j]
            )
    print()
    return


def main():
    '''
    Collects parameters from when the file is run and evaluates all
    combinations of those parameters on the given benchmark.
    '''
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evaluation', type=str, default='auc', choices=['auc'],
        help='The evaluation to perform (auc: LIF/MIF on ImageNet)'
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
        '--segmenters', type=str, nargs='+', default='grid',
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
        '--perturbers', type=str, nargs='+', default='black',
        choices=['black', 'gray', 'mean', 'histogram', 'blur', 'cv2'],
        help='The methods to use for perturbing the images'
    )
    parser.add_argument(
        '--explainers', type=str, nargs='+', default=['rise'],
        choices=['rise', 'ciu', 'lime', 'shap', 'pda', 'inverse_ciu'],
        help='Which methods to use to calculate attribution of the segments'
    )
    parser.add_argument(
        '--sample_sizes', type=int, nargs='+', default=[100], metavar='Nr',
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
    args = parser.parse_args()

    # Parse any additional key=value arguments
    kw_dicts = {'segmenter':{},'sampler':{},'perturber':{},'explainer':{}}
    for dict_name, kw_strs in zip((
        'segmenter', 'sampler', 'perturber', 'explainer'
    ) ,[
        args.segmenter_kw,args.sampler_kw,args.perturber_kw,args.explainer_kw
    ]):
        for key_value in kw_strs:
            key, value = key_value.split('=')
            value = cancast(value)
            kw_dicts[dict_name][key] = value

    # For each combination of net, segmenter, n_seg, perturber, and sample_size
    for (
        net, segmenter, n_seg, sampler, perturber, sample_size
    ) in itertools.product(
        args.nets, args.segmenters, args.n_segments, args.samplers,
        args.perturbers, args.sample_sizes
    ):
        # Get model and weights
        weights = 'IMAGENET1K_V2' if net == 'resnet50' else 'IMAGENET1K_V1'
        net = torchvision.models.__dict__[net](weights=weights)

        # Create the segmenter
        hw = round(np.sqrt(n_seg))
        if segmenter == 'grid':
            segmenter = GridSegmenter(
                hw, hw, kw_dicts['segmenter'].get('bilinear', False)
            )
        elif segmenter == 'slic':
            segmenter = WrapperSegmenter(
                slic, n_segments=n_seg, start_label=0, **kw_dicts['segmenter']
            )

        if args.blur:
            sigma = 70/hw # To get similar blur to bilinear in GridSegmenter
            segmenter = FadeMaskSegmenter(segmenter, sigma=sigma)

        # Create the sampler
        if sampler == 'single_feature':
            sampler = SingleFeatureSampler(**kw_dicts['sampler'])
            sample_size = None # This sampler has a fixed sample size
        elif sampler == 'random':
            sampler = RandomSampler(kw_dicts['sampler'].get('p', 0.5))
        elif sampler == 'shap':
            sampler = ShapSampler(
                kw_dicts['sampler'].get('ignore_warnings', False)
            )
        elif sampler == 'normal':
            sampler = SampleProbabilitySampler(
                distribution='normal', **kw_dicts['sampler']
            )
        elif sampler == 'inverse_feature':
            sampler = SingleFeatureSampler(inverse=True, **kw_dicts['sampler'])
            sample_size = None # This sampler has a fixed sample size

        # Create the perturber
        if perturber == 'black':
            perturber = SingleColorPerturber((0,0,0))
        elif perturber == 'gray':
            perturber = SingleColorPerturber((0.75,0.75,0.75))
        elif perturber == 'mean':
            perturber = SingleColorPerturber('mean')
        elif perturber == 'histogram':
            perturber = ColorHistogramPerturber(**kw_dicts['perturber'])
        elif perturber == 'blur':
            perturber = TransformPerturber(gaussian, **kw_dicts['perturber'])
        elif perturber == 'cv2':
            perturber = Cv2InpaintPerturber(**kw_dicts['perturber'])

        # Create the explainers
        explainers = []
        if 'rise' in args.explainers:
            explainers.append(RISEAttributer())
        if 'ciu' in args.explainers:
            explainers.append(OriginalCIUAttributer(**kw_dicts['explainer']))
        if 'lime' in args.explainers:
            explainers.append(LinearLIMEAttributer())
        if 'shap' in args.explainers:
            explainers.append(SHAPAttributer())
        if 'pda' in args.explainers:
            explainers.append(PDAAttributer(**kw_dicts['explainer']))
        if 'inverse_ciu' in args.explainers:
            explainers.append(OriginalCIUAttributer(
                inverse=True, **kw_dicts['explainer']
            ))

        # Do not use both attribution types if they are equivalent
        if not args.blur and kw_dicts['segmenter'].get('bilinear', False):
            attribution_types = ['segments']
        else:
            attribution_types = args.attribution_types

        if args.evaluation == 'auc':
            experiment = run_auc_experiment
            if args.compile:
                os.environ['TORCHDYNAMO_VERBOSE'] = '1'
                torch._dynamo.config.suppress_errors = True
                experiment = torch.compile(run_auc_experiment)
            for version in range(args.versions):
                if not args.quiet:
                    start = datetime.now()
                experiment(
                    segmenter, sampler, perturber, net, explainers,
                    sample_size=sample_size,
                    attribution_types=attribution_types, softmax=args.softmax,
                    auc_perturber=None, auc_sample_size=args.auc_samples,
                    use_true_class=args.use_true_class, fraction=args.fraction,
                    version=version, num_workers=10, compile=args.compile,
                    verbose=not args.quiet
                )
                if not args.quiet:
                    eval_time = (datetime.now()-start).total_seconds()
                    print(f'AUC Evaluation run in {eval_time} seconds')


def cancast(val):
    try:
        return float(val)
    except ValueError:
        if val.lower() == 'true': return True
        if val.lower() == 'false': return False
    return val


# When this file is executed independently, execute the main function
if __name__ == "__main__":
    main()
