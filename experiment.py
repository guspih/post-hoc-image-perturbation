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

# Import samplers
from samplers import RandomSampler, SingleFeatureSampler, ShapSampler
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

# Result file header as divided into parameters and results
result_parameters = [
    'segmenter', 'sampler', 'perturber', 'model', 'explainer', 'sample_size',
    'attribution_type', 'auc_perturber', 'auc_sample_size', 'use_true_class',
    'fraction', 'version'
]
result_values = [
    'timestamp', 'nr_model_calls', 'model_accuracy', 'lif', 'mif', 'srg',
    'norm_lif', 'norm_mif', 'norm_srg', 'var_lif', 'var_mif', 'var_srg',
    'var_norm_lif', 'var_norm_mif', 'var_norm_srg'
]

# Create log file if it does not already exist
log_file = log_dir / 'imagenet_auc_results.csv'
if not log_file.is_file():
    with open(log_file, 'w', newline='') as results:
        results_writer = csv.writer(results, delimiter='\t', quotechar='|')
        results_writer.writerow(result_parameters+result_values)


#perturbation_masks = torch.compile(perturbation_masks)


def run_auc_experiment(
    segmenter, sampler, perturber, model, explainers, sample_size=None,
    attribution_types=['segments', 'pixels'], auc_perturber = None,
    auc_sample_size=10, use_true_class=False, fraction=1, version=0,
    num_workers=10, verbose=False
):
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
    #auc_evaluator.__call__ = torch.compile(auc_evaluator.__call__)

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
    model = TorchModelWrapper(model, transforms, gpu=gpu, softmax_out=True)
    #model.__call__ = torch.compile(model.__call__)

    # Create experiment IDs
    ids = []
    for explainer, attribution_type in tests:
        ids.append([str(y) for y  in [
            segmenter, sampler, perturber, model, explainer, sample_size,
            attribution_type, auc_perturber, auc_sample_size, use_true_class,
            fraction, version
        ]])
    
    # Prune already run experiments
    new_ids = []
    new_tests = []
    for id, test in zip(ids, tests):
        ok = True
        with open(log_file, newline='') as results:
            results_reader = csv.reader(results, delimiter='\t', quotechar='|')
            for row in results_reader:
                if id == row[:len(result_parameters)]:
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
    #prediction_pipeline.__call__=torch.compile(prediction_pipeline.__call__)

    # Collect dataset
    imagenet = dataset_collector(
        'IMAGENET1K2012', split='val', download=False, transform=transforms2
    )
    imagenet_fraction = Subset(
        imagenet, [int() for x in np.linspace(
            0, 50000, int(50000*fraction), endpoint=False
        )]
    )
    imagenet_loader = DataLoader(
        imagenet_fraction, batch_size=100, shuffle=False, num_workers=num_workers
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
            step = i*256+k

            # Check if true and predicted class are the same for verification
            if target == predicted[k]:
                correct += 1

            # If not using the true classes, get the top predicted classes instead
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
                        # Create a return value container for suitable explainers
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
                    f'\rEvaluating on ImageNet validation set ({step+1}/{len(imagenet_fraction)}])',
                    end=''
                )

            # Iterate over explainers and calculate their AUC
            for j, (explainer, attribution_type) in enumerate(tests):
                # Get explanations and calculate the pixel-wise attribution
                attribution = explainer(ys[:, target], samples)[-2]
                if attribution_type == 'segments':
                    masks = prediction_pipeline.masks
                else:
                    masks = prediction_pipeline.transformed_masks
                pixel_map = perturbation_masks(
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
    
def test():
    # Create the segmenters, samplers, perturbers, and explainers
    grid_segmenter = GridSegmenter(10, 10, bilinear=False)
    slic_segmenter = WrapperSegmenter(slic, n_segments=50, compactness=10, start_label=0)
    fade_grid_segmenter = FadeMaskSegmenter(grid_segmenter, sigma=10)

    rise_segmenter = GridSegmenter(7,7, bilinear=True)

    shap_sampler = ShapSampler()
    ciu_sampler  = SingleFeatureSampler(add_none=True)
    rise_sampler = RandomSampler()

    mean_perturber = SingleColorPerturber(color='mean')
    blur_perturber = TransformPerturber(gaussian, sigma=10)
    rand_img = np.random.rand(224,224).astype(np.float32) # Create a random noisy image to replace masked pixels with
    noise_perturber = ReplaceImagePerturber(rand_img)
    inpaint_perturber = Cv2InpaintPerturber()
    histogram_perturber = ColorHistogramPerturber()
    random_perturber = RandomColorPerturber(draw_for_each=True)

    rise_perturber = SingleColorPerturber((0,0,0))

    shap_values = SHAPAttributer()
    original_ciu_values = OriginalCIUAttributer()
    rise_values = RISEAttributer()

    alexnet = torchvision.models.alexnet(weights='IMAGENET1K_V1')
    alexnet.eval()

    vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    vgg16.eval()

    os.environ['TORCHDYNAMO_VERBOSE'] = '1'
    torch._dynamo.config.suppress_errors = True

    #experiment = torch.compile(run_auc_experiment)

    run_auc_experiment(
        rise_segmenter, rise_sampler, rise_perturber, vgg16,
        [shap_values, rise_values], sample_size=4000,
        attribution_types=['segments', 'pixels'], verbose=True, num_workers=20,
        auc_sample_size=10
    )

def main():
    '''
    Collects parameters from when the file is run and evaluates all
    combinations of those parameters on the given benchmark.
    '''
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evaluation',
        type=str,
        default='auc',
        choices=['auc'],
        help='The evaluation to perform (auc: LIF/MIF on ImageNet)'
    )
    parser.add_argument(
        '--nets',
        type=str,
        nargs='+',
        default=['alexnet'],
        choices=['alexnet','vgg16','resnet50'],
        help='Which models to use for evaluation'
    )
    parser.add_argument(
        '--segmenters',
        type=str,
        nargs='+',
        default='grid',
        choices=['grid', 'slic'],
        help='Which segmentation methods to use'
    )
    parser.add_argument(
        '--n_segments',
        type=int,
        nargs='+',
        default=[49],
        help='The approximate number of segments to divide the images into'
    )
    parser.add_argument(
        '--samplers',
        type=str,
        nargs='+',
        default=['single_feature'],
        choices=['single_feature', 'random', 'shap', 'normal', 'inverse_feature'],
        help='The methods to use for sampling the segments to be perturbed'
    )
    parser.add_argument(
        '--blur',
        action='store_true',
        help='Use Gaussian filter to blur the perturbation masks'
    )
    parser.add_argument(
        '--perturber',
        type=str,
        nargs='+',
        default='black',
        choices=['black', 'gray', 'mean', 'histogram', 'blur', 'cv2'],
        help='The methods to use for perturbing the images'
    )
    parser.add_argument(
        '--explainers',
        type=str,
        nargs='+',
        default=['rise'],
        choices=['rise', 'ciu', 'lime', 'shap', 'pda', 'inverse_ciu'],
        help='Which methods to use to calculate attribution of th e'
    )
    parser.add_argument(
        '--sample_sizes',
        type=int,
        nargs='+',
        default=[100],
        help='The sample sizes (if applicable) used when sampling'
    )
    parser.add_argument(
        '--attribution_types',
        type=str,
        nargs='+',
        default=['segments'],
        choices=['segments', 'pixels'],
        help='How to calculate the per pixel attribution'
    )
    parser.add_argument(
        '--auc_samples',
        type=int,
        nargs='+',
        default=10,
        help='The number of samples used in auc evaluation'
    )
    parser.add_argument(
        '--use_true_class',
        action='store_true',
        help='Use actual class instead of predicted class for auc evaluation'
    )
    parser.add_argument(
        '--fraction',
        type=float,
        default=1,
        help='What fraction of the dataset to use in the evaluation'
    )
    parser.add_argument(
        '--versions',
        type=int,
        default=1,
        help='How many times to run non-deteministic evaluations'
    )
    args = parser.parse_args()
