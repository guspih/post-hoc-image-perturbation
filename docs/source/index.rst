.. Perturbation-based Post-hoc Explanations for Image Attribution documentation master file, created by
   sphinx-quickstart on Thu Aug 29 14:42:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for Perturbation-based Post-hoc Explanations for Image Attribution
================================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   post_hoc


Usage
=====

This work provides a generalize perturbation-based pipeline for image attribution that fits many existing works.
The main pipeline is split between the :ref:`samplers <samplers>`, :ref:`image_segmenters <image_segmenters>`, :ref:`image_perturbers <image_perturbers>`, and :ref:`explainers <explainers>` modules.
The :ref:`connectors <connectors>` module contains classes that combine the different pieces into end-to-end pipelines.
The  :ref:`image_visualizers <image_visualizers>`, :ref:`evaluation <evaluation>`, and :ref:`torch_utils <torch_utils>` modules provide image explanation visualization, fidelity metrics, and Torchvision integration respectively.

The Complete Image Attribution Pipeline
---------------------------------------

To explain a model's classification of an image using the package the following steps can be taken.
A showcase of these steps can be found in :code:`torchvision_showcase.ipynb`.

#. Segmenting

   * Initialize a Segmenter from :ref:`image_segmenters <image_segmenters>`.

   * (optional) Wrap the Segmenter using FadeMaskSegmenter to get faded masks for smoother perturbations.

   * Prepare the image to be a numpy array with shape :code:`[Height, Width, Channels]`.
   
   * Call the Segmenter with the image as input to return (:code:`Each pixel indexed with the corresponding segment` (not used further), :code:`one mask for each segment` (1=pixel belongs to segment, 0 otherwise), :code:`one (possibly faded) mask for each segment`).

#. Sampling

   * Intialize a Sampler from :ref:`samplers <samplers>`.

   * Call the Sampler with the number of segments :code:`M` the image was divided into and the number of samples to later perturb.

   * The Sampler returns an array of shape :code:`[sample size, M]` where each row is a sample indicating which segments should be perturbed (=0) or not (=1).

#. Perturbing

   * Use the :code:`perturbation_masks` function from the :ref:`image_segmenters <image_segmenters>` module to create perturbation masks.

      * The function takes the segment_masks (faded or not) and the samples and returns an array of shape :code:`[sample size, Height, Width]` of perturbation masks indicating for each sample which pixels to perturb (=0) and not (=1), or the strength of the perturbation for faded masks (0-1).

   * Initialize a Perturber from :ref:`image_perturbers <image_perturbers>`.

   * Call the Perturber with the image, the perturbation masks, and the samples.

   * The Perturber returns an array of shape :code:`[sample size*N, Height, Width, Channels]` containing all the perturbed versions of the image. The Perturber also returns the index :code:`[sample size*N, M]` indicating which segment is perturbed for each image (:code:`N` is usually 1).

#. Classifying

   * Pick a image classification model of you choice and transform the perturbed images to be suitable for that model.

   * (optional) If using an Torchvision model the :ref:`torch_utils <torch_utils>` provides useful wrappers and transforms to make it function with the other code.

   * Use the model to get classification predictions for the class you wish to explain (e.g. the true class or the top predicted class).

   * Format the desired predictions into a numpy array of shape :code:`[sample size*N]`.

#. Attributing

   * Initialize a Attributer from :ref:`explainers <explainers>`.

   * Call the Attributer with the predictions and the perturbation index of shape :code:`[sample size*N, M]`.

   * The Attributer returns a tuple with that Attributer's explanations, but :code:`tuple[-2]` always contain that Attributer's influence per feature scores. Which scores belong to what features are indexed by :code:`tuple[-1]`.

   * Among other things the most positively influential segment can be found as :code:`tuple[-1][np.argmax(tuple[-2])]` and the most influential (in either direction) as :code:`tuple[-1][np.argmax(np.abs(tuple[-2]))]`.

Building Connected Pipelines
----------------------------

Doing each of the steps described above can get tedious and take a lot of space.
Since these steps are standardized classes for creating end-to-end pipelines are provided by the :ref:`connectors <connectors>` module.
A showcase of using :ref:`connectors <connectors>` module can be found in :code:`torchvision_showcase_2.ipynb`.
To use the :code:`SegmentationAttribuitionPipeline` follow the steps below.

#. Preparations

   * Intialize a Segmenter, Sampler, Perturber, and Attributer

   * Format the image into a numpy array of shape :code:`[Height, Width, Channels]`

   * Create a model that takes numpy images in shape :code:`[Batch size, Height, Width, Channels]` and returns a numpy array of shape :code:`[Batch size, Outputs]` (for Torchvision models this can be achieved with the :ref:`torch_utils <torch_utils>` module).

   * Initialize :code:`SegmentationAttribuitionPipeline` from the :ref:`connectors <connectors>` module using the Segmenter, Sampler, Perturber, and Attributer. Also select whether to use per pixel explanation and what batch size to use for predictions.

#. Running the Pipeline

   * Call the Pipeline with the image, model, and sample size

   * The pipeline returns, for each output class, the attribution score per segment and, if chosen, the attribution per pixel.

Alternatively the :code:`SegmentationPredictionPipeline` can be used to only get the predictions per perturbed image and perturbation indexes which will allow you to calculate the attribution for the same pipeline with different Attributers.

Visualizing the Attribution
---------------------------

Beyond attributing influence to each segment, these scores can be visualized in order to make them clearer.
Some of the classes of the :ref:`image_visualizers <image_visualizers>` module can be used to visualize attribution of image segments.
:code:`TopVisualizer` and `HeatmapVisualizer` takes the attribution scores, the image, and the segment masks and visualize the scores on top of the image, by respecitvely displaying the top influential segments or displaying a heatmap of the scores on top of the image.
Using faded masks will cause the scores to be spread out per pixel instead of per segment.
Optionally, if per pixel scores are already obtained, that map can be passed as the scores and masks can be skipped entierly.

Evaluating the Attribution
--------------------------

While explanation methods should ideally be evaluated with user testing, this is not always feasible.
Instead some fidelity metrics have been introduced to enable automated evaluation.
This repository currently implements occlusion (or Area Under the Curve) and pointing game metrics for images in the :ref:`evaluation <evaluation>` module.
The :code:`ImageAUCEvaluator` can take an image, a model, and attribution scores and evaluate the fidelity of the attribution by iteratively occluding the least or most influential pixels and evaluating how much the prediction changes as the area under the prediction-occlusion curve.
The :code:`PointingGameEvaluator` can take a mask showing the salient area of the attributed image and calculates a whether the most attributed pixel falls within the area.

Indices and tables
==================

* :ref:`genindex`
