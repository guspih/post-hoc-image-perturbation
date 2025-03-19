# Perturbation-based Post-hoc Explanations for Image Attribution

This repository contains code for creating many combinations of image perturbation pipelines used to explain image prediction models.
It contains many common implementations of the different parts of the image perturbation pipeline, including some which are domain agnostic.
Additionally, the repository contains code for automatically evaluating such pipelines using occlusion metrics.

## Requirements

The code has been tested using `Python 3.11`, `PyTorch 2.3.0`, and `Torchvision 0.15.2`, but likely works with other versions as well.
In addition the code uses the following Python packages:
| Package          | Tested version | Required for                                  |
| ---------------- | -------------- | --------------------------------------------- |
| `numpy`          | `1.26.4`       | Everything                                    |
| `pillow`         | `10.3.0`       | Dataset collection and showcases              |
| `scikit-image`   | `0.22.0`       | Image processsing                             |
| `scipy`          | `1.13.1`       | Everything                                    |
| `opencv` (`cv2`) | `4.7.0`        | Image processing                              |
| `matplotlib`     | `3.8.4`        | Image attribution visualization and showcases |

The code works with and without GPU, though running automated experiments is significantly faster with GPU and debilitatingly slow without.

## Usage

How the different components of this repository can be used to create image explanation pipelines is showcased in the `torchvision_showcase` notebooks.
`torchvision_showcase.ipynb` notebook shows how the different objects can be instantiated and called one-by-one using eachother's outputs to finally calculate attribution scores.
`torchvision_showcase_2.ipynb` contains a similar example to the first but show how the same results can be achieved easier by using the provided wrapper objects.
Further information about the many components, what they do, and how they combine can be found in `docs/build/html/index.html`.

The experiments that this repository is used for can be automatically run using `experiment.py`.
Calling `python experiment.py --help` returns instructions for how to run experiments.


The showcases and experiments are currently optimized to work on a fairly high-end computer, and may need to be altered to work on other devices, especially those with small RAM and VRAM.
In the future, this will be addressed.

## Citation

The experiments in this repository has been used in the work [Segmentation and Smoothing Affect Explanation Quality More Than the Choice of Perturbation-based XAI Method for Image Explanations](https://arxiv.org/abs/2409.04116).
If you use this repository in you research, please cite that work.
The BibTex entry for the work can be found below:

```bash
@article{pihlgren2024smooth,
    title={Smooth-edged Perturbations Improve Perturbation-based Image Explanations},
    author={Gustav Grund Pihlgren and Kary Främling},
    doi = {10.48550/arXiv.2409.04116},
    journal = {arXiv preprint},
    year={2024}
}
```
