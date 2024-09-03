import os
import numpy
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
import torchvision.datasets
import torchvision.transforms
from torchvision.transforms import ToTensor
from PIL import Image
import requests
import zipfile
import tarfile
import csv
'''
This file contains functions for loading (and if necessary downloading) the
relevant datasets. Currently available datasets are STL-10, SVHN, ISBI12, MRD,
MSCOCO 2014, BAPPS, PASCAL VOC, ImageNet1K 2012
'''

# Path to root folder for datasets
from workspace_path import home_path
root_folder = home_path / 'datasets'
root_folder.mkdir(parents=True, exist_ok=True)


def dataset_collector(dataset, split, **kwargs):
    '''
    Wrapper for all collectors, will use the datasets dict to see if a given
    dataset and split is available and return it using its specified collector.

    Args:
        dataset (str): Key for the dataset in the datasets dict
        split (str): An available split for the given dataset
        **kwargs (dict): Any additional parameters
    Returns:
        torch.utils.data.Dataset: The collected dataset
    '''
    if dataset not in datasets:
        available_datasets = ', '.join(datasets.keys())
        raise ValueError(f'Unexpected value of dataset: {dataset}. '
                         f'Available datasets are {available_datasets}')
    dataset_info = datasets[dataset]
    if split not in dataset_info['split']:
        available_splits = ', '.join(dataset_info['split'])
        raise ValueError(
            f'Unexpected value of split: {split}. '
            f'Available splits for this dataset are {available_splits}')
    kwargs['dataset'] = dataset
    if not 'split_name' in dataset_info:
        kwargs['split'] = split
    else:
        kwargs[dataset_info['split_name']] = split
    if 'default_parameters' in dataset_info:
        for parameter, value in dataset_info['default_parameters'].items():
            if not parameter in kwargs:
                kwargs[parameter] = value
    return dataset_info['source'](**kwargs)


def torchvision_collector(dataset, **kwargs):
    '''
    Function for collecting and, if necessary, downloading torchvision datasets
    Available datasets are 'STL19', 'SVHN', and 'VOCDetection'.

    Args:
        dataset (str): The dataset to collect
        **kwargs (dict): Any additional parameters for collection
    Returns:
        torch.utils.data.Dataset: The collected Torchvision dataset
    '''
    if datasets[dataset]['downloadable'] and 'download' not in kwargs:
        kwargs['download'] = True
    return torchvision.datasets.__dict__[dataset](root_folder/dataset,**kwargs)


def isbi12_collector(split='train', **kwargs):
    '''
    Function for collecting and, if necessary, explaining how to download and
    package the ISBI12 (Segmentation of neuronal structures in EM stacks)
    dataset.

    Args:
        split (str): Which split to collect ('train' or 'unlabeled')
        **kwargs (dict): Any additional parameters for collection
    Returns:
        torch.utils.data.Dataset: The given split of the ISBI dataset
    '''
    if 'download' not in kwargs:
        kwargs['download'] = True

    (root_folder/'ISBI12').mkdir(parents=True, exist_ok=True)
    if not ((root_folder/'ISBI12/train-volume.tif').exists() and
            (root_folder/'ISBI12/train-labels.tif').exists() and
            (root_folder/'ISBI12/test-volume.tif').exists()):
        if kwargs['download']:
            kaggle_download(
                'soumikrakshit/isbi-challenge-dataset',
                root_folder/'ISBI12',
                unzip=True
            )
        else:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them'
            )
    if split == 'train':
        images = Image.open(root_folder / 'ISBI12/train-volume.tif')
        labels = Image.open(root_folder / 'ISBI12/train-labels.tif')
        data = [ToTensor()(images), ToTensor()(labels)]
    elif split == 'unlabeled':
        images = Image.open(root_folder + '/ISBI12/test-volume.tif')
        data = [ToTensor()(images)]
    else:
        raise ValueError(f'Unexpected value of split: {split}. \'train\' or '
                         f'\'unlabeled\' expected')
    return TensorDataset(*data)


def mrd_collector(split='train', **kwargs):
    '''
    Function for collecting and, if necessary, downloading the Massachusetts
    Roads Dataset.

    Args:
        split (str): Which split to collect ('train' or 'test')
        **kwargs (dict): Any additional parameters for collection
    Returns:
        torch.utils.data.Dataset: The given split of the MRD dataset
    '''
    (root_folder/'MRD').mkdir(parents=True, exist_ok=True)
    # This kaggle dataset does not have labels for all images, just fyi
    if not (root_folder/'MRD/road_segmentation_ideal').exists():
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them')
        kaggle_download(
            'insaff/massachusetts-roads-dataset',
            root_folder/'MRD',
            unzip=True
        )
    if split == 'train':
        folder = root_folder/'MRD/road_segmentation_ideal/training'
    elif split == 'test':
        folder = root_folder/'MRD/road_segmentation_ideal/testing'
    else:
        raise ValueError(
            f'Unexpected value of split: {split}. \'train\' or \'test\' '
            f'expected'
        )
    return MultipleFolderDataset(
        folder/'input', folder/'output',
        image_transform=kwargs.get('image_transform')
    )


def coco2014_collector(split='train', **kwargs):
    '''
    Function for collecting and, if necessary, downloading the MSCOCO 2014
    dataset.

    Args:
        split (str): Which split to collect ('train', 'val', or 'test')
        **kwargs (dict): Any additional parameters for collection
    Returns:
        torch.utils.data.Dataset: The given split of the MSCOCO dataset
    '''
    (root_folder/'COCO2014/train2014').mkdir(parents=True, exist_ok=True)
    (root_folder/'COCO2014/val2014').mkdir(parents=True, exist_ok=True)
    (root_folder/'COCO2014/test2014').mkdir(parents=True, exist_ok=True)
    if (
        len(os.listdir(root_folder/f'COCO2014/{split}2014')) < 2
    ):
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them'
            )
        urls = [f'http://images.cocodataset.org/zips/{split}2014.zip']
        filenames = [root_folder/f'COCO2014/{split}2014.zip']
        if split != 'test':
            urls.append(
                'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
            )
            filenames.append(
                root_folder/f'COCO2014/annotations_trainval2014.zip'
            )
        for url, filename in zip(urls, filenames):
            if not os.path.isfile(filename):
                download_raw_url(url=url, save_path=filename)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(root_folder/'COCO2014')
            os.remove(filename)
    return MultipleFolderDataset(
        root_folder/f'COCO2014/{split}2014',
        image_transform=kwargs.get('image_transform')
    )


def imagenet1K2012_collector(split='train', **kwargs):
    '''
    Function for collecting and, if necessary, downloading the ImageNet1K 2012
    dataset. The test set can be downloaded, but is not usable in torchvision.

    Args:
        split (str): Which split to collect ('train', 'val', or 'test')
        **kwargs (dict): Any additional parameters for collection
    Returns:
        torch.utils.data.Dataset: The given split of the ImageNet dataset
    '''
    (root_folder/f'IMAGENET1K2012/{split}').mkdir(parents=True, exist_ok=True)
    if split == 'test':
        split = 'test_v10102019'
    length = {
        'train': 1281167,
        'val': 50000,
        'test_v10102019': 100000
    }
    if sum(
        [len(f) for r,d,f in os.walk(root_folder/f'IMAGENET1K2012/{split}')]
    ) < length[split]:
        files = ['ILSVRC2012_devkit_t12.tar.gz', f'ILSVRC2012_img_{split}.tar']
        for file in files:
            filename = root_folder/f'IMAGENET1K2012/{file}'
            if not filename.is_file():
                if 'download' in kwargs and not kwargs['download']:
                    raise FileNotFoundError(
                        'Files are missing. Set \'download\' to True to '
                        'automatically download them'
                    )
                if not os.path.isfile(filename):
                    download_raw_url(
                        url=f'https://image-net.org/data/ILSVRC/2012/{file}',
                        save_path=filename
                    )
        if filename.is_file():
            os.remove(filename)
    [kwargs.pop(s, None) for s in ['dataset', 'download']]
    data = torchvision.datasets.ImageNet(
        root=root_folder/'IMAGENET1K2012',
        split=split,
        **kwargs
    )
    return data


def bapps_collector(split='train', subsplit='all', **kwargs):
    '''
    Function for collecting and, if necessary, downloading the BAPPS dataset.

    Args:
        split (str): Which split to collect ('train', 'val', 'jnd/val')
        subsplit (str): Which subsplit to collect ('all' collects all)
        **kwargs (dict): Any additional parameters for collection
    Returns:
        torch.utils.data.Dataset: The given split of the BAPPS dataset
    '''
    (root_folder/'BAPPS/train').mkdir(parents=True, exist_ok=True)
    (root_folder/'BAPPS/val').mkdir(parents=True, exist_ok=True)
    (root_folder/'BAPPS/jnd/val').mkdir(parents=True, exist_ok=True)
    if (
        len(os.listdir(root_folder/'BAPPS/train')) < 3
        or len(os.listdir(root_folder/'BAPPS/val')) < 6
        or len(os.listdir(root_folder/'BAPPS/jnd/val')) < 2
    ):
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them')
        print('Downloading BAPPS dataset...')
        dataset_link = 'https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset'
        dataset_parts = ['twoafc_train', 'twoafc_val', 'jnd']
        for part in dataset_parts:
            filename = root_folder/f'BAPPS/{part}.tar.gz'
            if not os.path.isfile(filename):
                download_raw_url(url=f'{dataset_link}/{part}.tar.gz',
                                 save_path=filename)
            with tarfile.TarFile(filename, 'r') as tar_ref:
                tar_ref.extractall(root_folder/'BAPPS')
    subsplits = os.listdir(root_folder/f'BAPPS/{split}')
    if subsplit == 'all':
        ret = []
        for subsplit in subsplits:
            dirs = os.listdir(root_folder/f'BAPPS/{split}/{subsplit}')
            paths = [root_folder/f'BAPPS/{split}/{subsplit}/{d}' for d in dirs]
            ret.append(MultipleFolderDataset(
                *paths, name=subsplit,
                image_transform=kwargs.get('image_transform'))
            )
        return ConcatDataset(ret)
    elif subsplit in subsplits:
        dirs = os.listdir(root_folder/f'BAPPS/{split}/{subsplit}')
        paths = [root_folder/f'BAPPS/{split}/{subsplit}/{d}' for d in dirs]
        return MultipleFolderDataset(
            *paths, name=subsplit,
            image_transform=kwargs.get('image_transform')
        )
    else:
        raise ValueError(
            f'Unexpected value of subsplit: {subsplit}. '
            f'Expected any of: all, {", ".join(subsplits+["all"])}'
        )


# Dictionary of available datasets and their attributes and parameters
datasets = {
    'STL10': {
        'full_name': 'STL-10 dataset',
        'source': torchvision_collector,
        'downloadable': True,
        'split': ['train', 'test', 'unlabeled', 'train+unlabeled'],
        'folds': [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'output_format': 'index',
        'nr_of_classes': 10
    },
    'SVHN': {
        'full_name': 'Street View House Numbers dataset',
        'source': torchvision_collector,
        'downloadable': True,
        'split': ['train', 'test', 'extra'],
        'output_format': 'index',
        'nr_of_classes': 10
    },
    'ISBI12': {
        'full_name': 
            'ISBI Challenge: Segmentation of neuronal structures in EM stacks',
        'source': isbi12_collector,
        'downloadable': True,
        'split': ['train', 'unlabeled'],
        'output_format': None  # TODO: annotate the format (eg 'onehot')
    },
    'MRD': {
        'full_name': 'Massachusetts Roads Dataset',
        'source': mrd_collector,
        'downloadable': True,
        'split': ['train', 'validation', 'test'],
        'output_format': None  # TODO: annotate the format (eg 'onehot')
    },
    'COCO2014': {
        'full_name': 'Common Objects in Context 2014',
        'source': coco2014_collector,
        'downloadable': True,
        'split': ['train', 'val', 'test'],
        'output_format': 'none'
    },
    'BAPPS': {
        'full_name': 'Berkeley Adobe Perceptual Patch Similarity',
        'source': bapps_collector,
        'downloadable': True,
        'split': ['train', 'val', 'jnd/val'],
        'output_format': None  # TODO: annotate the format (eg 'onehot')
    },
    'VOCSegmentation': {
        'full_name': 'PASCAL VOC2007',
        'source': torchvision_collector,
        'downloadable': True,
        'split': ['train', 'trainval', 'val', 'test'],
        'split_name': 'image_set',
        'default_parameters': {'year': '2007'},
        'output_format': None  # TODO: annotate the format (eg 'onehot')

    },
    'IMAGENET1K2012': {
        'full_name': 'ImageNet Large Scale Visual Recognition Challenge 2012',
        'source': imagenet1K2012_collector,
        'downloadable': True,
        'split': ['train', 'val', 'test'],
        'output_format': None  # TODO: annotate the format (eg 'onehot')
    }
}


def download_raw_url(url, save_path, show=False, chunk_size=128, decode=False):
    '''
    Downloads raw data from url. Reworked from:
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url

    Args:
        url (str): Url of raw data
        save_path (str): File name to store data under
        show (bool): Whether to print what is being downloaded
        chunk_size (int): How large chunks of data to collect at a time
    '''
    if show:
        print(f'\rDownloading URL: {url}', end='')
    r = requests.get(url, stream=True)
    if decode:
        r.raw.decode_content = True
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def _kaggle_init():
    '''
    Prepares and returns a Kaggle API using keys in api_keys.csv

    Returns:
        KaggleApi: A prepared and authenticated Kaggle API
    '''
    keyfile = home_path/'api_keys.csv'
    if not keyfile.is_file():
        raise RuntimeError(
            f'Missing file {keyfile}. Create it as an empty file and rerun.'
        )
    kaggle_username = None
    kaggle_key = None
    with open(home_path/'api_keys.csv') as keyfile:
        key_reader = csv.reader(keyfile)
        for key, value in key_reader:
            if key == 'kaggle_username':
                kaggle_username = value
            elif key == 'kaggle_key':
                kaggle_key = value  
    if kaggle_username is None or kaggle_key is None:
        raise RuntimeError(
            f'Kaggle keys missing. Log in to '
            f'https://www.kaggle.com/<username>/account and download an API '
            f'token via "Create API Token" and paste the username and key '
            f'into {keyfile} as "kaggle_username,<username>" and '
            f'"kaggle_key,<key>"')
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def kaggle_download(dataset, path, **kwargs):
    '''
    Downloads a given dataset from kaggle to a given folder.
    
    Args:
        dataset (str): Dataset to download
        path (str): Path to folder of dataset
        **kwargs (dict): Any additional parameters for downloading
    '''
    api = _kaggle_init()
    api.dataset_download_files(dataset, path, **kwargs)


class MultipleFolderDataset(Dataset):
    '''
    A dataset for loading data where data is contained in folders where each
    matching group of data has the same name (with possibly different
    file-endings).
    Allowed file types: .png, .tif, .tiff, .jpg, .jpeg, .bmp, .npy 

    Args:
        *args (str): Paths to the folders to extract from
        name (str): A name to be returned together with each datapoint
        image_transform (nn.Module): Transform applied when getting images
    '''
    def __init__(self, *args, name=None, image_transform=None):
        super().__init__()
        if len(args) < 1:
            raise RuntimeError('Must be given at least one path')
        self.name = name
        acceptable_endings = [
            'png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp', 'npy'
        ]
        folder_files = []
        for folder in args:
            files = os.listdir(folder)
            folder_files.append({
                f[:f.index('.')]: f[f.index('.') + 1:]
                for f in files if f[f.index('.') + 1:] in acceptable_endings
            })
        self.data_paths = []
        for filename, ending in folder_files[0].items():
            paths = [f'{args[0]}/{filename}.{ending}']
            for folder, arg in zip(folder_files[1:], args[1:]):
                if filename in folder:
                    paths.append(f'{arg}/{filename}.{folder[filename]}')
                else:
                    break
            if len(paths) != len(args):
                continue
            self.data_paths.append(paths)
        self.image_transform = image_transform
        if self.image_transform is None:
            self.image_transform = ToTensor()


    def __getitem__(self, index):
        image_endings = ['png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp']
        npy_endings = ['npy']

        ret = {} #= []
        for path in self.data_paths[index]:
            ending = path[path.index('.') + 1:]
            folder = path.split('/')[-2]
            if ending in image_endings:
                image = Image.open(path).convert(mode='RGB')
                ret[folder] = self.image_transform(image)
            elif ending in npy_endings:
                ret[folder] = torch.from_numpy(numpy.load(path))
            else:
                raise RuntimeError('Loading from unsupported file type')
        if not self.name is None:
            ret['name'] = self.name
        return ret

    def __len__(self):
        return len(self.data_paths)