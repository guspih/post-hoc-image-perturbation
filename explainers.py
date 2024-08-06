import torch
import numpy as np
import scipy.special

# Attributers (Explainers that calculates attribution)
class OriginalCIUAttributer():
    '''
    Calculates CIU values without intermediate concepts according to the
    py.ciu.image package. Inverse importance is primarily used with inverse
    inverse sampling (see naive_ciu_sampler). Expected utility is used in
    influence calculation to determine which features have positive influence.
    Args:
        inverse (bool): Whether to calculate importance as 1-importance
        expected_util (float/[float]]): Sets the baseline for influence>0
    '''

    def __init__(self, inverse=False, expected_util=0.5):
        self.inverse = inverse
        self.expected_util = expected_util
    
    def __str__(self):
        return f'OriginalCIUAttributer({self.inverse},{self.expected_util})'

    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns (array, array, array, array):
            [M] array with the contextual importance of each feature
            [M] array with the contextual utility of each feature
            [M] array with the influence (ci*(cu-E[cu])) of each feature
            [M,M] array indicating the feature each attribution scores maps to
        '''
        M = Z.shape[1]
        point = 1 if self.inverse else 0
        true_y = Y[np.where(np.all(Z==np.ones(M), axis=-1))]
        min_importance = np.zeros(M)
        max_importance = np.zeros(M)
        min_importance[:] = true_y
        max_importance[:] = true_y
        for y, z in zip(Y,Z):
            point_position = z==point
            nr_points = np.sum(point_position)
            if nr_points > 1 and nr_points < M:
                raise ValueError(
                    'OriginalCIUAttributer require all, none, or exactly one '
                    'feature to be perturbed per sample (or all but one when '
                    'inverse=True)'
                )
            if nr_points != 1:
                continue
            min_importance[point_position & (min_importance>y)] = y
            max_importance[point_position & (max_importance<y)] = y
        importance = (max_importance-min_importance)
        utility = (true_y - min_importance)/importance
        importance = importance/(np.max(Y)-np.min(Y))
        if self.inverse:
            importance = 1-importance
        influence = importance*(utility-self.expected_util)
        return importance, utility, influence, np.eye(M)

class CIUAttributer():
    '''
    Calculates CIU values for each feature combination in return_samples or
    each feature combination in Z if return_samples is None. Importance of a
    feature combination is the maximum difference in Y between the inputs where
    only some or none of the given features have been perturbed. Utility of a
    feature combination is the difference between Y of the original inputs and
    the minumum Y where the inputs where only some or none of the given
    features have been perturbed. Influence is the utility subtracted by the
    expected utility and then scaled by the importance.
    Args:
        expected_util (float/[float]]): Sets the baseline for influence>0
        return_samples (array): [X,M] array indicating features for attribution

    '''
    def __init__(self, expected_util=0.5, return_samples=None):
        self.expected_util = expected_util
        self.return_samples = return_samples

    def __str__(self):
        return f'CIUAttributer({self.expected_util},{self.return_samples})'

    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns (array, array, array, array):
            [X] array with the context importance of given feature combinations
            [X] array with the context utility of given feature combinations
            [X] array with the influence (ci*(cu-E[cu])) of the combinations
            [X,M] array mapping attribution scores to feature combinations
        '''
        M = Z.shape[1]
        if self.return_samples is None:
            unique_Z = np.unique(Z, axis=0)
        else:
            unique_Z = self.return_samples
        true_y = Y[np.where(np.all(Z==np.ones(M), axis=-1))]
        min_importance = np.zeros(unique_Z.shape[0])
        max_importance = np.zeros(unique_Z.shape[0])
        min_importance[:] = 10000
        max_importance[:] = -10000
        for y, z0 in zip(Y,Z):
            for i, z1 in enumerate(unique_Z):
                if np.all(z1*z0 == z1):
                    if min_importance[i] > y: min_importance[i] = y
                    if max_importance[i] < y: max_importance[i] = y
        importance = (max_importance-min_importance)
        utility = (true_y - min_importance)/importance
        importance = importance/(np.max(Y)-np.min(Y))
        influence = importance*(utility-self.expected_util)
        return importance, utility, influence, unique_Z

class SHAPAttributer():
    '''
    Estimates the shapley value attribution of each feature using KernelSHAP.
    The sum of the SHAP values and base value gives the Y value where nothing
    is perturbed. Each SHAP value measures how much the Y value changes if that
    feature is included (as opposed to if it is perturbed).
    '''
    def __str__(self):
        return f'SHAPAttributer()'

    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns (array, float, array):
            SHAP base value, approx. the value if all features are perturbed
            [M] array with the SHAP values of each feature
            [M,M] array mapping attribution scores to features
        '''
        M = Z.shape[1]
        S = Z.sum(axis=1)
        Z = np.concatenate((Z, torch.ones((Z.shape[0], 1))), axis=1)
        S_vals = np.unique(S)
        kernel_vals = np.sqrt(shap_kernel(M, S_vals))
        sqrt_pis = np.zeros(np.max(S_vals)+1)
        sqrt_pis[S_vals] = kernel_vals
        sqrt_pis = sqrt_pis[S]
        shap = np.linalg.lstsq(sqrt_pis[:, None] * Z, sqrt_pis * Y, rcond=None)
        return shap[0][-1], shap[0][:-1], np.eye(M)

class RISEAttributer():
    '''
    Calculates the attribution as is done in the RISE method. Each feature is
    attributed with the average Y value when that feature is not perturbed. To
    recreate full RISE the samples have to be random (see random_sampler).
    '''
    def __str__(self):
        return f'RISEAttributer()'
    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns (array, float, array):
            [M] array with the RISE attribution values of each feature
            [M,M] array mapping attribution scores to features
        '''
        importance = np.sum(Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1))),axis=0)
        occurance = np.sum(Z, axis=0)
        return importance/occurance, np.eye(Z.shape[1])

class LinearLIMEAttributer():
    '''
    Calculates attribution of each feature with LIME by fitting a linear
    surrogate model with the least squares method. The attribution of each
    feature is their weight in the linear surrogate.
    '''
    def __str__(self):
        return f'LinearLIMEAttributer()'
    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns (array, float, array):
            [M] array of the surrogate weights used as attributions
            [M,M] array mapping attribution scores to features
        '''
        Z = np.concatenate((Z, torch.ones((Z.shape[0], 1))), axis=1)
        return np.linalg.lstsq(Z, Y, rcond=None)[0][:-1], np.eye(Z.shape[1])

class PDAAttributer():
    '''
    Estimates the prediction difference analysis of each feature by calculating
    the difference between the true prediction and the average prediction when
    each feature is perturbed. Assumes probability predictions in range [0,1].
    Mode 'probdiff' uses the difference directly, 'infodiff' the difference
    between log2(probability), and 'evidence' the difference between
    log2(probability/(1-probability)).
    Args:
        divide_weight (bool): If True, weight of Y[n] is 1/sum(Z[n])
        mode (str): PDA mode to use ('probdiff', 'infodiff', 'evidence')
        c (int): #classes in Laplace correction of infodiff and evidence
        n (int): Size of training set for correction of infodiff and evidence
    '''
    def __init__(
        self, divide_weight=False, mode='probdiff', c=1000, n=1281167
    ):
        self.divide_weight = divide_weight
        self.mode = mode
        self.c = c
        self.n = n
        modes = ['probdiff','evidence','infodiff']
        if not self.mode in modes:
            raise ValueError(
                f'variant has to be one of {modes} but got {mode}'
            )

    def __str__(self):
        content = f'{self.divide_weight},{self.mode}'
        if not self.mode == 'probdiff': content += f',{self.c},{self.n}'
        return f'PDAAttributer({content})'

    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns (array, float, array):
            [M] array with the PDA values of each feature
            [M,M] array mapping attribution scores to features
        '''
        M = Z.shape[1]
        max_Y = np.max(Y)
        if max_Y > 1.0: #TODO: Consider removing
            Y = Y/max_Y
        true_y = Y[np.where(np.all(Z==np.ones(M), axis=-1))]
        Z = 1-Z
        if self.divide_weight:
            Z = Z/np.sum(Z, axis=1).reshape(-1,1)
        relevance = np.sum(Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1))), axis=0)
        weight = np.sum(Z, axis=0)
        avg_relevance = relevance/weight
        if not self.mode == 'probdiff':
            true_y = (true_y*self.n+1)/(self.n+self.c)
            avg_relevance = (avg_relevance*self.n+1)/(self.n+self.c)
            if self.mode == 'evidence':
                true_y = true_y/(1-true_y)
                avg_relevance = avg_relevance/(1-avg_relevance)
            true_y = np.log2(true_y)
            avg_relevance = np.log2(avg_relevance)
        if len(true_y) == 0: #TODO: Consider removing
            true_y = 0
        elif len(true_y) > 1:
            true_y = true_y[0]
        return true_y-avg_relevance, np.eye(M)


# Explainer utils
def shap_kernel(M, s):
    '''
    Hepler function for KernelSHAP that calculates the weight of a sample using
    the nr of features (M) and the nr of included features in each sample (s).
    Args:
        M (int): The number of features in the sample
        s (array): [N] array of the number of included features in each sample
    Returns (array): [N] array witht the SHAP kernel values for each s given M
    '''
    return np.nan_to_num(
        (M - 1) / (scipy.special.binom(M, s) * s * (M - s)),
        posinf=100000, neginf=100000
    )