import torch
import numpy as np
import itertools

# Attributers (Explainers that calculates attribution)
class OriginalCIUAttributer():
    '''
    Calculates CIU values without intermediate concepts according to the
    py.ciu.image package. Inverse importance is primarily used with inverse
    sampling (see naive_ciu_sampler). Expected utility is used in influence
    calculation to determine which features have positive influence.

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
        Returns:
            array: [M] the contextual importance of each feature
            array: [M] the contextual utility of each feature
            array: [M] the influence (ci*(cu-E[cu])) of each feature
            array: [M,M] map of attribution score to feature
        '''
        M = Z.shape[1]
        point = 1 if self.inverse else 0
        true_y = Y[np.all(Z==1, axis=-1)][0]
        min_importance = np.full(M, true_y)
        max_importance = np.full(M, true_y)
        for y, z in zip(Y,Z):
            if self.inverse:
                y = 1-y
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
        utility = (true_y - min_importance)/(importance+1e-12)
        importance = importance/(np.max(Y)-np.min(Y)+1e-12)
        influence = importance*(utility-self.expected_util)
        return importance, utility, influence, np.eye(M)

class CIUAttributer():
    '''
    Calculates CIU values for each feature combination in return_samples or each
    feature combination in Z if return_samples is None. Importance of a feature
    combination is the maximum difference in Y between the inputs where only
    some or none of the given features have been perturbed. Utility of a feature
    combination is the difference between Y of the original inputs and the
    minumum Y where the inputs where only some or none of the given features
    have been perturbed. Influence is the utility subtracted by the expected
    utility and then scaled by the importance.

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
        Returns:
            array: [X] the contextual importance of given feature combinations
            array: [X] the contextual utility of given feature combinations
            array: [X] the influence (ci*(cu-E[cu])) of the combinations
            array: [X,M] map from attribution scores to feature combinations
        '''
        M = Z.shape[1]
        indices = np.argsort(-np.sum(Z, axis=1))
        Z = Z[indices]
        Y = Y[indices]
        if self.return_samples is None:
            unique_Z = np.unique(Z, axis=0)
        else:
            unique_Z = self.return_samples
        unique_Z = unique_Z[np.argsort(-np.sum(unique_Z, axis=1))]
        true_y = Y[np.all(Z==1, axis=-1)][0]
        min_y = np.full(unique_Z.shape[0], 10000.0)
        max_y = np.full(unique_Z.shape[0], -10000.0)
        for y, z0 in zip(Y,Z):
            for i, z1 in enumerate(unique_Z):
                if np.all(z1*z0 == z1): #TODO: Speedup by using sorted Z
                    if min_y[i] > y: min_y[i] = y
                    if max_y[i] < y: max_y[i] = y
        importance = max_y-min_y
        utility = (true_y - min_y)/(importance+1e-12)
        importance = importance/(np.max(Y)-np.min(Y)+1e-12)
        influence = importance*(utility-self.expected_util)
        return importance, utility, influence, 1-unique_Z

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
        Returns:
            float: SHAP base value, approx. value if all features are perturbed
            array: [M] the SHAP values of each feature
            array: [M,M] map from attribution scores to features
        '''
        Z1 = np.concatenate((Z, torch.ones((Z.shape[0], 1))), axis=1)
        sqrt_pis = shap_kernel(Z1)
        shap = np.linalg.lstsq(sqrt_pis[:, None] * Z1, sqrt_pis * Y, rcond=None)
        return shap[0][-1], shap[0][:-1], np.eye(Z.shape[1])

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
        Returns:
            array: [M] the RISE attribution values of each feature
            array: [M,M] map from attribution scores to features
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
        Returns:
            array: [M] the surrogate weights used as attributions
            array: [M,M] map fromattribution scores to features
        '''
        Z = np.concatenate((Z, torch.ones((Z.shape[0], 1))), axis=1)
        return np.linalg.lstsq(Z, Y, rcond=None)[0][:-1], np.eye(Z.shape[1])

class ScikitLIMEAttributer():
    '''
    Calculates attribution of each feature with LIME by fitting the given
    scikit-learn model (or other model with similar API). The influence of each
    sample is determined by the given kernel. The attribution of each feature is
    their weight in the linear surrogate.

    Args:
        regressor (callable):
        kernel (callable):
    '''
    def __init__(self, regressor, kernel):
        self.regressor = regressor
        self.kernel = kernel

    def __str__(self):
        if hasattr(self.regressor, '__name__'):
            regressor_str = self.regressor.__name__
        else:
            regressor_str = self.regressor.__class__.__name__
        if hasattr(self.kernel, '__name__'):
            kernel_str = self.kernel.__name__
        else:
            kernel_str = self.kernel.__class__.__name__
        return f'ScikitLIMEAttributer({regressor_str},{kernel_str})'

    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns:
            array: [M] the surrogate weights used as attributions
            array: [M,M] map fromattribution scores to features
        '''
        weights = self.kernel(Z)
        self.regressor.fit(Z, Y, sample_weight=weights)
        return (
            self.regressor.intercept_, self.regressor.coef_, np.eye(Z.shape[1])
        )

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
        Returns:
            array: [M] the PDA values of each feature
            array: [M,M] map from attribution scores to features
        '''
        M = Z.shape[1]
        max_Y = np.max(Y)
        if max_Y > 1.0: #TODO: Consider removing
            Y = Y/max_Y
        true_y = Y[np.all(Z==1, axis=-1)]
        true_y_exist = len(true_y) > 0
        true_y = true_y if true_y_exist else 0
        X = 1-Z
        if self.divide_weight:
            X = X/np.sum(X, axis=1).reshape(-1,1)
        relevance = np.sum(X*(Y.reshape(list(Y.shape)+[1]*(X.ndim-1))), axis=0)
        weight = np.sum(X, axis=0)
        avg_relevance = relevance/weight
        if not self.mode == 'probdiff':
            true_y = (true_y*self.n+1)/(self.n+self.c)
            avg_relevance = (avg_relevance*self.n+1)/(self.n+self.c)
            if self.mode == 'evidence':
                true_y = true_y/(1-true_y)
                avg_relevance = avg_relevance/(1-avg_relevance)
            true_y = np.log2(true_y)
            avg_relevance = np.log2(avg_relevance)
            true_y = true_y if true_y_exist else 0
        return true_y-avg_relevance, np.eye(M)

class ExplainerAttributer():
    '''
    Wrapper that uses one attributer to attribute the influence values of
    another attributer.

    Args:
        explainer (callable): An attributer to explain the input data
        attribution_explainer (callable): An Attributer to explain the explainer
    '''
    def __init__(self, explainer, attribution_explainer):
        self.explainer = explainer
        self.attribution_explainer = attribution_explainer

    def __str__(self):
        content = f'{self.explainer},{self.attribution_explainer}'
        return f'ExplainerAttributer({content})'

    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns:
            any, optional:   Any number of outputs from attribution_explainer
            array: [X] The attribution scores of explainees to attributions
            array: [X,M] map of attribution score to feature combinations
        '''
        explanation = self.explainer(Y, Z)
        return(self.attribution_explainer(explanation[-2], explanation[-1]))

class OrderedAttributer():
    '''
    Wrapper that takes an attributer and then assigns each feature attribution
    in even steps between 0 and 1 according to the ascending sorting of the
    attributions by the wrapped attributer

    Args:
        explainer (callable): An attributer to explain the input data
    '''
    def __init__(self, explainer):
        self.explainer = explainer

    def __str__(self):
        return f'OrderedAttributer({self.explainer})'

    def __call__(self, Y, Z):
        explanation = self.explainer(Y, Z)
        sidx = np.argsort(explanation[-2])
        idx = np.concatenate(
            ([0],np.flatnonzero(np.diff(explanation[-2][sidx]))+1,
            [explanation[-2].size])
        )
        o = np.repeat(idx[:-1],np.diff(idx))[sidx.argsort()]
        o = o/np.max(o)
        return o, explanation[-1]

# Explainer utils
def shap_kernel(Z):
    '''
    Hepler function for KernelSHAP that calculates the weight of the samples (Z)
    using the nr of features and the nr of included features in each sample.

    Args:
        Z (array): [N,M] array indicating which features are included (=1)
    Returns:
        array: [N] the SHAP kernel weight for each sample
    '''
    import scipy.special
    M = Z.shape[1]
    S = Z.sum(axis=1).astype(int)
    S_vals = np.unique(S)
    kernel_vals = np.nan_to_num(
        (M - 1) / (scipy.special.binom(M, S_vals) * S_vals * (M - S_vals)),
        posinf=100000, neginf=100000
    )
    sqrt_pis = np.zeros(np.max(S_vals)+1)
    sqrt_pis[S_vals] = kernel_vals
    sqrt_pis = sqrt_pis[S]
    return sqrt_pis


##### TEST CODE BELOW #####


class CIUPlusAttributer1():
    '''
    Calculates the CIU values for each feature by calculating the them for each
    combination of features in in Z and distributing the excess importance of
    combinations among their constituent pieces. The importance of combinations
    is calucated using CIUAttributer.

    Args:
        expected_util (float/[float]]): Sets the baseline for influence>0
        return_samples (array): [X,M] array indicating features for attribution
    '''
    def __init__(self, expected_util=0.5, return_samples=None):
        self.expected_util = expected_util
        self.return_samples = return_samples

    def __str__(self):
        return f'CIUPlusAttributer({self.expected_util},{self.return_samples})'

    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns:
            array: [M] the contextual importance of each feature
            array: [M] the contextual utility of each feature
            array: [M] the influence (ci*(cu-E[cu])) of each feature
            array: [M,M] map of attribution score to feature
        '''
        M = Z.shape[1]
        indices = np.argsort(-np.sum(Z, axis=1))
        Z = Z[indices]
        Y = Y[indices]
        if self.return_samples is None:
            unique_Z = np.unique(Z, axis=0)
        else:
            unique_Z = self.return_samples
        unique_Z = unique_Z[np.argsort(-np.sum(unique_Z, axis=1))]
        true_y = Y[np.all(Z==1, axis=-1)][0]
        min_y = np.full(unique_Z.shape[0], 10000.0)
        max_y = np.full(unique_Z.shape[0], -10000.0)
        for y, z0 in zip(Y,Z):
            for i, z1 in enumerate(unique_Z):
                if np.all(z1*z0 == z1): #TODO: Speedup by using sorted Z
                    if min_y[i] > y: min_y[i] = y
                    if max_y[i] < y: max_y[i] = y
        excess_min = np.full(M, 10000.0)
        excess_max = np.full(M, -10000.0)
        combo_min = np.zeros(M)
        combo_max = np.zeros(M)
        new_min = min_y.copy()
        new_max = max_y.copy()
        current_z_sum = 1
        for i, (mn, mx, z0) in enumerate(zip(min_y, max_y, unique_Z)):
            new_mn, new_mx = new_min[i], new_max[i]
            idxs = z0==0
            z_sum = np.sum(idxs)
            if z_sum == 0:
                continue
            if z_sum > current_z_sum:
                current_z_sum = z_sum
                combo_max += excess_max*(excess_max>-10000.0)
                combo_min += excess_min*(excess_min<10000.0)
                excess_max[excess_max>-10000.0] = 0.0
                excess_min[excess_min<10000.0] = 0.0
            excess_mx = new_mx/z_sum
            excess_mn = new_mn/z_sum # TODO: Make min and max move to true_y
            excess_max[idxs & (excess_max < excess_mx)] = excess_mx
            excess_min[idxs & (excess_min > excess_mn)] = excess_mn
            for j, z1 in enumerate(unique_Z):
                z1_sum = M-np.sum(z1)
                if z1_sum <= z_sum:
                    continue
                if not np.all(z0*z1 == z1):
                    continue
                new_max[j] = max(0, min(new_max[j], max_y[j]-mx))
                new_min[j] = min(0, max(new_min[j], mn-min_y[j]))
        combo_max += excess_max
        combo_min += excess_min
        importance = combo_max-combo_min
        utility = (true_y - combo_min)/(importance+1e-12)
        importance = importance/(np.max(Y)-np.min(Y)+1e-12)
        influence = importance*(utility-self.expected_util)
        return importance, utility, influence, np.eye(M)

class SignificanceAttributer():
    '''
    Calculates the p-values for the distribution of y-values being greater/
    lesser when the features (or feature coalitions) are included/occluded.

    Args:
        mode ('include','occlude','both'): What distribution to consider
        power (int): What size of coalitions to consider (1= indvidual features)
        other ('subset','all','inverse'): Which distributions to compare to
    '''

    def __init__(self, mode='include', other='subset', power=1):
        from scipy.stats import ttest_ind
        assert mode in ['include', 'occlude', 'both']
        assert other in ['subset', 'all', 'inverse']
        self.ttest_ind = ttest_ind
        self.mode = mode
        self.other = other
        self.power = power

    def __str__(self):
        return f'SignificanceAttributer({self.mode},{self.other},{self.power})'

    def __call__(self, Y, Z):
        '''
        Args:
            Y (array): [N] array of all model values for the perturbed inputs
            Z (array): [N,M] array indicating which features were perturbed (0)
        Returns:
            array: [X] liklihoods that the coaltions are are greater than other
            array: [X,M] map of attribution score to feature coaltions
        '''
        combos = itertools.combinations(range(Z.shape[1]), r=self.power)
        combo_p = dict([(combo, 0) for combo in combos])

        if self.other == 'subset':
            others = itertools.combinations(range(Z.shape[1]), r=self.power-1)
            others = dict(
                [(subset, Y[np.all(Z[:,subset], axis=1)]) for subset in others]
            )
        elif self.others == 'all':
            others = {(): Y}
        elif self.others == 'inverse':
            others = dict(
                [(combo, Y[np.all(Z[:,combo]==0, axis=1)]) for combo in combos]
            )

        for combo in combo_p.keys():
            combo_ys = Y[np.all(Z[:,combo], axis=1)]

            if self.other == 'subset':
                other_idxs = [
                    tuple(x for j,x in enumerate(combo) if j!=i)
                    for i in range(len(combo))
                ]
            elif self.other == 'all':
                other_idxs = [()]
            elif self.other_idxs == 'inverse':
                other_idxs = [combo]

            for other_idx in other_idxs:
                other_ys = others[other_idx]
                new_p = self.ttest_ind(
                    combo_ys, other_ys, axis=0, alternative='greater'
                ).pvalue
                combo_p[combo] = 1-(1-combo_p[combo])*(1-new_p)

        p_values = 1-np.array(list(combo_p.values()))
        coalition_map = np.zeros((len(combo_p), Z.shape[1]))
        for i, combo in enumerate(combo_p.keys()):
            coalition_map[i, combo] = 1
        return p_values, coalition_map

class TestCombinationAttributer():
    '''
    Calculates a bunch of values for combinations of features to assess if there
    is a good way to auomatically find features that group together.
    '''

    def __init__(self):
        self.storage = {}

    def __str__(self):
        return f'TestCombinationAttributer(mono_significance)'

    def __call__(self, Y, Z):
        #self.storage = {}
        #for z, y in zip(Z, Y):
        #    self.storage[tuple(z)] = y

        #return self.significance_grouping(Y,Z,p_value=0.05), np.eye(Z.shape[1])
        return self.mono_significance(Y, Z), np.eye(Z.shape[1])

    def mono_influence(self, Y, Z):
        inv_Z = 1-Z
        included = np.sum(Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1))), axis=0)
        occluded = np.sum(inv_Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1))), axis=0)
        occurance = np.sum(Z, axis=0)
        nocurance = np.sum(inv_Z, axis=0)
        return included/occurance, occluded/nocurance

    def combo_influence(self, Y, Z, power=2):
        empty = np.zeros(Z.shape[1])
        combos = itertools.combinations(range(Z.shape[1]), r=power)
        included = dict([(combo, 0) for combo in combos])
        occluded = dict([(combo, 0) for combo in combos])
        occurance = dict([(combo, 0) for combo in combos])
        nocurance = dict([(combo, 0) for combo in combos])
        inv_Z = 1-Z
        for z, y in zip(Z,Y):
            for combo in combos:
                if all(z[combo]):
                    included[combo] = included[combo] + y
                    occurance[combo] = occurance[combo] + 1
                elif all(1-z[combo]):
                    occluded[combo] = occluded[combo] + y
                    nocurance[combo] = nocurance[combo] + 1
        for combo in combos:
            included[combo] = included[combo]/occurance[combo]
            occluded[combo] = occluded[combo]/nocurance[combo]
        return included, occluded

    def mono_significance(self, Y, Z, p_value=0.005):
        import scipy.stats
        inv_Z = 1-Z
        included = Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1)))
        occluded = inv_Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1)))

        included[Z==0] = np.nan
        occluded[Z==1] = np.nan

        pvalue = scipy.stats.ttest_ind(
            included, occluded, axis=0, nan_policy='omit', alternative='greater'
        ).pvalue
        #print(pvalue)
        is_significant = pvalue <= p_value

        return -np.log10(pvalue)

    def significance_grouping(self, Y, Z, p_value=0.01):
        # Adding segments until no longer significant improvement
        import scipy.stats
        inv_Z = 1-Z
        included = Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1)))
        occluded = inv_Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1)))

        singles = included.copy()
        singles[Z==0] = np.nan
        single_means = np.nanmean(singles, axis=0)
        singles_order = np.argsort(single_means)[::-1]

        significant_group = np.zeros(Z.shape[1])
        test_group = np.zeros(Z.shape[1])
        test_group[singles_order[0]] = 1
        for i in range(Z.shape[1]-1):
            significant_group[singles_order[i]] = 1
            combo_ys = Y[np.all(Z[:, significant_group.astype(bool)], axis=1)]
            test_group[singles_order[i+1]] = 1
            next_ys = Y[np.all(Z[:, test_group.astype(bool)], axis=1)]
            p = scipy.stats.ttest_ind(
                next_ys, combo_ys, axis=0, nan_policy='omit',
                alternative='greater'
            ).pvalue
            if p > p_value:
                break

        return significant_group


    def combo_significance(self, Y, Z, power=2, p_value=0.01):
        # combo vs. single
        import scipy.stats
        inv_Z = 1-Z
        included = Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1)))
        occluded = inv_Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1)))

        singles = included.copy()
        singles[Z==0] = np.nan

        combos = itertools.combinations(range(Z.shape[1]), r=power)
        combo_p = dict([(combo, 1) for combo in combos])
        for combo in combo_p.keys():
            combo_ys = Y[np.all(Z[:,combo], axis=1)]
            pvalue = 0
            for single in combo:
                single_p = scipy.stats.ttest_ind(
                        combo_ys, singles[:, single], axis=0, nan_policy='omit',
                        alternative='greater'
                    ).pvalue
                print('!', single_p, pvalue)
                pvalue = np.max((single_p, pvalue))
            combo_p[combo] = pvalue

        print(min(combo_p.values()))
        center = np.argmax(self.mono_significance(Y, Z))
        significant_group = np.full(Z.shape[1],0.1)
        significant_group[center] = 1
        for i in range(Z.shape[1]):
            if i == center:
                continue
            key = (center, i) if i>center else (i, center)
            #print(combo_p[key])
            if combo_p[key] < p_value:
                significant_group[i] = 1

        return significant_group

    def combo_signficance1(self, Y, Z, power=2):
        #combo vs. singles-combo (probably bad if one component is insignificant)
        combos = itertools.combinations(range(Z.shape[1]), r=power)
        double_y = dict([(combo, 0) for combo in combos])
        single_y = dict([(combo, 0) for combo in combos])
        double_n = dict([(combo, 0) for combo in combos])
        single_n = dict([(combo, 0) for combo in combos])
        inv_Z = 1-Z
        for z, y in zip(Z,Y):
            for combo in combos:
                if all(z[combo]):
                    double_y[combo] = double_y[combo] + y
                    double_n[combo] = double_n[combo] + 1
                elif any(z[combo]):
                    single_y[combo] = single_y[combo] + y
                    single_n[combo] = single_n[combo] + 1
        for combo in combos:
            double_y[combo] = double_y[combo]/double_n[combo]
            single_y[combo] = single_y[combo]/single_n[combo]
        return double_y, single_y

    def feature_support(self, Y, Z):
        avg_included, avg_occluded = self.mono_influence(Y, Z)
        top_z = np.argmax(avg_included-avg_occluded)
        bottom_mask = Z[:,top_z] == 0
        print(bottom_mask)
        avg_included, avg_occluded2 = self.mono_influence(
            Y[bottom_mask], Z[bottom_mask, :]
        )
        avg_included[top_z] = 1
        avg_occluded[top_z] = 0

        return avg_included > avg_occluded, np.eye(Z.shape[1])
