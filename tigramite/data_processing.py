"""Tigramite data processing functions."""

# Authors: Jakob Runge <jakob@jakob-runge.com>
#          Andreas Gerhardus <andreas.gerhardus@dlr.de>
# License: GNU General Public License v3.0

from __future__ import print_function
from collections import defaultdict, OrderedDict
import sys
import warnings
from copy import deepcopy
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy import stats
from numba import jit
from sklearn.preprocessing import StandardScaler

#TODO pep8 style conventions; ie spaces around comparison etc

index_codes = {'x' : 0,
               'y' : 1,
               'z' : 2,
               'e' : 3}

def get_var(varlag):
    return varlag[0]
def get_lag(varlag):
    return varlag[1]

"""The following functions are for manipuating data extracted from DataFrame.
Data is stored in the usual way, with shape (time, N variables). This is 
consistent with the orientation expected by external processors such as sklearn.
However, data returned from construct_array is of shape (n<=N variables, time) for 
computational efficiency. Efficient_representation=True should be used with 
output of construct_array.
"""

def _select_variables(A, I=None, efficient_representation=False, lags=None):
    """Select data variables of array A using indices I. 
    When efficient_representation=False, each column contains a different variable, 
    as is standard for data storage. This is used for tigramite dataframe data 
    storage and for external processors like sklearn. When efficient_representation=True, 
    each row contains a different variable. Efficient should be set to True when using 
    the output of construct_array.
    """ 
    if not efficient_representation:
        A = A.T
    #Now A is in the efficient representation, (n variables, n samples)
    if I is not None:
        A = A[I,:]
    if lags is not None:
        lags = np.array(lags)
        if np.any(lags > 1):
            raise ValueError("lags must be non-positive")
        num_vars = _get_num_variables(A, efficient_representation=True)
        if len(lags) != num_vars:
            raise ValueError("lags has incompatible length")
        A_old = deepcopy(A)
        new_sample_length = _get_num_samples(A, efficient_representation=True)+min(lags)
        A = np.empty((num_vars, new_sample_length))
        start = lags - np.min(lags)
        end   = new_sample_length+lags
        for i, row in enumerate(A_old):
            A[i,:] = row[start[i]:end[i]]
    if not efficient_representation:
        A = A.T
    #now A is back in its original representation
    return A

def _get_num_variables(A, efficient_representation=False):
    if efficient_representation:
        return A.shape[0]
    else:
        return A.shape[1]
    
def _select_samples(A, I=None, efficient_representation=False):
    return _select_variables(A, I, efficient_representation = not efficient_representation)

def _get_num_samples(A, efficient_representation=False):
    return _get_num_variables(A, efficient_representation = not efficient_representation)

#This function selects data variables using indices I and 
#transposes to be in the correct orientation for sklearn.
#Inverse action appears below.
def _to_sklearn(A, I=None):
    return _select_variables(A, I=I, efficient_representation=True).T

#inverse of the transpose above. 
def _from_sklearn(A):
    return A.T

#for caching information that might change when a decomposition is applied for a data transform
transform_attrs = ['values',         #dict by dataset
                   'Ndata',          #int                    Update at end from vector_lengths
                   'vector_vars',    #dict                   Update at end from vector_lengths
                   'vector_lengths', #np.array of integers
                   'T'               #dictionary by dataset
                  ]

class DataFrame():
    """Data object containing single or multiple time series arrays and optional 
    mask, as well as variable definitions.

    Parameters
    ----------
    data : array-like
        if analysis_mode == 'single':
         Numpy array of shape (observations T, variables N)
         OR
         Dictionary with a single entry whose value is a numpy array of
         shape (observations T, variables N)
        if analysis_mode == 'multiple':
         Numpy array of shape (multiple datasets M, observations T,
         variables N)
         OR
         Dictionary whose values are numpy arrays of shape
         (observations T_i, variables N), where the number of observations
         T_i may vary across the multiple datasets but the number of variables
         N is fixed. 
    mask : array-like, optional (default: None)
        Optional mask array, must be of same format and shape as data.
    data_type : array-like
        Binary data array of same shape as array which describes whether 
        individual samples in a variable (or all samples) are continuous 
        or discrete: 0s for continuous variables and 1s for discrete variables.
    missing_flag : number, optional (default: None)
        Flag for missing values in dataframe. Dismisses all time slices of
        samples where missing values occur in any variable. For
        remove_missing_upto_maxlag=True also flags samples for all lags up to
        2*tau_max (more precisely, this depends on the cut_off argument in
        self.construct_array(), see further below). This avoids biases, see
        section on masking in Supplement of Runge et al. SciAdv (2019).
    vector_vars : dict or integer list 
        Dictionary of vector variables of the form,
        Eg. {0: [(0, 0), (1, 0)], 1: [(2, 0)], 2: [(3, 0)], 3: [(4, 0)]}
        The keys are the new vectorized variables and respective tuple values
        are the individual components of the vector variables. In the method of
        construct_array(), the individual components are parsed from vector_vars
        and added (accounting for lags) to the list that creates X, Y and Z for
        conditional independence test.
        OR
        List of integer vector-lengths for automatic creation of vector_vars. 
        assumes vectors do not overlap, appear in order in the dataframe, and use only lag=0
    var_names : list of strings, optional (default: range(N))
        Names of variables, must match the number of variables. If None is
        passed, variables are enumerated as [0, 1, ...]
    datatime : array-like, optional (default: None)
        Timelabel array. If None, range(T) is used.
    remove_missing_upto_maxlag : bool, optional (default: False)
        Whether to remove not only missing samples, but also all neighboring
        samples up to max_lag (as given by cut_off in construct_array).
    analysis_mode : string, optional (default: 'single')
        Must be 'single' or 'multiple'.
        Determines whether data contains a single (potentially multivariate)
        time series (--> 'single') or multiple time series (--> 'multiple').
    reference_points : None, int, or list (or 1D array) of integers,
        optional (default:None)
        Determines the time steps --- relative to the shared time axis as
        defined by the optional time_offset argument (see below) --- that are
        used to create samples for conditional independence testing.
        Set to [0, 1, ..., T_max-1] if None is passed, where T_max is
        self.largest_time_step, see below.
        All values smaller than 0 and bigger than T_max-1 will be ignored.
        At least one value must be in [0, 1, ..., T_max-1].
    time_offsets : None or dict, optional (default: None)
        if analysis_mode == 'single':
         Must be None.
         Shared time axis defined by the time indices of the single time series
        if analysis_mode == 'multiple' and data is numpy array:
         Must be None.
         All datasets are assumed to be already aligned in time with
         respect to a shared time axis, which is the time axis of data
        if analysis_mode == 'multiple' and data is dictionary:
         Must be dictionary of the form {key(m): time_offset(m), ...} whose
         set of keys agrees with the set of keys of data and whose values are
         non-negative integers, at least one of which is 0. The value
         time_offset(m) defines the time offset of dataset m with
         respect to a shared time axis.
    transform: sklearn preprocessing or decomposition object, optional (default: None)
        Used to transform data prior to analysis. For example,
        sklearn.preprocessing.StandardScaler for simple standardization. 
        sklearn.pipeline.Pipeline can be used for sequential transformations. 
        Additional custom sklearn-based transformations can be found in 
        data_processing.py under ###Custom sklearn-based transformations###. 
        For vectorized data, tigramite.data_processing.StandardTotalVarianceScaler
        is recommended. The fitted parameters are stored. 

    Attributes
    ----------
    self._initialized_from : string
        Specifies the data format in which data was given at instantiation.
        Possible values: '2d numpy array', '3d numpy array', 'dict'.
    self.values : dictionary
        Dictionary holding the observations given by data internally mapped to a
        dictionary representation as follows:
        If analysis_mode == 'single': for self._initialized_from == '2d numpy array' this
        is {0: data} and for self._initialized_from == 'dict' this is data.
        If analysis_mode == 'multiple': If self._initialized_from == '3d numpy array', this is
        {m: data[m, :, :] for m in range(data.shape[0])} and for self._initialized_from == 'dict' this
        is data.
    self.datasets : list
        List of the keys identifiying the multiple datasets, i.e.,
        list(self.values.keys())
    self.data_transform : sklearn preprocessing or decomposition object
        Is transform
    self.fitted_transform : dictionary
        #TODO remake using sklearn ColumnTransformer once it's updated
        Is {var : deepcopy(self.data_transform).fit(var_data) for var in self.vector_vars.keys()}
        where data is data for that variable from each dataset concatenated together, with appropriate lags
    self.original_data : dictionary
        Untransformed data and data attributes
    self.mask : dictionary
        Mask internally mapped to a dictionary representation in the same way as
        data is mapped to self.values
    self.data_type : array-like
        Binary data array of same shape as array which describes whether 
        individual samples in a variable (or all samples) are continuous 
        or discrete: 0s for continuous variables and 1s for discrete variables.
    self.missing_flag:
        Is missing_flag
    self.var_names:
        If var_names is not None:
            Is var_names
        If var_names is None:
            Is {i: i for i in range(self.N)}
    self.vector_vars:
        If vector_vars is None:
            Is {i: (i, 0) for i in range(self.N)}
        elif vector_vars is a dictionary:
            Is vector_vars
        elif vector_vars is an integer list:
            dictionary created internally assuming vectors do not overlap and span 
            the dataframe, appear in order in the dataframe, and use only lag=0
    self.vector_lengths:
        If vector_vars is an integer list:
            Is np.array(vector_vars)
        elif vector_vars is a dictionary:
            np.array containing the length of each vector
        elif vector_vars is None:
            an array of 1s for non-vectorized data
    self.has_vector_data:
        True if vector_vars is not None, else False
    self.datatime : dictionary
        Time axis for each of the multiple datasets.
    self.analysis_mode : string
        Is analysis_mode
    self.reference_points: array-like
        If reference_points is not None:
            1D numpy array holding all specified reference_points, less those
            smaller than 0 and larger than self.largest_time_step-1
        If reference_points is None:
            Is np.array(self.largest_time_step)
    self.reference_points_is_none: boolean
        Is self.reference_points is None
    self.time_offsets : dictionary
        If time_offsets is not None:
            Is time_offsets
        If time_offsets is None:
            Is {key: 0 for key in self.values.keys()}
    self.time_offsets_is_none : boolean
        Is self.time_offsets is None
    self.M : int
        Number of datasets
    self.N : int
        Number of variables (constant across datasets)
    self.Ndata : int
        Number of input variables in dataframe
    self.T : dictionary
        Dictionary {key(m): T(m), ...}, where T(m) is the time length of
        datasets m and key(m) its identifier as in self.values
    self.largest_time_step : int
        max_{0 <= m <= M} [ T(m) + time_offset(m)], i.e., the largest (latest)
        time step relative to the shared time axis for which at least one
        observation exists in the dataset.
    self.smallest_time_step : int TODO Jakob check this
        min_{0 <= m <= M} [ T(m) + time_offset(m)], i.e., the smalest (earliest)
        time step relative to the shared time axis for which at least one
        observation exists in the dataset.
    self.bootstrap : dictionary
        Whether to use bootstrap. Must be a dictionary with keys random_state,
        boot_samples, and boot_blocklength.
    self.remove_missing_upto_maxlag #TODO Jakob
    """

    def __init__(self, data, mask=None, missing_flag=None, vector_vars=None, var_names=None,
        data_type=None, datatime=None, analysis_mode ='single', reference_points=None,
        time_offsets=None, remove_missing_upto_maxlag=False, transform=None):

        # Check that a valid analysis mode, specified by the argument
        # 'analysis_mode', has been chosen
        if analysis_mode in ['single', 'multiple']:
            self.analysis_mode = analysis_mode
        else:
            raise ValueError("'analysis_mode' is '{}', must be 'single' or "\
                "'multiple'.".format(analysis_mode))
            
        # Check for correct type and format of 'data', internally cast to the
        # analysis mode 'multiple' case in dictionary representation
        if self.analysis_mode == 'single':
            # In this case the 'time_offset' functionality must not be used
            if time_offsets is not None:
                raise ValueError("'time_offsets' must be None in analysis "\
                    "mode'single'.")

            # 'data' must be either
            # - np.ndarray of shape (T, N)
            # - np.ndarray of shape (1, T, N)
            # - a dictionary with one element whose value is a np.ndarray of
            # shape (T, N)
            
            if isinstance(data, np.ndarray):
                _data_shape = data.shape
                if len(_data_shape) == 2:
                    self.values = {0: np.copy(data)}
                    self._initialized_from = "2d numpy array"
                elif len(_data_shape) == 3 and _data_shape[0] == 1:
                    self.values = {0: np.copy(data[0, :, :])}
                    self._initialized_from = "3d numpy array"
                else:
                    raise TypeError("In analysis mode 'single', 'data' given "\
                        "as np.ndarray. 'data' is of shape {}, must be of "\
                        "shape (T, N) or (1, T, N).".format(_data_shape))

            elif isinstance(data, dict):
                if len(data) == 1:
                    _data = next(iter(data.values()))
                    if isinstance(_data, np.ndarray):
                        if len(_data.shape) == 2:
                            self.values = data.copy()
                            self._initialized_from = "dict"
                        else:
                            raise TypeError("In analysis mode 'single', "\
                                "'data'given as dictionary. The single value "\
                                "is a np.ndarray of shape {}, must be of "\
                                "shape (T, N).".format(_data.shape))
                    else:
                        raise TypeError("In analysis mode 'single', 'data' "\
                            "given as dictionary. The single value is of type "\
                            "{}, must be np.ndarray.".format(type(_data)))

                else:
                    raise ValueError("In analysis mode 'single', 'data' given "\
                        "as dictionary. There are {} entries in 'data', there "\
                        "must be exactly one entry.".format(len(data)))

            else:
                raise TypeError("In analysis mode 'single'. 'data' is of type "\
                    "{}, must be np.ndarray or dict.".format(type(data)))

        elif self.analysis_mode == 'multiple':
            # 'data' must either be a
            # - np.ndarray of shape (M, T, N)
            # - dict whose values of are np.ndarray of shape (T_i, N), where T_i
            # may vary across the values

            if isinstance(data, np.ndarray):
                _data_shape = data.shape
                if len(_data_shape) == 3:
                    self.values = {i: np.copy(data[i, :, :]) for i in range(_data_shape[0])}
                    self._initialized_from = "3d numpy array"
                else:
                    raise TypeError("In analysis mode 'multiple', 'data' "\
                        "given as np.ndarray. 'data' is of shape {}, must be "\
                        "of shape (M, T, N).".format(_data_shape))

                # In this case the 'time_offset' functionality must not be used
                if time_offsets is not None:
                    raise ValueError("In analysis mode 'multiple'. Since "\
                        "'data' is given as np.ndarray, 'time_offsets' must "\
                        "be None.")

            elif isinstance(data, dict):
                _N_list = set()
                for dataset_key, dataset_data in data.items():
                    if isinstance(dataset_data, np.ndarray):
                        _dataset_data_shape = dataset_data.shape
                        if len(_dataset_data_shape) == 2:
                            _N_list.add(_dataset_data_shape[1])
                        else:
                            raise TypeError("In analysis mode 'multiple', "\
                                "'data' given as dictionary. 'data'[{}] is of "\
                                "shape {}, must be of shape (T_i, N).".format(
                                    dataset_key, _dataset_data_shape))

                    else:
                        raise TypeError("In analysis mode 'multiple', 'data' "\
                            "given as dictionary. 'data'[{}] is of type {}, "\
                            "must be np.ndarray.".format(dataset_key,
                                type(dataset_data)))

                if len(_N_list) == 1:
                    self.values = data.copy()
                    self._initialized_from = "dict"
                else:
                    raise ValueError("In analysis mode 'multiple', 'data' "\
                        "given as dictionary. All entries must be np.ndarrays "\
                        "of shape (T_i, N), where T_i may vary across the "\
                        "entries while N must not vary. In the given 'data' N "\
                        "varies.")

            else:
                raise TypeError("In analysis mode 'multiple'. 'data' is of "\
                    "type {}, must be np.ndarray or dict.".format(type(data)))

        # Store the keys of the datasets in a separated attribute
        self.datasets = list(self.values.keys())

        # Save the data format and check for NaNs:
        self.M = len(self.values) # (Number of datasets)

        self.T = dict() # (Time lengths of the individual datasets)
        for dataset_key, dataset_data in self.values.items():
            if np.isnan(dataset_data).sum() != 0:
                raise ValueError("NaNs in the data.")

            _dataset_data_shape = dataset_data.shape
            self.T[dataset_key] = _dataset_data_shape[0]
            self.Ndata = _dataset_data_shape[1] # (Number of variables) 
            # N does not vary across the datasets
            #TODO then why is it in the for loop?! shouldn't it be a check rather than setting it potentially again and again?

        self.has_vector_data = (vector_vars is not None)  
        
        def check_type(lst, tp, msg):
            check = [isinstance(x, tp) for x in lst]
            if not all(check):
                typ = next(type(x) for x in lst if not isinstance(x, tp))
                raise TypeError(msg.format(typ))
        
        # Setup dictionary of variables for vector mode
        if vector_vars is not None:
            if isinstance(vector_vars, dict):
                check_type(vector_vars.keys(), int, "keys in vector_vars dictionary must be integers, not {}")
                if not all([k>=0 for k in vector_vars.keys()]):
                    raise ValueError("keys in vector_vars dictionary must be non-negative")
                check_type([v for _, v in vector_vars.items()], list, "values in vector_vars dictionary must be lists, not {}")
                check_type([e for _, v in vector_vars.items() for e in v], tuple, "values in vector_vars dictionary must be lists of tuples, not lists of {}")
                check_type([e for _, v in vector_vars.items() for t in v for e in t], int, "every element in every tuple in the vector_vars dictionary must be an integer, not {}")
                if not all([n[0]>=0 and n[0]<self.Ndata for _, v in vector_vars.items() for n in v]):
                    raise ValueError("the first element in each tuple in the vector_vars dictionary must be an integer between 0 and {} (the last input variable)".format(self.Ndata-1))
                if not all([n[1]<=0 for _, v in vector_vars.items() for n in v]):
                    raise ValueError("the second element in each tuple in the vector_vars dictionary must be a non-positive integer")
                nodes_w_duplicates = sum([len(set(v))<len(v) for _, v in vector_vars.items()])
                if nodes_w_duplicates > 0:
                    raise ValueError("{} vector nodes contain duplicate input variables".format(nodes_w_duplicates))
                v_lst = [e[0] for _, v in vector_vars.items() for e in v]
                sl = len(set(v_lst))
                vl = len(v_lst)
                if sl<vl:
                    msg = "Some input variables appear in multiple vector nodes! There are {} overlaps of input variables in the vectorized dataframe"
                    warnings.warn(msg.format(vl-sl))
                if sl<self.Ndata:
                    msg = "{} input variables are not included in any vector nodes!"
                    warnings.warn(msg.format(self.Ndata-sl))
                self.vector_vars = vector_vars
                self.vector_lengths = np.array([len(self.vector_vars[k]) for k in self.vector_vars.keys()])
            elif isinstance(vector_vars, list):
                if not all(isinstance(n, int) for n in vector_vars):
                    t = next(type(x) for x in vector_vars if not isinstance(x, int))
                    raise TypeError("vector_vars does not accept a list of {}; please input a dictionary or a list of integers.".format(t))
                svl=sum(vector_vars)
                if svl!=self.Ndata:
                    raise ValueError("VECTOR_LENGTHS don't span the dataset, covering only {} of {} input variables".format(svl, self.Ndata))
                self.vector_lengths = np.array(vector_vars)
                self.vector_vars = self.make_vector_node_dict(vector_vars)
            else:
                raise TypeError("vector_vars must be a dictionary of (var, lag) pairs or an integer list, not type {}".format(type(vector_vars)))
        else:
            self.vector_vars = dict(zip(range(self.Ndata), [[(i, 0)] 
                                for i in range(self.Ndata)]))
            self.vector_lengths = np.array([1]*self.Ndata)
        
        self.N = len(self.vector_vars)
        
        #cache original data
        self.original_data = {attr : deepcopy(self.__getattribute__(attr)) for attr in transform_attrs}
        #apply the transform; overwrites the variables above, consulting cached information each time it's called.
        self.update_transform(transform)
        
        # Warnings
        if self.analysis_mode == 'single' and self.N > next(iter(self.T.values())):
            warnings.warn("In analysis mode 'single', 'data'.shape = ({}, {});"\
                " is it of shape (observations, variables)?".format(self.T[0],
                    self.N))

        if self.analysis_mode == 'multiple' and self.M == 1:
            warnings.warn("In analysis mode 'multiple'. There is just a "\
                "single dataset, is this as intended?'")
        
        # Save the variable names. If unspecified, use the default
        if var_names is None:
            self.var_names = {i: i for i in range(self.N)}
        else:
            _len_var_names = len(var_names)
            if(_len_var_names != self.N):
                raise ValueError("{} var_names provided, but data has {} variables".format(_len_var_names, self.N))
            self.var_names = var_names

        self.mask = None
        if mask is not None:
            self.mask = self._check_mask(mask = mask)
            
        self.data_type = None
        if data_type is not None:
            self.data_type = self._check_mask(mask = data_type, check_data_type=True)

        # Check and prepare the time offsets
        self._check_and_set_time_offsets(time_offsets)
        self.time_offsets_is_none = time_offsets is None

        # Set the default datatime if unspecified
        if datatime is None:
            self.datatime = {m: np.arange(self.time_offsets[m], 
                self.time_offsets[m] + self.T[m]) for m in self.values.keys()}
        else:
            if not isinstance(datatime, dict):
                self.datatime = {0: datatime}   
            else:
                self.datatime = datatime

        # Save the largest/smallest relevant time step
        self.largest_time_step = np.add(np.asarray(list(self.T.values())), np.asarray(list(self.time_offsets.values()))).max()
        self.smallest_time_step = np.add(np.asarray(list(self.T.values())), np.asarray(list(self.time_offsets.values()))).min()

        # Check and prepare the reference points
        self._check_and_set_reference_points(reference_points)
        self.reference_points_is_none = reference_points is None

        # Save the 'missing_flag' value
        self.missing_flag = missing_flag
        if self.missing_flag is not None:
            for dataset_key in self.values:
                self.values[dataset_key][self.values[dataset_key] == self.missing_flag] = np.nan
        self.remove_missing_upto_maxlag = remove_missing_upto_maxlag

        # If PCMCI.run_bootstrap_of is called, then the
        # bootstrap random draw can be set here
        self.bootstrap = None
        
    def update_transform(self, transform=None):
        self.data_transform = transform
        
        if self.data_transform is None:
            self.fitted_transform = None
            for attr in transform_attrs:
                self.__setattr__(attr, self.original_data[attr])
        else:
            self.fitted_transform = {}
            new_values = {k: [] for k in self.original_data['values'].keys()}
            new_vector_lengths = []

            max_lag = np.max(-1*np.array([varlag[1] for _, vl_list in self.original_data['vector_vars'].items() for varlag in vl_list]))

            for d in self.original_data['T'].keys():
                self.T[d] = self.original_data['T'][d]-max_lag 
            #TODO this will look different if ColumnTransformer is modified in a usable way
            for vec_var, input_varlag_list in self.original_data['vector_vars'].items():
                vec_data = np.concatenate(
                    [_select_variables(data_by_set, I=[get_var(varlag) for varlag in input_varlag_list],
                                       lags=[get_lag(varlag) for varlag in input_varlag_list]) 
                     for _, data_by_set in self.original_data['values'].items()])
                self.fitted_transform[vec_var] = deepcopy(self.data_transform).fit(vec_data)
                for d, data_by_set in self.original_data['values'].items():
                    p_array = self.fitted_transform[vec_var].transform(vec_data)
                    new_values[d] += [p_array]
                new_vector_lengths+=[_get_num_variables(p_array)]
            for d in self.datasets:
                new_values[d] = np.concatenate(new_values[d], axis=1)
            self.values = new_values
            self.vector_lengths = np.array(new_vector_lengths)
            self.vector_vars = self.make_vector_node_dict(self.vector_lengths)
            self.Ndata = np.sum(self.vector_lengths)
        
    def make_vector_node_dict(self, vector_lengths, include_lag=True, key_func=None):
        if key_func is None:
            key_func = lambda x: x
        d = {key_func(i): [int(x+np.sum(vector_lengths[:i])) for x in range(vector_lengths[i])] for i in range(len(vector_lengths))}
        if include_lag:
            return {k: [(e,0) for e in d[k]] for k in d.keys()}
        else:
            return d

    def get_index_code(self, n):
        return index_codes[n]

    def _check_mask(self, mask, check_data_type=False):
        """Checks that the mask is:
            * The same shape as the data
            * Is an numpy ndarray (or subtype)
            * Does not contain any NaN entries

        """
        # Check that there is a mask if required
        _use_mask = mask

        # If we have a mask, check it
        if _use_mask is not None:
            # Check data type and generic format of 'mask', map to multiple datasets mode
            # dictionary representation
            if isinstance(_use_mask, np.ndarray):
                if len(_use_mask.shape) == 2:
                    _use_mask_dict = {0: _use_mask}
                elif len(_use_mask.shape) == 3:
                    if _use_mask.shape[0] == self.M:
                        _use_mask_dict = {i: _use_mask[i, :, :] for i in range(self.M)}
                    else:
                        raise ValueError("Shape mismatch: {} datasets "\
                            " in 'data' but {} in 'mask', must be "\
                            "identical.".format(self.M, _use_mask.shape[0]))

                else:
                    raise TypeError("'data' given as 3d np.ndarray. "\
                        "'mask' is np.ndarray of shape {}, must be of "\
                        "shape (M, T, N).".format(_use_mask.shape))

            elif isinstance(_use_mask, dict):
                if len(_use_mask) == self.M:
                    for dataset_key in self.values.keys():
                        if _use_mask.get(dataset_key) is None:
                            raise ValueError("'data' has key {} (type {}) "\
                                "but 'mask' does not, keys must be "\
                                "identical.".format(dataset_key,
                                    type(dataset_key)))

                    _use_mask_dict = _use_mask

                else:
                    raise ValueError("Shape mismatch: {} datasets "\
                        "in 'data' but {} in 'mask', must be "\
                        "identical.".format(self.M, len(_use_mask)))
            else:
                raise TypeError("'mask' is of type "\
                    "{}, must be dict or array.".format(type(_use_mask)))

            # Check for consistency with shape of 'self.values' and for NaNs
            for dataset_key, dataset_data in self.values.items():
                _use_mask_dict_data = _use_mask_dict[dataset_key] 
                if _use_mask_dict_data.shape == dataset_data.shape:
                    if np.sum(np.isnan(_use_mask_dict_data)) != 0:
                        raise ValueError("NaNs in the data mask")
                    if check_data_type:
                        if not set(np.unique(_use_mask_dict_data)).issubset(set([0, 1])):
                            raise ValueError("Type mask contains other values than 0 and 1")
                else:
                    if self.analysis_mode == 'single':
                        raise ValueError("Shape mismatch: 'data' is of shape "\
                            "{}, 'mask' is of shape {}. Must be "\
                            "identical.".format(dataset_data.shape,
                                _use_mask_dict_data.shape))
                    elif self.analysis_mode == 'multiple':
                        raise ValueError("Shape mismatch: dataset {} "\
                            "is of shape {} in 'data' and of shape {} in "\
                            "'mask'. Must be identical.".format(dataset_key,
                                dataset_data.shape,
                                _use_mask_dict_data.shape))

            # Return the mask in dictionary format
            return _use_mask_dict

    def _check_and_set_time_offsets(self, time_offsets):
        """Check the argument 'time_offsets' for consistency and bring into
        canonical format"""

        if time_offsets is not None:

            assert self.analysis_mode == 'multiple'
            assert self._initialized_from == 'dict'

            # Check data type and generic format of 'time_offsets', map to
            # dictionary representation
            if isinstance(time_offsets, dict):
                if len(time_offsets) == self.M:
                    for dataset_key in self.values.keys():
                        if time_offsets.get(dataset_key) is None:
                            raise ValueError("'data' has key {} (type {}) but "\
                                "'time_offsets' does not, keys must be "\
                                "identical.".format(dataset_key,
                                    type(dataset_key)))

                    self.time_offsets = time_offsets

                else:
                    raise ValueError("Shape mismatch: {} datasets in "\
                        "'data' but {} in 'time_offsets', must be "\
                        "identical.".format(self.M, len(time_offsets)))

            else:
                raise TypeError("'time_offsets' is of type {}, must be "\
                    "dict.".format(type(time_offsets)))

            # All time offsets must be non-negative integers, at least one of
            # which is zero
            found_zero_time_offset = False
            for time_offset in self.time_offsets.values():
                if np.issubdtype(type(time_offset), np.integer):
                    if time_offset >= 0:
                        if time_offset == 0:
                            found_zero_time_offset = True
                    else:
                        raise ValueError("A dataset has time offset "\
                            "{}, must be non-negative.".format(time_offset))

                else:
                    raise TypeError("There is a time offset of type {}, must "\
                        "be int.".format(type(time_offset)))

            if not found_zero_time_offset:
                raise ValueError("At least one time offset must be 0.")

        else:
            # If no time offsets are specified, all of them are zero
            self.time_offsets = {dataset_key: 0 for dataset_key in self.values.keys()}

    def _check_and_set_reference_points(self, reference_points):
        """Check the argument 'reference_point' for consistency and bring into
        canonical format"""

        # Check type of 'reference_points' and its elements
        if reference_points is None:
            # If no reference point is specified, use as many reference points
            # as possible
            self.reference_points = np.arange(self.largest_time_step)

        elif isinstance(reference_points, int):
            # If a single reference point is specified as an int, convert it to
            # a single element numpy array
            self.reference_points = np.array([reference_points])

        elif isinstance(reference_points, np.ndarray):
            # Check that all reference points are ints
            for ref_point in reference_points:
                if not np.issubdtype(type(ref_point), np.integer):
                    raise TypeError("All reference points must be integers.")

            self.reference_points = reference_points

        elif isinstance(reference_points, list):
            # Check that all reference points are ints
            for ref_point in reference_points:
                if not isinstance(ref_point, int):
                    raise TypeError("All reference points must be integers.")

            # If given as a list, cast to numpy array
            self.reference_points = np.asarray(reference_points)

        else:
            raise TypeError("Unsupported data type of 'reference_points': Is "\
                "{}, must be None or int or a list or np.ndarray of "\
                "ints.".format(type(reference_points)))

        # Remove negative reference points
        if np.sum(self.reference_points < 0) > 0:
            warnings.warn("Some reference points were negative. These are "\
                "removed.")
            self.reference_points = self.reference_points[self.reference_points >= 0]

        # Remove reference points that are larger than the largest time step
        if np.sum(self.reference_points >= self.largest_time_step) > 0:
            warnings.warn("Some reference points were larger than the largest "\
                "relevant time step, which here is {}. These are "\
                "removed.".format(self.largest_time_step - 1))
            self.reference_points = self.reference_points[self.reference_points < self.largest_time_step]

        # Raise an error if no valid reference points was specified
        if len(self.reference_points) == 0:
            raise ValueError("No valid reference point.") 

    def construct_array(self, X, Y, 
                        Z, #user-defined conditions in causal effect estimation, potential separating set in causal discovery
                        tau_max,
                        extraZ=None, #adjustment set in causal effect estimation, doesn't exist for causal discovery
                        mask=None,
                        mask_type=None,
                        data_type=None,
                        remove_duplicates=True,
                        return_node_info=False,
                        return_cleaned_xyz=False,
                        do_checks=True,
                        remove_overlaps=True,
                        cut_off='2xtau_max',
                        verbosity=0):
        """Constructs array from variables X, Y, Z from data.
        Data is of shape (T, N) if analysis_mode == 'single', where T is the
        time series length and N the number of variables, and of (n_ens, T, N)
        if analysis_mode == 'multiple'.

        Parameters
        ----------
        X, Y, Z, extraZ : list of tuples
            For a dependence measure I(X;Y|Z), X, Y, Z can be multivariate of
            the form [(var1, -lag), (var2, -lag), ...]. At least one varlag in Y 
            has to be at lag zero. extraZ is only used in CausalEffects class.
        tau_max : int
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X and Z all have the same sample size.
        mask : array-like, optional (default: None)
            Optional mask array, must be of same shape as data.  If it is set,
            then it overrides the self.mask assigned to the dataframe. If it is
            None, then the self.mask is used, if it exists.
        mask_type : {None, 'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, the mask
            is not used. Explained in tutorial on masking and missing values.
        data_type : array-like
            Binary data array of same shape as array which describes whether 
            individual samples in a variable (or all samples) are continuous 
            or discrete: 0s for continuous variables and 1s for discrete variables.
            If it is set, then it overrides the self.data_type assigned to the dataframe.
        return_cleaned_xyz : bool, optional (default: False)
            Whether to return cleaned X,Y,Z, where possible duplicates are
            removed.
        do_checks : bool, optional (default: True)
            Whether to perform sanity checks on input X,Y,Z
        remove_duplicates: bool, optional (default: True)
            Recommended False for effect estimation with vectorized dataframes.
            When False, prevents removing input-level duplicates and overlaps from the dataframe -- 
            a step which destroys vector-node-based logic. 
        return_node_info : bool, optional (default: False)
            Whether to return VECTOR_NODES, which uses unique integers to identify which vector (var, lag)
            produced each variable in the ARRAY, and NODE_DICT, which translates the int into the vector 
            (var, lag) pair. These only have meaning when remove_duplicates is False.
        remove_overlaps : bool, optional (default: True)
            Whether to remove input variables from X and Y if they overlap with Z, for the purpose of causal discovery. -- breaking change
                        
            #TODO triple-check this logic with Jakob
            "Duplicates" refers to repeats within a group X, Y, Z, or extraZ
            "Overlaps" refers to nodes in more than one group
                
            in causal discovery, Z is the attempted separating set, and extraZ doesn't exist
            in causal effect estimation:
                dp.Z = models.conditions = causal_effects.S = user-defined conditions
                dp.extraZ = models.Z = adjustment set
                
            macro-level overlaps and duplicates cannot happen in causal discovery:
                -X and Y are each only one macro-node
                -Z (attempted separating set) is constructed to avoid X and Y.
                -extraZ doesn't exist
            macro-level overlaps and duplicates can happen in effect estimation:
                -all macro-duplicates from user input X, Y, conditions throw errors in effect_estimation and models
                -Next to _check_validity, check for duplicates in user adjustment_set and throw error
                -adjustment set and conditions may have overlaps because of user input. Remove from adjustment set if overlaps with conditions.
            => here we only worry about micro-level overlaps and duplicates

            return_node_info: will be true for causal effect estimation.
            (remove_overlaps, remove_duplicates)
            causal discovery: (T, T) #Talk to Wiebke and Urmi about their use cases and what we would want.
                -remove duplicates from Z (X and Y have only one macro-node and extraZ doesn't exist, so can't hurt if easier)
                -remove overlaps from x and y if appear in Z -- breaking change
            causal effect estimation with transformation: (F, F)
                -don't remove anything
            causal effect estimation without transformation: (F, T)
                -remove micro duplicates from X, Z, extraZ (removing from Y would reduce computation but not accuracy, and could make it hard to interpret later)
                -remove overlaps from extraz and z if appear in X or other Z
                
                TODOTODO!
                BRAINSTORMED IDEAS:
                -data transformation during creation of dataframe
                -function that could be called by user or Models to update that transformation
                -insist on such transformation if there are overlaps? Or might we have some clever conditional independence test or kernel thing?
                -should dimension reduction ever be done for Z together or always at the Macro-node level?
                    I argue: always the macro-node level, or there's no point in having the data appear in more than one node -- one would be enough
                -could then remove transformation logic from models? or keep both?
                -let overlaps throw warnings and give nans during causal discovery?
                -think about multiple datasets and dummy variables -- should they be labelled
            
        cut_off : {'2xtau_max', 'tau_max', 'max_lag', 'max_lag_or_tau_max', 2xtau_max_future}
            If cut_off == '2xtau_max':
                - 2*tau_max samples are cut off at the beginning of the time
                  series ('beginning' here refers to the temporally first
                  time steps). This guarantees that (as long as no mask is
                  used) all MCI tests are conducted on the same samples,
                  independent of X, Y, and Z.

                - If at time step t_missing a data value is missing, then the
                  time steps t_missing, ..., t_missing + 2*tau_max are cut
                  out. The latter part only holds if
                  remove_missing_upto_maxlag=True.

            If cut_off ==  'max_lag':
                - max_lag(X, Y, Z) samples are cut off at the beginning of the
                  time series, where max_lag(X, Y, Z) is the maximum lag of
                  all nodes in X, Y, and Z. These are all samples that can in
                  principle be used.

                - If at time step t_missing a data value is missing, then the
                  time steps t_missing, ..., t_missing + max_lag(X, Y, Z) are
                  cut out. The latter part only holds if
                  remove_missing_upto_maxlag=True.

            If cut_off == 'max_lag_or_tau_max':
                - max(max_lag(X, Y, Z), tau_max) are cut off at the beginning.
                  This may be useful for modeling by comparing multiple
                  models on the same samples. 

                - If at time step t_missing a data value is missing, then the
                  time steps t_missing, ..., t_missing + max(max_lag(X, Y,
                  Z), tau_max) are cut out. The latter part only holds if
                  remove_missing_upto_maxlag=True.

            If cut_off == 'tau_max':
                - tau_max samples are cut off at the beginning. This may be
                  useful for modeling by comparing multiple models on the
                  same samples. 

                - If at time step t_missing a data value is missing, then the
                  time steps t_missing, ..., t_missing + max(max_lag(X, Y,
                  Z), tau_max) are cut out. The latter part only holds if
                  remove_missing_upto_maxlag=True.
                
            If cut_off == '2xtau_max_future':
                First, the relevant time steps are determined as for cut_off ==
                'max_lag'. Then, the temporally latest time steps are removed
                such that the same number of time steps remains as there would
                be for cut_off == '2xtau_max'. This may be useful when one is
                mostly interested in the temporally first time steps and would
                like all MCI tests to be performed on the same *number* of
                samples. Note, however, that while the *number* of samples is
                the same for all MCI tests, the samples themselves may be
                different.
        verbosity : int, optional (default: 0)
            Level of verbosity.

        Returns
        -------
        array, xyz [,XYZ] [,vector_nodes, node_dict], data_type : Tuple of data 
            array of shape (dim, n_samples), xyz identifier array of shape (dim,) 
            identifying which row in array corresponds to X, Y, and Z, and the 
            type mask that indicates which samples are continuous or discrete. 
            For example: X = [(0, -1)], Y = [(1, 0)], Z = [(1, -1), (0, -2)] 
            yields an array of shape (4, n_samples) and xyz is 
            xyz = numpy.array([0,1,2,2]). If return_cleaned_xyz is True, also 
            outputs the cleaned XYZ lists. If return_node_info is True, also
            outputs the vector_nodes array and node_dict, which together provide
            the vector information that would otherwise be lost during the creation
            of the array from a vectorized dataset. Node dict defines vector 
            node-and-lag indices in terms of their (var, lag) tuple,
            while vector_nodes identifies which vector node-and-lag provided each entry 
            in the returned 'array'.
        """

        # # This version does not yet work with bootstrap
        # try:
        #     assert self.bootstrap is None
        # except AssertionError:
        #     print("This version does not yet work with bootstrap.")
        #     raise

        if extraZ is None:
            extraZ = []
        
        if Z is None:
            Z = []
        
        #assume no overlap here -- should be checked before arguments are passed in.
        all_nodes = X+Y+Z+extraZ
        #vector variables may appear multiple times in all_nodes at different lags. So we define new indices, and the following
        #dict maps these indices to the vector-level tuple (var, lag).
        node_dict = {i: all_nodes[i] for i in range(len(all_nodes))}
        #array like xyz which shows which vector-level (var, lag) pair the variables in OBSERVATION_ARRAY come from.
        #node_dict[k][0] retrieves just the node index to use for vector_lengths
        vector_nodes = np.array([k for k in node_dict.keys() for i in range(self.vector_lengths[node_dict[k][0]])])
                    
        # If vector-valued variables exist, add them
        def vectorize(varlag):     
            vectorized_var = []
            for (var, lag) in varlag:
                for (vector_var, vector_lag) in self.vector_vars[var]:
                    vectorized_var.append((vector_var, vector_lag + lag))
            return vectorized_var

        X = vectorize(X) 
        Y = vectorize(Y) 
        Z = vectorize(Z) 
        extraZ = vectorize(extraZ) 
        
        def warn_duplicates_overlaps(R, X, n, msg):
            if n is not None:
                if len(R)<len(X):
                    Warnings.warn(msg.format(n))
        
        def remove_duplicates(X, n=None):
            R = list(OrderedDict.fromkeys(X))
            warn_duplicates_overlaps(R, X, n, "Removing input-level duplicates from {}")
            return R
        
        #O, a list of nodes; i.e. X+extraZ
        def remove_overlaps(X, O, n=None):
            R = [node for node in X if (node not in O)]
            warn_duplicates_overlaps(R, X, n, "Removing input-level overlaps from {}")
            return R
        
        if remove_duplicates: #when a transformation that needs duplicates and overlaps will not be applied
            #remove input-level duplicates within each set of nodes
            #not necessary for Y because it will always be a target, whether causal discovery or effect estimation
            #TODO if happens for X during effect estimation, then how will we pass in data later?!
            X = remove_duplicates(X, 'x')
            Z = remove_duplicates(Z, 'z')
            extraZ = remove_duplicates(extraZ, 'e')
            if remove_overlaps: #True for causal discovery
                #remove input-level overlaps from X and Y if appear in Z -- TODO breaking change!
                X = remove_overlaps(X, Z)
                Y = remove_overlaps(Y, Z)
            else: #for causal effect estimation
                #remove input-level overlaps...
                #...from adjustment set if appears in user conditions or X (X is prescribed, and Z may be prescribed)
                extraZ = remove_overlaps(extraZ, X+Z)
                #...from user conditions if appears in X
                Z = remove_overlaps(Z, X, 'user-defined conditioning set')
                #TODO removing Y is a breaking change, but seems good.  

        XYZ = X + Y + Z + extraZ
        dim = len(XYZ)

        # Check that all lags are non-positive and indices are in [0,N-1]
        if do_checks:
            self._check_nodes(Y, XYZ, self.Ndata, dim)

        # Use the mask, override if needed
        _mask = mask
        if _mask is None:
            _mask = self.mask
        else:
            _mask = self._check_mask(mask = _mask)
            
        _data_type = data_type
        if _data_type is None:
            _data_type = self.data_type
        else:
            _data_type = self._check_mask(mask = _data_type, check_data_type=True)

        # Figure out what cut off we will be using
        if cut_off == '2xtau_max':
            max_lag = 2*tau_max
        elif cut_off == 'max_lag':
            max_lag = abs(np.array(XYZ)[:, 1].min())
        elif cut_off == 'tau_max':
            max_lag = tau_max
        elif cut_off == 'max_lag_or_tau_max':
            max_lag = max(abs(np.array(XYZ)[:, 1].min()), tau_max)
        elif cut_off == '2xtau_max_future':
            ## TODO: CHECK THIS
            max_lag = abs(np.array(XYZ)[:, 1].min())
        else:
            raise ValueError("max_lag must be in {'2xtau_max', 'tau_max', 'max_lag', "\
                "'max_lag_or_tau_max', '2xtau_max_future'}")
            
        #TODO make this an accessible function for use in models.py?
        xyz = np.array([self.get_index_code(name)
                        for var, name in zip([X, Y, Z, extraZ], ['x', 'y', 'z', 'e'])
                        for _ in var])

        # Run through all datasets and fill a dictionary holding the
        # samples taken from the individual datasets
        samples_datasets = dict()
        data_types = dict()
        self.use_indices_dataset_dict = dict()

        for dataset_key, dataset_data in self.values.items():

            # Apply time offset to the reference points
            ref_points_here = self.reference_points - self.time_offsets[dataset_key]

            # Remove reference points that are out of bounds or are to be
            # excluded given the choice of 'cut_off'
            ref_points_here = ref_points_here[ref_points_here >= max_lag]
            ref_points_here = ref_points_here[ref_points_here < self.T[dataset_key]]

            # Keep track of which reference points would have remained for
            # max_lag == 2*tau_max
            if cut_off == '2xtau_max_future':
                ref_points_here_2_tau_max = self.reference_points - self.time_offsets[dataset_key]
                ref_points_here_2_tau_max = ref_points_here_2_tau_max[ref_points_here_2_tau_max  >= 2*tau_max]
                ref_points_here_2_tau_max = ref_points_here_2_tau_max[ref_points_here_2_tau_max  < self.T[dataset_key]]

            # Sort the valid reference points (not needed, but might be useful
            # for detailed debugging)
            ref_points_here = np.sort(ref_points_here)

            # For cut_off == '2xtau_max_future' reduce the samples size the
            # number of samples that would have been obtained for cut_off ==
            # '2xtau_max', removing the temporally latest ones
            if cut_off == '2xtau_max_future':
                n_to_cut_off = len(ref_points_here) - len(ref_points_here_2_tau_max)
                assert n_to_cut_off >= 0
                if n_to_cut_off > 0:
                    ref_points_here = np.sort(ref_points_here)
                    ref_points_here = ref_points_here[:-n_to_cut_off]

            # If no valid reference points are left, continue with the next dataset
            if len(ref_points_here) == 0:
                continue

            if self.bootstrap is not None:

                boot_blocklength = self.bootstrap['boot_blocklength']

                if boot_blocklength == 'cube_root':
                    boot_blocklength = max(1, int(len(ref_points_here)**(1/3)))
                # elif boot_blocklength == 'from_autocorrelation':
                #     boot_blocklength = \
                #         get_block_length(overlapping_residuals.T, xyz=np.zeros(N), mode='confidence')
                elif type(boot_blocklength) is int and boot_blocklength > 0:
                    pass
                else:
                    raise ValueError("boot_blocklength must be integer > 0, 'cube_root', or 'from_autocorrelation'")

                # Chooses THE SAME random seed for every dataset, maybe that's what we want...
                # If the reference points are all the same, this will give the same bootstrap
                # draw. However, if they are NOT the same, they will differ. 
                # TODO: Decide whether bootstrap draws should be the same for each dataset and
                # how to achieve that if the reference points differ...
                # random_state = self.bootstrap['random_state']
                random_state = deepcopy(self.bootstrap['random_state'])

                # Determine the number of blocks total, rounding up for non-integer
                # amounts
                n_blks = int(math.ceil(float(len(ref_points_here))/boot_blocklength))

                if n_blks < 10:
                    raise ValueError("Only %d block(s) for block-sampling,"  %n_blks +
                                     " choose smaller boot_blocklength!")

                # Get the starting indices for the blocks
                blk_strt = random_state.choice(np.arange(len(ref_points_here) - boot_blocklength), size=n_blks, replace=True)
                # Get the empty array of block resampled values
                boot_draw = np.zeros(n_blks*boot_blocklength, dtype='int')
                # Fill the array of block resamples
                for i in range(boot_blocklength):
                    boot_draw[i::boot_blocklength] = ref_points_here[blk_strt + i]
                # Cut to proper length
                ref_points_here = boot_draw[:len(ref_points_here)]

            # Construct the data array holding the samples taken from the
            # current dataset
            samples_datasets[dataset_key] = np.zeros((dim, len(ref_points_here)), dtype = dataset_data.dtype)
            for i, (var, lag) in enumerate(XYZ):
                samples_datasets[dataset_key][i, :] = dataset_data[ref_points_here + lag, var]

            # Build the mask array corresponding to this dataset
            if _mask is not None:
                mask_dataset = np.zeros((dim, len(ref_points_here)), dtype = 'bool')
                for i, (var, lag) in enumerate(XYZ):
                    mask_dataset[i, :] = _mask[dataset_key][ref_points_here + lag, var]

            # Take care of masking
            use_indices_dataset = np.ones(len(ref_points_here), dtype = 'int')

            # Build the type mask array corresponding to this dataset
            if _data_type is not None:
                data_type_dataset = np.zeros((dim, len(ref_points_here)), dtype = 'bool')
                for i, (var, lag) in enumerate(XYZ):
                    data_type_dataset[i, :] = _data_type[dataset_key][ref_points_here + lag, var]
                data_types[dataset_key] = data_type_dataset
            
            # Remove all values that have missing value flag, and optionally as well the time
            # slices that occur up to max_lag after
            if self.missing_flag is not None:
                missing_anywhere = np.array(np.where(np.any(np.isnan(samples_datasets[dataset_key]), axis=0))[0])

                if self.remove_missing_upto_maxlag:
                    idx_to_remove = set(idx + tau for idx in missing_anywhere for tau in range(max_lag + 1))
                else:
                    idx_to_remove = set(idx for idx in missing_anywhere)
                
                use_indices_dataset[np.array(list(idx_to_remove), dtype='int')] = 0
            
            if _mask is not None:
                # Remove samples with mask == 1 conditional on which mask_type
                # is used

                # Iterate over defined mapping from letter index to number index,
                # i.e. 'x' -> 0, 'y' -> 1, 'z'-> 2, 'e'-> 3
                for idx, cde in index_codes.items():
                    # Check if the letter index is in the mask type
                    if (mask_type is not None) and (idx in mask_type):
                        # If so, check if any of the data that correspond to the
                        # letter index is masked by taking the product along the
                        # node-data to return a time slice selection, where 0
                        # means the time slice will not be used
                        slice_select = np.prod(mask_dataset[xyz == cde, :] == False, axis=0)
                        use_indices_dataset *= slice_select

            # Accordingly update the data array
            samples_datasets[dataset_key] = samples_datasets[dataset_key][:, use_indices_dataset == 1]

        ## end for dataset_key, dataset_data in self.values.items()

        # Save used indices as attribute
        if len(ref_points_here) > 0:
            self.use_indices_dataset_dict[dataset_key] = ref_points_here[use_indices_dataset==1]
        else:
            self.use_indices_dataset_dict[dataset_key] = []

        # Concatenate the arrays of all datasets
        array = np.concatenate(tuple(samples_datasets.values()), axis = 1)
        if _data_type is not None:
            type_array = np.concatenate(tuple(data_types.values()), axis = 1)
        else:
            type_array = None
        
        # print(np.where(np.isnan(array)))
        # print(array.shape)

        # Check whether there is any valid sample
        if array.shape[1] == 0:
            raise ValueError("No valid samples")

        # Print information about the constructed array
        if verbosity > 2:
            self.print_array_info(array, X, Y, Z, self.missing_flag, mask_type, type_array, extraZ)

        # Return the array and xyz and optionally (X, Y, Z) and/or vector node information
        if return_node_info and return_cleaned_xyz:
            return array, xyz, (X, Y, Z), vector_nodes, node_dict, type_array
        
        if return_node_info:
            return array, xyz, vector_nodes, node_dict, type_array
        
        if return_cleaned_xyz:
            return array, xyz, (X, Y, Z), type_array

        return array, xyz, type_array

    #TODO Y currently unused!
    def _check_nodes(self, Y, XYZ, N, dim):
        """
        Checks that:
        * The requests XYZ nodes have the correct shape
        * All lags are non-positive
        * All indices are less than N
        * One of the Y nodes has zero lag

        Parameters
        ----------
        Y : list of tuples
            Of the form [(var, -tau)], where var specifies the variable
            index and tau the time lag.
        XYZ : list of tuples
            List of nodes chosen for current independence test
        N : int
            Total number of listed nodes
        dim : int
            Number of nodes excluding repeated nodes
        """
        if np.array(XYZ).shape != (dim, 2):
            raise ValueError("X, Y, Z must be lists of tuples in format"
                             " [(var, -lag),...], eg., [(2, -2), (1, 0), ...]")
        if np.any(np.array(XYZ)[:, 1] > 0):
            raise ValueError("nodes are %s, " % str(XYZ) +
                             "but all lags must be non-positive")
        if (np.any(np.array(XYZ)[:, 0] >= N)
                or np.any(np.array(XYZ)[:, 0] < 0)):
            raise ValueError("var indices %s," % str(np.array(XYZ)[:, 0]) +
                             " but must be in [0, %d]" % (N - 1))
        # if np.all(np.array(Y)[:, 1] != 0):
        #     raise ValueError("Y-nodes are %s, " % str(Y) +
        #                      "but one of the Y-nodes must have zero lag")

    def print_array_info(self, array, X, Y, Z, missing_flag, mask_type, data_type=None, extraZ=None):
        """
        Print info about the constructed array

        Parameters
        ----------
        array : Data array of shape (dim, T)
            Data array.
        X, Y, Z, extraZ : list of tuples
            For a dependence measure I(X;Y|Z), Y is of the form [(varY, 0)],
            where var specifies the variable index. X typically is of the form
            [(varX, -tau)] with tau denoting the time lag and Z can be
            multivariate [(var1, -lag), (var2, -lag), ...] .
        missing_flag : number, optional (default: None)
            Flag for missing values. Dismisses all time slices of samples where
            missing values occur in any variable and also flags samples for all
            lags up to 2*tau_max. This avoids biases, see section on masking in
            Supplement of [1]_.
        mask_type : {'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, the mask
            is not used. Explained in tutorial on masking and missing values.
        data_type : array-like
            Binary data array of same shape as array which describes whether 
            individual samples in a variable (or all samples) are continuous 
            or discrete: 0s for continuous variables and 1s for discrete variables.
        """
        if extraZ is None:
            extraZ = []
        indt = " " * 12
        print(indt + "Constructed array of shape %s from"%str(array.shape) +
              "\n" + indt + "X = %s" % str(X) +
              "\n" + indt + "Y = %s" % str(Y) +
              "\n" + indt + "Z = %s" % str(Z))
        if extraZ is not None:  
            print(indt + "extraZ = %s" % str(extraZ))
        if self.mask is not None and mask_type is not None:
            print(indt+"with masked samples in %s removed" % mask_type)
        if self.data_type is not None:
            print(indt+"with %s % discrete values" % np.sum(data_type)/data_type.size)
        if self.missing_flag is not None:
            print(indt+"with missing values = %s removed" % self.missing_flag)


def get_acf(series, max_lag=None):
    """Returns autocorrelation function.

    Parameters
    ----------
    series : 1D-array
        data series to compute autocorrelation from

    max_lag : int, optional (default: None)
        maximum lag for autocorrelation function. If None is passed, 10% of
        the data series length are used.

    Returns
    -------
    autocorr : array of shape (max_lag + 1,)
        Autocorrelation function.
    """
    # Set the default max lag
    if max_lag is None:
        max_lag = int(max(5, 0.1*len(series)))
    # Initialize the result
    autocorr = np.ones(max_lag + 1)
    # Iterate over possible lags
    for lag in range(1, max_lag + 1):
        # Set the values
        y1_vals = series[lag:]
        y2_vals = series[:len(series) - lag]
        # Calculate the autocorrelation
        autocorr[lag] = np.corrcoef(y1_vals, y2_vals, ddof=0)[0, 1]
    return autocorr

def get_block_length(array, xyz, mode):
    """Returns optimal block length for significance and confidence tests.

    Determine block length using approach in Mader (2013) [Eq. (6)] which
    improves the method of Pfeifer (2005) with non-overlapping blocks In
    case of multidimensional X, the max is used. Further details in [1]_.
    Two modes are available. For mode='significance', only the indices
    corresponding to X are shuffled in array. For mode='confidence' all
    variables are jointly shuffled. If the autocorrelation curve fit fails,
    a block length of 5% of T is used. The block length is limited to a
    maximum of 10% of T.

    Mader et al., Journal of Neuroscience Methods,
    Volume 219, Issue 2, 15 October 2013, Pages 285-291

    Parameters
    ----------
    array : array-like
        data array with X, Y, Z in rows and observations in columns

    xyz : array of ints
        XYZ identifier array of shape (dim,).

    mode : str
        Which mode to use.

    Returns
    -------
    block_len : int
        Optimal block length.
    """
    # Inject a dependency on siganal, optimize
    from scipy import signal, optimize
    # Get the shape of the array
    dim, T = array.shape
    # Initiailize the indices
    indices = range(dim)
    if mode == 'significance':
        indices = np.where(xyz == index_codes['x'])[0]

    # Maximum lag for autocov estimation
    max_lag = int(0.1*T)
    # Define the function to optimize against
    def func(x_vals, a_const, decay):
        return a_const * decay**x_vals

    # Calculate the block length
    block_len = 1
    for i in indices:
        # Get decay rate of envelope of autocorrelation functions
        # via hilbert trafo
        autocov = get_acf(series=array[i], max_lag=max_lag)
        autocov[0] = 1.
        hilbert = np.abs(signal.hilbert(autocov))
        # Try to fit the curve
        try:
            popt, _ = optimize.curve_fit(
                f=func,
                xdata=np.arange(0, max_lag+1),
                ydata=hilbert,
            )
            phi = popt[1]
            # Formula of Pfeifer (2005) assuming non-overlapping blocks
            l_opt = (4. * T * (phi / (1. - phi) + phi**2 / (1. - phi)**2)**2
                     / (1. + 2. * phi / (1. - phi))**2)**(1. / 3.)
            block_len = max(block_len, int(l_opt))
        except RuntimeError:
            warnings.warn("Error - curve_fit failed for estimating block_shuffle length, using"
                  " block_len = %d" % (int(.05 * T)))
            # block_len = max(int(.05 * T), block_len)
    # Limit block length to a maximum of 10% of T
    block_len = min(block_len, int(0.1 * T))
    return block_len


def lowhighpass_filter(data, cutperiod, pass_periods='low'):
    """Butterworth low- or high pass filter.

    This function applies a linear filter twice, once forward and once
    backwards. The combined filter has linear phase.

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    cutperiod : int
        Period of cutoff.
    pass_periods : str, optional (default: 'low')
        Either 'low' or 'high' to act as a low- or high-pass filter

    Returns
    -------
    data : array
        Filtered data array.
    """
    try:
        from scipy.signal import butter, filtfilt
    except:
        print('Could not import scipy.signal for butterworth filtering!')

    fs = 1.
    order = 3
    ws = 1. / cutperiod / (0.5 * fs)
    b, a = butter(order, ws, pass_periods)
    if np.ndim(data) == 1:
        data = filtfilt(b, a, data)
    else:
        for i in range(data.shape[1]):
            data[:, i] = filtfilt(b, a, data[:, i])

    return data


def smooth(data, smooth_width, kernel='gaussian',
           mask=None, residuals=False, verbosity=0):
    """Returns either smoothed time series or its residuals.

    the difference between the original and the smoothed time series
    (=residuals) of a kernel smoothing with gaussian (smoothing kernel width =
    twice the sigma!) or heaviside window, equivalent to a running mean.

    Assumes data of shape (T, N) or (T,)
    :rtype: array
    :returns: smoothed/residual data

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    smooth_width : float
        Window width of smoothing, 2*sigma for a gaussian.
    kernel : str, optional (default: 'gaussian')
        Smoothing kernel, 'gaussian' or 'heaviside' for a running mean.
    mask : bool array, optional (default: None)
        Data mask where True labels masked samples.
    residuals : bool, optional (default: False)
        True if residuals should be returned instead of smoothed data.
    verbosity : int, optional (default: 0)
        Level of verbosity.

    Returns
    -------
    data : array-like
        Smoothed/residual data.
    """

    if verbosity > 0:
        print("%s %s smoothing with " % ({True: "Take residuals of a ",
                                      False: ""}[residuals], kernel) +
          "window width %.2f (=2*sigma for a gaussian!)" % (smooth_width))

    totaltime = len(data)
    if kernel == 'gaussian':
        window = np.exp(-(np.arange(totaltime).reshape((1, totaltime)) -
                             np.arange(totaltime).reshape((totaltime, 1))
                             ) ** 2 / ((2. * smooth_width / 2.) ** 2))
    elif kernel == 'heaviside':
        import scipy.linalg
        wtmp = np.zeros(totaltime)
        wtmp[:int(np.ceil(smooth_width / 2.))] = 1
        window = scipy.linalg.toeplitz(wtmp)

    if mask is None:
        if np.ndim(data) == 1:
            smoothed_data = (data * window).sum(axis=1) / window.sum(axis=1)
        else:
            smoothed_data = np.zeros(data.shape)
            for i in range(data.shape[1]):
                smoothed_data[:, i] = (
                    data[:, i] * window).sum(axis=1) / window.sum(axis=1)
    else:
        if np.ndim(data) == 1:
            smoothed_data = ((data * window * (mask==False)).sum(axis=1) /
                             (window * (mask==False)).sum(axis=1))
        else:
            smoothed_data = np.zeros(data.shape)
            for i in range(data.shape[1]):
                smoothed_data[:, i] = ((
                    data[:, i] * window * (mask==False)[:, i]).sum(axis=1) /
                    (window * (mask==False)[:, i]).sum(axis=1))

    if residuals:
        return data - smoothed_data
    else:
        return smoothed_data


def weighted_avg_and_std(values, axis, weights):
    """Returns the weighted average and standard deviation.

    Parameters
    ---------
    values : array
        Data array of shape (time, variables).
    axis : int
        Axis to average/std about
    weights : array
        Weight array of shape (time, variables).

    Returns
    -------
    (average, std) : tuple of arrays
        Tuple of weighted average and standard deviation along axis.
    """

    values[np.isnan(values)] = 0.
    average = np.ma.average(values, axis=axis, weights=weights)

    variance = np.sum(weights * (values - np.expand_dims(average, axis)
                                    ) ** 2, axis=axis) / weights.sum(axis=axis)

    return (average, np.sqrt(variance))


def time_bin_with_mask(data, time_bin_length, mask=None):
    """Returns time binned data where only about non-masked values is averaged.

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    time_bin_length : int
        Length of time bin.
    mask : bool array, optional (default: None)
        Data mask where True labels masked samples.

    Returns
    -------
    (bindata, T) : tuple of array and int
        Tuple of time-binned data array and new length of array.
    """

    T = len(data)

    time_bin_length = int(time_bin_length)

    if mask is None:
        sample_selector = np.ones(data.shape)
    else:
        # Invert mask
        sample_selector = (mask == False)

    if np.ndim(data) == 1.:
        data.shape = (T, 1)
        if mask is not None:
            mask.shape = (T, 1)
        else:
            sample_selector = np.ones(data.shape)

    bindata = np.zeros(
        (T // time_bin_length,) + data.shape[1:], dtype="float32")
    for index, i in enumerate(range(0, T - time_bin_length + 1,
                                    time_bin_length)):
        # print weighted_avg_and_std(fulldata[i:i+time_bin_length], axis=0,
        # weights=sample_selector[i:i+time_bin_length])[0]
        bindata[index] = weighted_avg_and_std(data[i:i + time_bin_length],
                                              axis=0,
                                              weights=sample_selector[i:i +
                                              time_bin_length])[0]

    T, grid_size = bindata.shape

    return (bindata.squeeze(), T)

def trafo2normal(data, mask=None, thres=0.001):
    """Transforms input data to standard normal marginals.

    Assumes data.shape = (T, dim)

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    thres : float
        Set outer points in CDF to this value.
    mask : bool array, optional (default: None)
        Data mask where True labels masked samples.

    Returns
    -------
    normal_data : array-like
        data with standard normal marginals.
    """

    def trafo(xi):
        xisorted = np.sort(xi)
        yi = np.linspace(1. / len(xi), 1, len(xi))
        return np.interp(xi, xisorted, yi)

    normal_data = np.copy(data)

    if np.ndim(data) == 1:
        if mask is None:
            nonmasked = np.where(np.isnan(data) == False)[0]
        else:
            nonmasked = np.where((mask==0)*(np.isnan(data) == False))

        u = trafo(data[nonmasked])
        u[u==0.] = thres
        u[u==1.] = 1. - thres
        normal_data[nonmasked] = stats.norm.ppf(u)
    else:
        for i in range(data.shape[1]):
            if mask is None:
                nonmasked = np.where(np.isnan(data[:,i]) == False)[0]
            else:
                nonmasked = np.where((mask[:, i]==0)*(np.isnan(data[:, i]) == False))
                # nonmasked = np.where(mask[:, i]==0)
            # print(data[:, i].shape, nonmasked.shape)
            uniform = trafo(data[:, i][nonmasked])
            
            # print(data[-3:, i][nonmasked])

            uniform[uniform==0.] = thres
            uniform[uniform==1.] = 1. - thres
            normal_data[:, i][nonmasked] = stats.norm.ppf(uniform)

    return normal_data

@jit(nopython=True)
def _get_patterns(array, array_mask, patt, patt_mask, weights, dim, step, fac, N, T):
    v = np.zeros(dim, dtype='float')

    start = step * (dim - 1)
    for n in range(0, N):
        for t in range(start, T):
            mask = 1
            ave = 0.
            for k in range(0, dim):
                tau = k * step
                v[k] = array[t - tau, n]
                ave += v[k]
                mask *= array_mask[t - tau, n]
            ave /= dim
            var = 0.
            for k in range(0, dim):
                var += (v[k] - ave) ** 2
            var /= dim
            weights[t - start, n] = var
            if (v[0] < v[1]):
                p = 1
            else:
                p = 0
            for i in range(2, dim):
                for j in range(0, i):
                    if (v[j] < v[i]):
                        p += fac[i]
            patt[t - start, n] = p
            patt_mask[t - start, n] = mask

    return patt, patt_mask, weights

def ordinal_patt_array(array, array_mask=None, dim=2, step=1,
                        weights=False, verbosity=0):
    """Returns symbolified array of ordinal patterns.

    Each data vector (X_t, ..., X_t+(dim-1)*step) is converted to its rank
    vector. E.g., (0.2, -.6, 1.2) --> (1,0,2) which is then assigned to a
    unique integer (see Article). There are faculty(dim) possible rank vectors.

    Note that the symb_array is step*(dim-1) shorter than the original array!

    Reference: B. Pompe and J. Runge (2011). Momentary information transfer as
    a coupling measure of time series. Phys. Rev. E, 83(5), 1-12.
    doi:10.1103/PhysRevE.83.051122

    Parameters
    ----------
    array : array-like
        Data array of shape (time, variables).
    array_mask : bool array
        Data mask where True labels masked samples.
    dim : int, optional (default: 2)
        Pattern dimension
    step : int, optional (default: 1)
        Delay of pattern embedding vector.
    weights : bool, optional (default: False)
        Whether to return array of variances of embedding vectors as weights.
    verbosity : int, optional (default: 0)
        Level of verbosity.

    Returns
    -------
    patt, patt_mask [, patt_time] : tuple of arrays
        Tuple of converted pattern array and new length
    """
    from scipy.misc import factorial

    array = array.astype('float64')

    if array_mask is not None:
        assert array_mask.dtype == 'int32'
    else:
        array_mask = np.zeros(array.shape, dtype='int32')


    if np.ndim(array) == 1:
        T = len(array)
        array = array.reshape(T, 1)
        array_mask = array_mask.reshape(T, 1)

    # Add noise to destroy ties...
    array += (1E-6 * array.std(axis=0)
              * random_state.random((array.shape[0], array.shape[1])).astype('float64'))

    patt_time = int(array.shape[0] - step * (dim - 1))
    T, N = array.shape

    if dim <= 1 or patt_time <= 0:
        raise ValueError("Dim mist be > 1 and length of delay vector smaller "
                         "array length.")

    patt = np.zeros((patt_time, N), dtype='int32')
    weights_array = np.zeros((patt_time, N), dtype='float64')

    patt_mask = np.zeros((patt_time, N), dtype='int32')

    # Precompute factorial for c-code... patterns of dimension
    # larger than 10 are not supported
    fac = factorial(np.arange(10)).astype('int32')

    # _get_patterns assumes mask=0 to be a masked value
    array_mask = (array_mask == False).astype('int32')

    (patt, patt_mask, weights_array) = _get_patterns(array, array_mask, patt, patt_mask, weights_array, dim, step, fac, N, T)

    weights_array = np.asarray(weights_array)
    patt = np.asarray(patt)
    # Transform back to mask=1 implying a masked value
    patt_mask = np.asarray(patt_mask) == False

    if weights:
        return patt, patt_mask, patt_time, weights_array
    else:
        return patt, patt_mask, patt_time


def quantile_bin_array(data, bins=6):
    """Returns symbolified array with equal-quantile binning.

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    bins : int, optional (default: 6)
        Number of bins.

    Returns
    -------
    symb_array : array
        Converted data of integer type.
    """
    T, N = data.shape

    # get the bin quantile steps
    bin_edge = int(np.ceil(T / float(bins)))

    symb_array = np.zeros((T, N), dtype='int32')

    # get the lower edges of the bins for every time series
    edges = np.sort(data, axis=0)[::bin_edge, :].T
    bins = edges.shape[1]

    # This gives the symbolic time series
    symb_array = (data.reshape(T, N, 1) >= edges.reshape(1, N, bins)).sum(
        axis=2) - 1

    return symb_array.astype('int32')


def var_process(parents_neighbors_coeffs, T=1000, use='inv_inno_cov',
                verbosity=0, initial_values=None):
    """Returns a vector-autoregressive process with correlated innovations.

    Wrapper around var_network with possibly more user-friendly input options.

    DEPRECATED. Will be removed in future.
    """
    print("data generating models are now in toymodels folder: "
         "from tigramite.toymodels import structural_causal_processes as toys.")
    return None

def structural_causal_process(links, T, noises=None, 
                        intervention=None, intervention_type='hard',
                        seed=None):
    """Returns a structural causal process with contemporaneous and lagged
    dependencies.

    DEPRECATED. Will be removed in future.
    """
    print("data generating models are now in toymodels folder: "
         "from tigramite.toymodels import structural_causal_processes as toys.")
    return None

#imported and mentioned in a different way
###Custom sklearn-based transformations###

class StandardTotalVarianceScaler(StandardScaler):
    '''As in sklearn's StandardScaler, but total variance is scaled rather  
    than feature variance. Total variance of an array A_ij with I samples  
    and J features is defined to be sum_ij(a_ij)/I.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, X, y=None, sample_weight=None, preserve_relative_scaling=True):
        #When preserve_relative_scaling is True, only the total variance of the entire 
        #dataset is used to determine a constant scaling for all columns. When False,
        #columns are normalized individually and then further scaled down to account for
        #the number of columns.
        T = super().fit(X, sample_weight=sample_weight)
        if preserve_relative_scaling:
            #The original StandardScaler checks for constant features
            #and will have trouble if all features are constant
            #here, we care only about the total variance, which should not
            #have this problem.
            T.scale_ = np.sqrt(np.sum(T.var_))*np.ones(T.scale_.shape)
        else:
            #Usually np.sqrt(var). Here, we artificially inflate
            #this by np.sqrt(the number of features) so the data 
            #will be further scaled down for a total variance of 1.
            #as implemented here, this still scales features individually.
            #could instead scale the whole vector by the total variance,
            #and thus preserve relative scaling within the vector.
            T.scale_ *= np.sqrt(T.n_features_in_)
        return T


if __name__ == '__main__':
    
    from tigramite.toymodels.structural_causal_processes import structural_causal_process
    ## Generate some time series from a structural causal process
    def lin_f(x): return x
    def nonlin_f(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))

    links = {0: [((0, -1), 0.9, lin_f)],
             1: [((1, -1), 0.8, lin_f), ((0, -1), 0.3, nonlin_f)],
             2: [((2, -1), 0.7, lin_f), ((1, 0), -0.2, lin_f)],
             }

    random_state_1 = np.random.default_rng(seed=1)
    random_state_2 = np.random.default_rng(seed=2)
    random_state_3 = np.random.default_rng(seed=3)

    noises = [random_state_1.standard_normal, random_state_2.standard_normal, random_state_3.standard_normal]

    ens = 3
    data_ens = {}
    for i in range(ens):
        data, nonstat = structural_causal_process(links,
                T=100, noises=noises)
        data[10, 1] == 999.
        data_ens[i] = data
    # print(data.shape)

    frame = DataFrame(data_ens, missing_flag=999.,
        analysis_mode = 'multiple')

    print(frame.T)

    X=[(0, 0)]
    Y=[(0, 0)]
    Z=[(0, -3)]
    tau_max=5
    frame.construct_array(X, Y, Z, tau_max,
                        extraZ=None,
                        mask=None,
                        mask_type=None,
                        return_cleaned_xyz=False,
                        do_checks=True,
                        cut_off='2xtau_max',
                        verbosity=4)
