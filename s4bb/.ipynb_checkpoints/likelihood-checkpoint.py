"""
===========
Likelihoods
===========

"""

import warnings
import numpy as np
from scipy.optimize import minimize
from .util import vecp_to_matrix, matrix_to_vecp

class Likelihood():
    """
    The Likelihood object contains theory model(s), bandpower covariance matrix,
    and other information necessary to calculate the likelihood for provided
    bandpowers.

    """
    
    def __init__(self, maplist, bias=None, bpcm=None, models=[]):
        """
        Construct a new Likelihood object

        Parameters
        ----------
        maplist : list of MapDef objects
        bias : XSpec
        bpcm : BpCov
        models : list of Model objects

        """
        
        self.maplist = maplist
        self.bias = None
        self.bpcm = None
        self.set_models(models)        
        if bias is not None:
            self.set_bias(bias)
        if bpcm is not None:
            self.set_bpcm(bpcm)
        # Use compute_fiducial_bpcm method to set this.
        self.fiducial = None

    def set_bias(self, bias):
        """
        Assigns the noise bias that must be added to theory expectation
        values before comparing to data.

        Parameters
        ----------
        bias : XSpec
            XSpec object containing noise bias bandpowers for all auto and
            cross spectra. Note that noise biases must be specified for cross
            spectra, even if they are zero.

        Returns
        -------
        None

        """
        
        assert bias.maplist == self.maplist
        if self.bpcm is not None:
            assert bias.nbin() == self.bpcm.nbin
        for model in self.models:
            assert bias.nbin() == model.nbin()
        self.bias = bias

    def set_bpcm(self, bpcm):
        """
        Assigns the bandpower covariance matrix object used to calculate
        covariance of auto and cross spectra.

        Parameters
        ----------
        bpcm : BpCov
            BpCov object (or derived class) that can calculate the bandpower
            covariance matrix for all auto and cross spectra.

        Returns
        -------
        None

        """
        
        assert bpcm.maplist == self.maplist
        if self.bias is not None:
            assert bpcm.nbin == self.bias.nbin()
        for model in self.models:
            assert bpcm.nbin == model.nbin()
        self.bpcm = bpcm

    def set_models(self, models):
        """
        Assigns the theory model(s) used to calculate bandpower expectation
        values.

        Parameters
        ----------
        models : list of Model objects
            A list containing one or more objects of Model or derived class.

        Returns
        -------
        None

        """
        
        for model in models:
            assert model.maplist == self.maplist
            if self.bias is not None:
                assert model.nbin() == self.bias.nbin()
            if self.bpcm is not None:
                assert model.nbin() == self.bpcm.nbin
        self.models = models

    def select(self, maplist=None, ellind=None):
        """
        Make a new Likelihood object with selected maps and/or ell bins.

        Note that this method wipes out any existing fiducial model calculation.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new Likelihood object. Defaults to None,
            which means that the new Likelihood object will have the same map list
            as the existing object.
        ellind : list, optional
            List of ell bins to keep for the new Likelihood object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        lik_new : Likelihood
            New Likelihood object with updated maps and ell bins.        

        """

        # Call select methods of the objects that make up the Likelihood.
        new_bias = self.bias.select(maplist=maplist, ellind=ellind)
        new_bpcm = self.bpcm.select(maplist=maplist, ellind=ellind)
        new_models = []
        for mod in self.models:
            new_models.append(mod.select(maplist=maplist, ellind=ellind))
        # Then construct new Likelihood object.
        lik_new = Likelihood(new_bias.maplist, bias=new_bias,
                             bpcm=new_bpcm, models=new_models)
        return lik_new
        
    def nmap(self):
        """Get number of maps used for this likelihood"""
        
        return len(self.maplist)

    def nspec(self):
        """Get number of spectra used for this likelihood"""
        
        n = self.nmap()
        return (n * (n + 1) // 2)

    def nbin(self):
        """Get number of ell bins used for this likelihood"""
        
        # There are a variety of ways to get nbin, some of which might not be
        # defined. They should all be consistent.
        try:
            return self.bias.nbin()
        except AttributeError:
            pass
        try:
            return self.bpcm.nbin
        except AttributeError:
            pass
        if len(self.models) > 0:
            return self.models[0].nbin()
        return None
        
    def nparam(self):
        """Get number of model parameters used for this likelihood"""
        
        n = 0
        for model in models:
            n += model.nparam()
        return n

    def param_names(self):
        """Get list of model parameter names"""
        
        names = []
        for model in self.models:
            for param in model.param_names():
                names.append(param)
        return names

    def param_list_to_dict(self, param_list):
        """Convert list of model parameter values to dictionary form"""
        
        param_dict = {}
        i = 0
        for model in self.models:
            n = model.nparam()
            temp_dict = model.param_list_to_dict(param_list[i:i+n])
            for (key,val) in temp_dict.items():
                param_dict[key] = val
            i += n
        return param_dict

    def param_dict_to_list(self, param_dict):
        """Convert model parameter dictionary to list of parameter values"""
        
        param_list = []
        for model in self.models:
            temp_list = model.param_dict_to_list(param_dict)
            for param in temp_list:
                param_list.append(param)
        return param_list

    def expv(self, param, include_bias=True):
        """
        Calculate model expectation values for specified parameters.

        Parameters
        ----------
        param : list or dict
            Model parameters, either in list or dict form. See
            param_list_to_dict and param_dict_to_list to convert between forms.
        include_bias : bool, optional
            By default, expectation values will include noise bias. Set to
            False if you want the theory expectation values only, with no bias.

        Returns
        -------
        expval : array, shape=(nspec,nbin)
            Array of bandpower expectation values

        """
        
        expval = np.zeros(shape=(self.nspec(), self.nbin()))
        if type(param) == dict:
            for model in self.models:
                expval += model.expv(param)
        elif type(param) == list:
            i = 0
            for model in self.models:
                n = model.nparam()
                expval += model.expv(param[i:i+n])
                i += n
        else:
            raise AttributeError('param must be list or dict')
        if include_bias:
            expval += self.bias[:,:,0]
        return expval
            
    def compute_fiducial_bpcm(self, expv, noffdiag=None, mask_noise=True):
        """
        Calculates and stores bandpower covariance matrix for chosen signal 
        model.

        Parameters
        ----------
        expv : array
            Array of bandpower expectation values with shape (N,M), where N is
            the number of spectra and M is the number of ell bins.
        noffdiag : int, optional
            Number of off-diagonal blocks to keep for the bandpower covariance
            matrix. These blocks contain covariance between different ell bins.
            If this parameter is set to 0, then we only keep covariance for
            bandpowers of the same ell bin. If parameter is set to 1, then we
            keep covariance between bandpowers of adjacent ell bins, etc.
            Default value is None, which means that we keep covariance between
            all bandpowers.
        mask_noise : bool, optional
            If True (default), then set some bandpower covariance matrix terms
            to zero under assumption that noise is independent between different
            maps. If False, do not make this assumption.

        Returns
        -------
        None

        """

        # Overwrites any previous compute_fiducial_bpcm calculation.
        self.fiducial = {}
        # Record fiducial model bandpowers in matrix form (include noise bias)
        self.fiducial['Cf'] = vecp_to_matrix(expv + self.bias[:,:,0])
        # Also, take the square root (cholesky decomposition) in each ell bin
        self.fiducial['Cf12'] = np.transpose(np.linalg.cholesky(
            np.transpose(self.fiducial['Cf'], (2,0,1))), (1,2,0))
        # Get bandpower covariance matrix.
        try:
            # BpCov_signoi
            self.fiducial['M'] = self.bpcm.get(sig_model=expv, noffdiag=noffdiag,
                                               mask_noise=mask_noise)
        except TypeError:
            # BpCov base class
            raise RuntimeWarning('BpCov does not support signal scaling\n' +
                                 'Should use BpCov_signoi class instead')
            self.fiducial['M'] = self.bpcm.get(noffdiag=noffdiag)
        # Calculate inverse bandpower covariance matrix.
        # Using pseudo-inverse because this matrix is very large and often
        # contains some very small singular values.
        self.fiducial['Minv'] = np.linalg.pinv(self.fiducial['M'])

    def chi2(self, expv, data):
        """
        Calculates chi^2 for specified model expectation values and data.

        Parameters
        ----------
        expv : array
            Array of bandpower expectation values with shape (N,M), where N is
            the number of spectra and M is the number of ell bins.
        data : array
            Array of data bandpowers with same shape as expv.

        Returns
        -------
        logL : float
            chi^2, which is -2 * log(likelihood) for Gaussian likelihood.

        """

        # Take difference of data and expectation value.
        x = data - expv
        # Concatenate ell bins to get a long vector that matches bpcm ordering.
        x = np.reshape(x, (np.prod(dev.shape),), order='F')
        # Calculate chi^2
        logL = x @ self.fiducial['Minv'] @ x
        return logL

    def hl_likelihood(self, expv, data):
        """
        Calculates Hamimeche-Lewis likelihood for specified model expectation
        values and data.

        Parameters
        ----------
        expv : array
            Array of bandpower expectation values with shape (N,M), where N is
            the number of spectra and M is the number of ell bins.
        data : array
            Array of data bandpowers with same shape as expv.

        Returns
        -------
        logL : float
            -2 * log(likelihood). See equations 47-49 of Hamimeche-Lewis (2008),
            PRD 77, 103013.

        """

        # Convert expv and data from vecp to matrix form.
        C = vecp_to_matrix(expv)
        Chat = vecp_to_matrix(data)
        nmap = C.shape[0]
        nspec = nmap * (nmap + 1) // 2
        # Transform expv and data to Gaussian-like quantity.
        nbin = self.nbin()
        X = np.zeros(C.shape)
        for i in range(nbin):
            try:
                Cn12 = np.linalg.cholesky(np.linalg.inv(C[:,:,i]))
                (eigval, eigvec) = np.linalg.eigh(np.transpose(Cn12) @ Chat[:,:,i] @ Cn12)
            except np.linalg.LinAlgError:
                # If Cn12 is not positive definite or we can't diagonalize the
                # bandpower ratio matrix, return a very large value of -2*log(L)
                # to indicate that this a bad part of parameter space.
                return 1e10
            g = np.sign(eigval - 1.0) * np.sqrt(2 * (eigval - np.log(eigval) - 1.0))
            X[:,:,i] = (self.fiducial['Cf12'][:,:,i] @
                        (eigvec @ np.diag(g) @ np.transpose(eigvec)) @
                        np.transpose(self.fiducial['Cf12'][:,:,i]))
        # Convert X to vecp ordering and then concatenate ell bins to get a
        # long vector that matches our bpcm.
        Xv = np.reshape(matrix_to_vecp(X), (nspec*nbin,), order='F')
        # Calculate -2*log(L) as chi^2
        logL = Xv @ self.fiducial['Minv'] @ Xv
        return logL

    def mlsearch(self, data, start, free=None, limits={},
                 method='L-BFGS-B', options={}):
        """
        Maximum likelihood search to find model parameters that best match data.

        Using L-BFGS-B minimizer from scipy.optimize. In the future, could
        add ability to use other minimizers, specify convergence options, etc.

        Parameters
        ----------
        data : array
            Array of data bandpowers with shape (N,M), where N is the number of
            spectra and M is the number of ell bins.
        start : list or dict
            Model parameters to use as the starting point for the search. These
            can either be in list or dict form. See param_list_to_dict and
            param_dict_to_list to convert between forms.
        free : list, optional
            List containing the names of parameters that are allowed to vary in
            the maximum likelihood search. Default is None, which means that
            all model parameters are allowed to vary.
        limits : dict, optional
            Dictionary specifying allowed ranges of parameters. The dict should
            be keyed on parameter names with values set to two-element lists
            containing the lower and upper limits. One or both of these limits
            can be set to None to indicate no bound. Parameters that are not
            found in the dict are assumed to be unbounded.
        method : str, optional
            Choice of minimizer. See scipy.optimize.minimize for details.
        options : dict, optional
            Dictionary of options to pass to the minimizer.

        Returns
        -------
        result : dict
            Dictionary containing best-fit parameters (for parameters that are
            free to vary) and starting parameters (for parameters that are
            fixed).
        fval : float
            Minimum value of -2*log(L) found in the search.
        status : int
            Termination status of the optimizer. 0 indicates success. Other
            values indicate different reasons for failure to converge.
        
        """

        # If start point is provided as a list, convert to dict.
        if type(start) is not dict:
            start_dict = self.param_list_to_dict(start)
        else:
            start_dict = start
        # If free=None, then vary all parameters
        if free is None:
            free = self.param_names()

        # Define function to be minimized
        def minfun(p, lik, data, start_dict, free):
            # Merge free and fixed parameters
            for (key,val) in zip(free,p):
                start_dict[key] = val
            # Get model expectation values.
            expval = lik.expv(start_dict, include_bias=True)
            # Calculate likelihood.
            return lik.hl_likelihood(expval, data)

        # Parameter limits provided to minimize function
        bounds = []
        for param in free:
            try:
                bounds.append(limits[param])
            except KeyError:
                bounds.append((None, None))

        # Run minimizer
        guess = [start_dict[key] for key in free]
        fit = minimize(minfun, guess, bounds=bounds, 
                       args=(self, data, start_dict.copy(), free),
                       method=method, options=options)
        result = start_dict.copy()
        for (key,val) in zip(free,fit.x):
            result[key] = val
        return (result, fit.fun, fit.status)
