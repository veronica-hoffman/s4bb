"""
============================
Cross-spectrum theory models
============================

"""

import numpy as np
from .util import specgen
from .bandpass import GHz, h, k, delta_Tcmb_to_delta_I, Bandpass

class Model():
    """
    Base class for theory models

    """

    # Class variable defines the number of parameters in the model. This should
    # be overwritten to an appropriate value for all derived classes!
    nparameters = 0
    
    def __init__(self, maplist, wf):
        """
        Constructor for Model base class (not an actual model!)

        Parameters
        ----------
        maplist : list of MapDef objects
        wf : BPWF object

        """
        
        self.maplist = maplist
        # Check that bpwf has compatible maplist
        assert wf.maplist == maplist
        self.wf = wf

    def nmap(self):
        """Get the number of maps for this model"""
        
        return len(self.maplist)

    def nspec(self):
        """Get the number of spectra for this model"""
        
        N = self.nmap()
        return (N * (N + 1) // 2)

    def nbin(self):
        """Get the number of ell bins for this model"""
        
        return self.wf.nbin

    def nparam(self):
        """Get the number of parameters for this model"""
        
        return self.__class__.nparameters

    def param_names(self):
        """Get list of parameter names"""

        return []
    
    def param_list_to_dict(self, param_list):
        """
        Convert list of parameters to a structured dictionary

        Parameters
        ----------
        param_list : list
            List of 0 parameters in following order:

        Returns
        -------
        param_dict : dict
            Dictionary of parameters

        """
        
        # No parameters, return empty dict
        return {}

    def param_dict_to_list(self, param_dict):
        """
        Convert parameters dictionary to an ordered list

        Parameters
        ----------
        param_dict : dict
            Dictionary of parameters with the following 0 keys:

        Returns
        -------
        param_list : list
            List of 0 parameters in following order:

        """
        
        # No parameters, return empty list
        return []

    def theory_spec(self, param, m0, m1, lmax=None):
        """
        Return six theory spectra (TT,EE,BB,TE,EB,TB) for specified maps.

        Note that the m0 and m1 indexes refer to MapDef objects that specify
        both the map name and field. So if, for example, these both point to
        B-mode maps, then you might expect this function to return a BB spectrum
        only. Instead we return all six spectra because there might be leakage
        from TT->BB, EE->BB, etc, that is calculated during application of the
        bandpower window functions.

        Parameters
        ----------
        param : list or dict
            Model parameters in the form of an ordered list or a dict. See
            param_dict_to_list docstring for details of the ordering.
        m0 : int
            Index of the first map in the cross spectrum.
        m1 : int
            Index of the second map in the cross spectrum.
        lmax : int
            Maximum ell to evaluate specta. Default is the max ell value of the
            model bandpower window functions.

        Returns
        -------
        spec : array, shape=(6,lmax+1)
            Model theory spectra (TT,EE,BB,TE,EB,TB) for the specified
            parameters.

        """

        # Check that m0,m1 are valid.
        assert m0 < self.nmap()
        assert m1 < self.nmap()
        # If parameters are supplied as a dict, convert to list.
        if type(param) == dict:
            param = self.param_dict_to_list(param)
        assert len(param) == self.nparam()
        # Get lmax
        if lmax is None:
            lmax = self.wf.lmax()
        # Empty model
        return np.zeros(shape=(6,lmax+1))
        
    def expv(self, param):
        """
        Return model expectation values for specified parameters

        Parameters
        ----------
        param : list or dict
            Model parameters in the form of an ordered list or a dict. See
            param_dict_to_list docstring for details of the ordering.

        Returns
        -------
        expval : array, shape=(nspec,nbin)
            Array containing model expectation values for all spectra in all
            ell bins.

        """

        # Allocate array for expectation values.
        expval = np.zeros(shape=(self.nspec(),self.nbin()))
        # Loop over spectra
        for (i,m0,m1) in specgen(self.nmap()):
            spec = self.theory_spec(param, m0, m1)
            # Use window functions to calculate how the six spectra couple to
            # these bandpowers.
            for (j,spectype) in enumerate(['TT','EE','BB','TE','EB','TB']):
                expval[i,:] += self.wf.expv(spec[j,:], spectype, i)
        return expval

    def select(self, maplist=None, ellind=None):
        """
        Make a new Model object with selected maps and/or ell bins.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new Model object. Defaults to None,
            which means that the new object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new Model object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        mod_new : Model
            New Model object with updated maps and ell bins.

        """

        wf_new = self.wf.select(maplist, ellind)
        return Model(maplist, wf_new)

class Model_cmb(Model):
    """
    CMB spectra with r and Alens parameters

    """

    # Two parameter model: r, Alens
    nparameters = 2

    def __init__(self, maplist, wf, Cl_unlens, Cl_lens, Cl_tensor, rval, lmin=0):
        """
        Constructor

        Parameters
        ----------
        maplist : list of MapDef objects
        wf : BPWF object
        Cl_unlens : array
        Cl_lens : array
        Cl_tensor : array
        rval : float

        """

        # Invoke base class constructor
        super().__init__(maplist, wf)
        # Each of the Cl inputs should either contain four (TT,EE,BB,TE) or
        # six (TT,EE,BB,TE,EB,TB) spectra. If lmin > 0, pad the beginning of
        # the spectra with zeros.
        # Unlensed CMB spectra
        self.Cl_unlens = np.zeros(shape=(6,lmin+Cl_unlens.shape[1]))
        if Cl_unlens.shape[0] == 4:
            self.Cl_unlens[0:4,lmin:] = Cl_unlens
        elif Cl_unlens.shape[0] == 6:
            self.Cl_unlens[:,lmin:] = Cl_unlens
        else:
            raise ValueError('Cl_unlens must contain 4 or 6 spectra')
        # Lensed CMB spectra
        self.Cl_lens = np.zeros(shape=(6,lmin+Cl_lens.shape[1]))
        if Cl_lens.shape[0] == 4:
            self.Cl_lens[0:4,lmin:] = Cl_lens
        elif Cl_lens.shape[0] == 6:
            self.Cl_lens[:,lmin:] = Cl_lens
        else:
            raise ValueError('Cl_lens must contain 4 or 6 spectra')
        # Tensor CMB spectra
        self.rval = rval
        self.Cl_tensor = np.zeros(shape=(6,lmin+Cl_tensor.shape[1]))
        if Cl_tensor.shape[0] == 4:
            self.Cl_tensor[0:4,lmin:] = Cl_tensor
        elif Cl_tensor.shape[0] == 6:
            self.Cl_tensor[:,lmin:] = Cl_tensor
        else:
            raise ValueError('Cl_tensor must contain 4 or 6 spectra')

    def param_names(self):
        """Get list of parameter names"""

        return ['r', 'Alens']

    def param_list_to_dict(self, param_list):
        """
        Convert list of parameters to a structured dictionary

        Parameters
        ----------
        param_list : list
            List of 2 parameters in following order: r, Alens

        Returns
        -------
        param_dict : dict
            Dictionary of parameters

        """
        
        param_dict = {'r': param_list[0],
                      'Alens': param_list[1]}
        return param_dict

    def param_dict_to_list(self, param_dict):
        """
        Convert parameters dictionary to an ordered list

        Parameters
        ----------
        param_dict : dict
            Dictionary of parameters with the following 2 keys: 'r', 'Alens'

        Returns
        -------
        param_list : list
            List of 2 parameters in following order: r, Alens

        """
        
        param_list = [param_dict['r'], param_dict['Alens']]
        return param_list

    def theory_spec(self, param, m0, m1, lmax=None):
        """
        Return six theory spectra (TT,EE,BB,TE,EB,TB) for specified maps.

        Note that the m0 and m1 indexes refer to MapDef objects that specify
        both the map name and field. So if, for example, these both point to
        B-mode maps, then you might expect this function to return a BB spectrum
        only. Instead we return all six spectra because there might be leakage
        from TT->BB, EE->BB, etc, that is calculated during application of the
        bandpower window functions.

        Parameters
        ----------
        param : list or dict
            Model parameters in the form of an ordered list or a dict. See
            param_dict_to_list docstring for details of the ordering.
        m0 : int
            Index of the first map in the cross spectrum.
        m1 : int
            Index of the second map in the cross spectrum.
        lmax : int
            Maximum ell to evaluate specta. Default is the max ell value of the
            model bandpower window functions.

        Returns
        -------
        spec : array, shape=(6,lmax+1)
            Model theory spectra (TT,EE,BB,TE,EB,TB) for the specified
            parameters.

        """

        # Check that m0,m1 are valid.
        assert m0 < self.nmap()
        assert m1 < self.nmap()
        # If parameters are supplied as a dict, convert to list.
        if type(param) == dict:
            param = self.param_dict_to_list(param)
        assert len(param) == self.nparam()
        # Get lmax
        if lmax is None:
            lmax = self.wf.lmax()
        # Allocate array for spectra
        spec = np.zeros(shape=(6,lmax+1))
        # Check whether theory spectra extend all the way to lmax
        N = min(lmax + 1, self.Cl_unlens.shape[1],
                self.Cl_lens.shape[1], self.Cl_tensor.shape[1])
        # If m0 or m1 is a lensing template, then the model will contain
        # lensing B modes only.
        if self.maplist[m0].lensing_template or self.maplist[m1].lensing_template:
            spec[2,0:N] = param[1] * self.Cl_lens[2,0:N]
        else:
            # Combine lensed and unlensed CMB spectra using Alens parameter
            spec[:,0:N] = ((1 - param[1]) * self.Cl_unlens[:,0:N] +
                           param[1] * self.Cl_lens[:,0:N])
            # Add in tensor CMB spectra using r parameter
            spec[:,0:N] += (param[0] / self.rval) * self.Cl_tensor[:,0:N]
        # Done
        return spec

    def select(self, maplist=None, ellind=None):
        """
        Make a new Model_cmb object with selected maps and/or ell bins.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new Model object. Defaults to None,
            which means that the new object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new Model object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        mod_new : Model_cmb
            New Model_cmb object with updated maps and ell bins.

        """

        wf_new = self.wf.select(maplist=maplist, ellind=ellind)
        return Model_cmb(wf_new.maplist, wf_new, self.Cl_unlens, self.Cl_lens,
                         self.Cl_tensor, self.rval)

class Model_fg(Model):
    """
    Polarized dust and synchrotron foreground model

    Include dust-sync correlation, dust decorrelation, sync decorrelation

    """

    # NOW 15 parameter model:
    #   Dust parameters: A_d, alpha_d, beta_d, T_d, EEBB_d
    #   Sync parameters: A_s, alpha_s, beta_s, EEBB_s
    #   Dust-sync correlation parameter: epsilon
    #   Dust decorrelation parameters: Delta_d, gamma_d
    #   Sync decorrelation parameters: Delta_s, gamma_s
    #   NEW: FREQUENCY SHIFT
    nparameters = 15

    def __init__(self, maplist, wf, ell_pivot=80.0,
                 dust_pivot=[353.0, 217.0], sync_pivot=[23.0, 33.0]):
        """
        Constructor

        Parameters
        ----------
        maplist : list of MapDef objects
        wf : BPWF object
        ell_pivot : float
        dust_pivot : list of 2 floats
        sync_pivot : list of 2 floats

        """

        # Invoke base class constructor
        super().__init__(maplist, wf)
        self.ell_pivot = ell_pivot
        self.dust_pivot = dust_pivot
        self.sync_pivot = sync_pivot

    def param_names(self): #MODIFIED TO HAVE NEW PARAMETER
        """Get list of parameter names"""
        
        return ['A_d', 'alpha_d', 'beta_d', 'T_d', 'EEBB_d',
                'A_s', 'alpha_s', 'beta_s', 'EEBB_s',
                'epsilon', 'Delta_d', 'gamma_d', 'Delta_s', 'gamma_s', 'freq_shift']

    def param_list_to_dict(self, param_list):
        """
        Convert list of parameters to a structured dictionary

        Parameters
        ----------
        param_list : list
            List of 14 parameters, in order returned by param_list method

        Returns
        -------
        param_dict : dict
            Dictionary of parameters

        """

        param_dict = {'A_d': param_list[0], 'alpha_d': param_list[1],
                      'beta_d': param_list[2], 'T_d': param_list[3],
                      'EEBB_d': param_list[4], 'A_s': param_list[5],
                      'alpha_s': param_list[6], 'beta_s': param_list[7],
                      'EEBB_s': param_list[8], 'epsilon': param_list[9],
                      'Delta_d': param_list[10], 'gamma_d': param_list[11],
                      'Delta_s': param_list[12], 'gamma_s': param_list[13],
                      'freq_shift': param_list[14]}
        return param_dict

    def param_dict_to_list(self, param_dict):
        """
        Convert parameters dictionary to an ordered list

        Parameters
        ----------
        param_dict : dict
            Dictionary of parameters

        Returns
        -------
        param_list : list
            List of 14 parameters, in order returned by param_list method

        """

        # Fill in default values for some parameters if they are not present
        # in the dict.
        param_list = []
        # Dust parameters
        param_list.append(param_dict['A_d'])
        param_list.append(param_dict['alpha_d'])
        param_list.append(param_dict['beta_d'])
        try:
            param_list.append(param_dict['T_d'])
        except KeyError:
            param_list.append(19.6)
        try:
            param_list.append(param_dict['EEBB_d'])
        except KeyError:
            param_list.append(2.0)
        # Sync parameters
        param_list.append(param_dict['A_s'])
        param_list.append(param_dict['alpha_s'])
        param_list.append(param_dict['beta_s'])
        try:
            param_list.append(param_dict['EEBB_s'])
        except KeyError:
            param_list.append(2.0)
        # Dust-sync correlation
        param_list.append(param_dict['epsilon'])
        # Dust decorrelation
        try:
            param_list.append(param_dict['Delta_d'])
        except KeyError:
            param_list.append(1.0)
        try:
            param_list.append(param_dict['gamma_d'])
        except KeyError:
            param_list.append(0.0)
        # Sync decorrelation
        try:
            param_list.append(param_dict['Delta_s'])
        except KeyError:
            param_list.append(1.0)
        try:
            param_list.append(param_dict['gamma_s'])
        except KeyError:
            param_list.append(0.0)
        try: #ADD frequency shift 
            param_list.append(param_dict['freq_shift'])
        except KeyError:
            param_list.append(0.0)
        # Finally done
        return param_list

    def dust_scale(self, bandpass, beta_d, T_d):
        """Calculates dust brightness relative to pivot frequency"""

        fn = lambda nu: nu**(3 + beta_d) / (np.exp(h * nu * GHz / (k * T_d)) - 1.0)
        scale = bandpass.bandpass_integral(fn) / fn(self.dust_pivot[0])
        conv = bandpass.cmb_unit_conversion() / delta_Tcmb_to_delta_I(self.dust_pivot[0])
        return (scale / conv)

    def sync_scale(self, bandpass, beta_s):
        """Calculates sync brightness relative to pivot frequency"""

        fn = lambda nu: nu**(2 + beta_s)
        scale = bandpass.bandpass_integral(fn) / fn(self.sync_pivot[0])
        conv = bandpass.cmb_unit_conversion() / delta_Tcmb_to_delta_I(self.sync_pivot[0])
        return (scale / conv)

    def decorr(self, ell, bandpass0, bandpass1, pivot, Delta, gamma):
        """Calculate correlation coefficient between two bands"""

        # Scaling based on frequency ratio between the two bands.
        # Uses bandpass average frequency, not a bandpass integral.
        nu0 = bandpass0.nu_eff()
        nu1 = bandpass1.nu_eff()
        scale_nu = (np.log(nu0 / nu1) / np.log(pivot[0] / pivot[1]))**2
        # Scaling in ell follows power law.
        scale_ell = (ell / self.ell_pivot)**gamma
        # Exponential function used to map correlation coefficients into the
        # range [0,1], even for the of high ell and/or large frequency ratio.
        # We use an analytic continuation for non-physical values of Delta that
        # are greater than 1.
        if Delta > 1:
            return (2.0 - np.exp(np.log(2.0 - Delta) * scale_nu * scale_ell))
        else:
            return np.exp(np.log(Delta) * scale_nu * scale_ell)
    
    def theory_spec(self, param, m0, m1, lmax=None):
        """
        Return six theory spectra (TT,EE,BB,TE,EB,TB) for specified maps.

        Note that the m0 and m1 indexes refer to MapDef objects that specify
        both the map name and field. So if, for example, these both point to
        B-mode maps, then you might expect this function to return a BB spectrum
        only. Instead we return all six spectra because there might be leakage
        from TT->BB, EE->BB, etc, that is calculated during application of the
        bandpower window functions.

        Parameters
        ----------
        param : list or dict
            Model parameters in the form of an ordered list or a dict. See
            param_dict_to_list docstring for details of the ordering.
        m0 : int
            Index of the first map in the cross spectrum.
        m1 : int
            Index of the second map in the cross spectrum.
        lmax : int
            Maximum ell to evaluate specta. Default is the max ell value of the
            model bandpower window functions.

        Returns
        -------
        spec : array, shape=(6,lmax+1)
            Model theory spectra (TT,EE,BB,TE,EB,TB) for the specified
            parameters.

        """

        # Check that m0,m1 are valid.
        assert m0 < self.nmap()
        assert m1 < self.nmap()
        # If parameters are supplied as a dict, convert to list.
        # Note that this function will fill in some parameters that might not
        # have been defined in the dict.
        if type(param) == dict:
            param = self.param_dict_to_list(param)
        assert len(param) == self.nparam()
        # Get lmax
        if lmax is None:
            lmax = self.wf.lmax()
        # Allocate array for spectra
        ell = np.arange(lmax + 1)
        spec = np.zeros(shape=(6,lmax+1))
        # No foreground response for lensing template
        if self.maplist[m0].lensing_template or self.maplist[m1].lensing_template:
            pass
        else:
            #ADDED FREQUENCY SHIFT PARAMETER HANDLING
            freq_shift = param[14]  # in GHz
        
            #worry about shifted bandpasses only if freq_shift is non-zero
            if freq_shift != 0.0:
                #shift the bandpass edges for both maps
                bp0 = Bandpass.tophat(
                    self.maplist[m0].bandpass.nu.min() + freq_shift,
                    self.maplist[m0].bandpass.nu.max() + freq_shift
                )
                bp1 = Bandpass.tophat(
                    self.maplist[m1].bandpass.nu.min() + freq_shift,
                    self.maplist[m1].bandpass.nu.max() + freq_shift
                )
            else:
                #otherwise use original bandpasses
                bp0 = self.maplist[m0].bandpass
                bp1 = self.maplist[m1].bandpass
            
            with np.errstate(divide='ignore'):
                dust_ell_scale = (ell / self.ell_pivot)**param[1]
                dust_ell_scale[ell == 0.0] = 0.0
            fdust0 = self.dust_scale(bp0, param[2], param[3])
            fdust1 = self.dust_scale(bp1, param[2], param[3])
            if param[10] == 1.0:
                dust_decorr = np.ones(ell.shape)
            else:
                dust_decorr = self.decorr(ell, bp0,
                                          bp1,
                                          self.dust_pivot, param[10], param[11])
            with np.errstate(divide='ignore'):
                sync_ell_scale = (ell / self.ell_pivot)**param[6]
                sync_ell_scale[ell == 0.0] = 0.0
            fsync0 = self.sync_scale(bp0, param[7])
            fsync1 = self.sync_scale(bp1, param[7])
            if param[12] == 1.0:
                sync_decorr = np.ones(ell.shape)
            else:
                sync_decorr = self.decorr(ell, bp0,
                                          bp1,
                                          self.sync_pivot, param[12], param[13])
            # BB
            BB_d = param[0] * fdust0 * fdust1 * dust_ell_scale * dust_decorr
            BB_s = param[5] * fsync0 * fsync1 * sync_ell_scale * sync_decorr
            BB_c = (param[9] * (fdust0 * fsync1 + fdust1 * fsync0) *
                    np.sqrt(param[0] * param[5] * dust_ell_scale *
                            sync_ell_scale * dust_decorr * sync_decorr))
            spec[2,:] = BB_d + BB_s + BB_c
            # EE -- include EE/BB ratio parameters for dust, sync
            spec[1,:] = (param[4] * BB_d + param[8] * BB_s +
                         np.sqrt(param[4] * param[8]) * BB_c)
            # No foreground TT,TE,EB,TB in this model
            # At some point, should do something about TT,TE at least
        # Done
        return spec

    def select(self, maplist=None, ellind=None):
        """
        Make a new Model_fg object with selected maps and/or ell bins.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new Model object. Defaults to None,
            which means that the new object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new Model object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        mod_new : Model_fg
            New Model_fg object with updated maps and ell bins.

        """

        wf_new = self.wf.select(maplist, ellind)
        return Model_fg(wf_new.maplist, wf_new, ell_pivot=self.ell_pivot,
                        dust_pivot=self.dust_pivot, sync_pivot=self.sync_pivot)
