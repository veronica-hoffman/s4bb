"""
===========================
Bandpower covariance matrix
===========================

Bandpower covariance matrix can be defined as a simple static matrix or a more
complicated object that can calculate bandpower covariance while varying noise
and/or signal levels. This implemented with the BpCov base class and derived
classes.

All bandpower covariance classes should include the following instance
variables and methods:
* maplist  : instance variable containing a list of MapDef objects
* nbin     : instance variable specifying the number of ell bins
* nmap()   : method that returns the number of maps, i.e. len(maplist)
* nspec()  : method that returns the number of spectra, i.e. nmap*(nmap+1)/2
* get()    : method that returns the bandpower covariance matrix
* select() : returns a new covariance matrix object defined for a subset of
             maps and/or ell bins

The bandpower covariance matrix returned by the get() function is organized
into blocks by ell bin. The size of each block is Nspec x Nspec, where
Nspec = Nmap * (Nmap + 1) / 2. Within a block, the spectra follow vecp
ordering, as discussed in spectra.py (which also contains helper functions).

"""

import numpy as np
from .util import specind, specgen, MapDef

class BpCov():
    """
    A BpCov object contains the bandpower covariance matrix for the full set
    of auto and cross spectra derived from a set of maps.

    This class can be used for a static bandpower covariance matrix or as a
    base class for more complex bpcm constructions.

    """
    
    def __init__(self, maplist, nbin, matrix=None):
        """
        Creates a BpCov object for specified list of maps and number of ell
        bins.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define the bandpowers.
        nbin : int
            Number of ell bins, assumed to be the same for all spectra.
        matrix : array, shape (N,N), optional
            Array specifying a bandpower covariance matrix. This could be
            calculated analytically or derived from sims. The shape of this
            array must be (N,N), where N = nbin * nmap * (nmap + 1) / 2.

        """
        
        self.maplist = maplist
        self.nbin = nbin
        self.matrix = {}
        if matrix is not None:
            self.set(matrix)

    @classmethod
    def from_xspec(cls, spec):
        """
        Creates a BpCov object from the sample covariance of a set of spectra
        provided as an XSpec object.

        Parameters
        ----------
        spec : XSpec
            Object containing realization of the spectra, from which we will
            estimate the covariance matrix.

        Returns
        -------
        bpcm : BpCov object
            Bandpower covariance matrix estimated from the spectra.

        """

        # XSpec object stores spectra as array with shape (nspec, nbin, nrlz).
        # We need to reshape this to (nspec*nbin, nrlz), with spectra
        # incrementing quickly and ell bins incrementing slowly along axis 0.
        nspec = spec.nspec()
        nbin = spec.nbin()
        nrlz = spec.nrlz()
        specarr = np.reshape(spec.spec, (nspec * nbin, nrlz), 'F')
        # Calculate covariance.
        matrix = np.cov(specarr)
        return BpCov(spec.maplist, nbin, matrix=matrix)
    
    def nmap(self):
        """Returns the number of maps defined for this object."""
        
        return len(self.maplist)

    def nspec(self):
        """Returns the number of spectra defined for this object."""
        
        nmap = self.nmap()
        return nmap * (nmap + 1) // 2

    def set(self, matrix):
        """
        Assigns a (static) bandpower covariance matrix.

        Parameters
        ----------
        matrix : array, shape (N,N), optional
            Array specifying a bandpower covariance matrix. This could be
            calculated analytically or derived from sims. The shape of this
            array must be (N,N), where N = nbin * nmap * (nmap + 1) / 2.

        Returns
        -------
        None

        """
        
        # Check that matrix has the right size
        assert (self.nspec() * self.nbin == matrix.shape[0])
        assert (self.nspec() * self.nbin == matrix.shape[1])
        self.matrix['total'] = matrix
            
    def get(self, noffdiag=None):
        """
        Returns an array containing bandpower covariance matrix.

        Parameters
        ----------
        noffdiag : int, optional
            If set to a non-negative integer, then this parameter defines the
            maximum range in ell bins that is retained for offdiagonal
            bandpower covariance. For example, if noffdiag=1 then we keep the
            covariances between bin i and bins i-1, i, and i+1, but zero out
            the covariances with any other bins. Default value is None, which
            means to keep covariances between *all* ell bins.

        Returns
        -------
        M : array, shape (N,N)
            Bandpower covariance matrix. The shape of this array is (N,N),
            where N = nbin * nmap * (nmap + 1) / 2. If the noffdiag argument
            is set, then some offdiagonal blocks of this matrix may be set to
            zero.

        """

        # Returns the matrix with requested masking of off-diagonal blocks
        return self.matrix['total'] * self.mask_ell(noffdiag=noffdiag)

    def mask_ell(self, noffdiag=None):
        """
        Returns an array of ones with some offdiagonal blocks (optionally) set
        to zero.

        Parameters
        ----------
        noffdiag : int, optional
            If set to a non-negative integer, then this parameter defines the
            maximum range in ell bins that is retained for offdiagonal
            bandpower covariance. For example, if noffdiag=1 then we keep the
            covariances between bin i and bins i-1, i, and i+1, but zero out
            the covariances with any other bins. Default value is None, which
            means to keep covariances between *all* ell bins.

        Returns
        -------
        M : array, shape (N,N)
            Mask array with shape (N,N), where N = nbin * nmap * (nmap + 1) / 2.
            This array consists of blocks that are set to one if they are
            within noffdiag of the diagonal, or zero otherwise.
        
        """
        
        N = self.nspec() * self.nbin
        # By default, don't mask anything.
        if noffdiag is None:
            return np.ones(shape=(N,N))
        # If noffdiag is negative, set it to zero.
        if noffdiag < 0:
            noffdiag = 0
        # Otherwise, construct mask that is set to one for blocks that are
        # within noffdiag of the diagonal, zero otherwise.
        mask = np.zeros(shape=(N,N))
        for i in range(self.nbin):
            x0 = i * self.nspec()
            x1 = (i + 1) * self.nspec()
            y0 = max(0, i - noffdiag) * self.nspec()
            y1 = min(self.nbin, i + noffdiag + 1) * self.nspec()
            mask[x0:x1,y0:y1] = 1.0
        return mask
    
    def select(self, maplist=None, ellind=None):
        """
        Make a new BpCov object with selected maps and/or ell bins.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new BpCov object. Defaults to None,
            which means that the new BpCov object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new BpCov object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        bpcov_new : BpCov
            New BpCov object with updated maps and ell bins.        

        """

        # Process maplist argument.
        if maplist is None:
            # Not a deep copy, but I don't expect anyone to change the
            # MapDef objects out from under me.
            maplist = self.maplist.copy()
        # Process ellind argument.
        if ellind is None:
            ellind = range(self.nbin)

        # Determine mapping from old to new bandpower ordering.
        nmap_new = len(maplist)
        nspec_new = nmap_new * (nmap_new + 1) // 2
        ind0 = np.zeros(nspec_new)
        for (i,m0,m1) in specgen(nmap_new):
            i0 = self.maplist.index(maplist[m0])
            i1 = self.maplist.index(maplist[m1])
            ind0[i] = specind(self.nmap(), i0, i1)
        # Expand over selected ell bins.
        nbin_new = len(ellind)
        ind1 = np.zeros(nspec_new * nbin_new, dtype=int)
        for (i,x) in enumerate(ellind):
            ind1[i*nspec_new:(i+1)*nspec_new] = ind0 + x * self.nspec()
        # Apply this selection to the covariance matrix.
        matrix = self.matrix['total'][np.ix_(ind1,ind1)]

        # Return new BpCov object
        return BpCov(maplist, nbin_new, matrix)

class BpCov_signoi(BpCov):
    """
    Bandpower covariance matrix that separately tracks signal and noise
    degrees of freedom. 

    This allows for:
    - Additional conditioning of the covariance matrix by applying assumptions
      that signal and noise maps are statistically independent and (optionally)
      that noise maps for different bands are statistically independent.
    - Ability to rescale the signal part of the covariance matrix to obtain a
      matrix that applies for a different signal model than was simulated.
    - Ability to rescale the noise part of the covariance matrix for
      forecasting.

    """

    def __init__(self, maplist, nbin, components=None, sig_model=None):
        """
        Creates a BpCov_signoi object for specified list of maps and number
        of ell bins.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define possible output spectra.
        nbin : int
            Number of ell bins, assumed to be the same for all spectra.
        components : list of arrays, optional
            A list containing six bandpower covariance matrix components. Each
            of the six matrices should have shape (N,N), where
            N = nbin * nmap * (nmap + 1) / 2. The definition and ordering of
            the six arrays is:
          - components[0] = Signal-only covariance, i.e. terms like
                            Cov(si x sj, sk x sl).
          - components[1] = Noise-only covariance, i.e. terms like
                            Cov(ni x nj, nk x nl).
          - components[2] = Signal x noise covariance part 0, i.e. terms like
                            Cov(si x nj, sk x nl).
          - components[3] = Signal x noise covariance part 1, i.e. terms like
                            Cov(si x nj, nk x sl).
          - components[4] = Signal x noise covariance part 2, i.e. terms like
                            Cov(ni x sj, sk x nl).
          - components[5] = Signal x noise covariance part 3, i.e. terms like
                            Cov(ni x sj, nk x sl).
        sig_model : array, optional
            Array containing bandpower expectation values for the signal model
            that was used for the covariance matrix components. If the
            components were calculated from signal and noise sim bandpowers,
            then this argument should be the mean of the signal bandpowers.
            Shape should be (nspec,nbin).

        """

        # Call base class constructor for some initial setup.
        super().__init__(maplist, nbin)
        # Add dictionary for noise masks.
        self.mask = {}
        # Store covariance matrix components.
        if components is not None:
            self.set(components, sig_model)

    @classmethod
    def from_xspec(cls, maplist, spec):
        """
        Constructs a BpCov_signoi object from the full set of spectra obtained
        from signal simulations and noise simulations.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define the bandpowers.
        spec : XSpec object
            Every map from the maplist argument should show up twice in this
            XSpec object, one with simtype='signal' and again with
            simtype='noise'

        Returns
        -------
        bpcm : BpCov object
            Bandpower covariance matrix estimated from the spectra.        

        """

        # Size of the problem
        nmap = len(maplist)
        nspec = nmap * (nmap + 1) // 2
        nbin = spec.nbin()
        N = nspec * nbin # total number of bandpowers
        nrlz = spec.nrlz()

        # Make additional maplists by copying the maplist argument but with
        # simtype set to 'signal' and 'noise'.
        # We don't need to worry about bandpass or beam here.
        maplist_sig = []
        maplist_noi = []
        for m in maplist:
            maplist_sig.append(MapDef(m.name, m.field, simtype='signal',
                                      lensing_template=m.lensing_template))
            maplist_noi.append(MapDef(m.name, m.field, simtype='noise',
                                      lensing_template=m.lensing_template))
        
        # Get signal-only, noise-only, signal x noise, and noise x signal
        # spectra.
        spec_sig = np.zeros(shape=(N,nrlz))
        spec_noi = np.zeros(shape=(N,nrlz))
        spec_sn = np.zeros(shape=(N,nrlz))
        spec_ns = np.zeros(shape=(N,nrlz))
        for (i,m0,m1) in specgen(nmap):
            # signal x signal spectra
            isig = specind(spec.nmap(), spec.maplist.index(maplist_sig[m0]),
                           spec.maplist.index(maplist_sig[m1]))
            spec_sig[i:N:nspec,:] = spec.spec[isig,:,:]
            # noise x noise spectra
            inoi = specind(spec.nmap(), spec.maplist.index(maplist_noi[m0]),
                           spec.maplist.index(maplist_noi[m1]))
            spec_noi[i:N:nspec,:] = spec.spec[inoi,:,:]
            # signal x noise spectra
            isn = specind(spec.nmap(), spec.maplist.index(maplist_sig[m0]),
                          spec.maplist.index(maplist_noi[m1]))
            spec_sn[i:N:nspec,:] = spec.spec[isn,:,:]
            # noise x signal spectra
            ins = specind(spec.nmap(), spec.maplist.index(maplist_noi[m0]),
                          spec.maplist.index(maplist_sig[m1]))
            spec_ns[i:N:nspec,:] = spec.spec[ins,:,:]

        # Save the mean of signal sims.
        sig_mean = np.reshape(np.mean(spec_sig, axis=1), (nspec, nbin), 'F')
            
        # Calculate covariance matrix components
        sig = np.cov(spec_sig)
        noi = np.cov(spec_noi)
        # This covariance call creates a 2N x 2N covariance matrix that can be
        # split apart into sn0, sn1, sn2, and sn3.
        sn = np.cov(spec_sn, spec_ns)
        sn0 = sn[0:N,0:N]
        sn1 = sn[0:N,N:] # Need to check that sn1/sn2 are defined correctly.
        sn2 = sn[N:,0:N]
        sn3 = sn[N:,N:]

        return BpCov_signoi(maplist, nbin, [sig, noi, sn0, sn1, sn2, sn3],
                            sig_mean)

    def set(self, components, sig_model):
        """
        Assigns bandpower covariance matrix components.

        Parameters
        ----------
        components : list of arrays
            A list containing six bandpower covariance matrix components. Each
            of the six matrices should have shape (N,N), where
            N = nbin * nmap * (nmap + 1) / 2. The definition and ordering of
            the six arrays is:
          - components[0] = Signal-only covariance, i.e. terms like
                            Cov(si x sj, sk x sl).
          - components[1] = Noise-only covariance, i.e. terms like
                            Cov(ni x nj, nk x nl).
          - components[2] = Signal x noise covariance part 0, i.e. terms like
                            Cov(si x nj, sk x nl).
          - components[3] = Signal x noise covariance part 1, i.e. terms like
                            Cov(si x nj, nk x sl).
          - components[4] = Signal x noise covariance part 2, i.e. terms like
                            Cov(ni x sj, sk x nl).
          - components[5] = Signal x noise covariance part 3, i.e. terms like
                            Cov(ni x sj, nk x sl).
        sig_model : array, optional
            Array containing bandpower expectation values for the signal model
            that was used for the covariance matrix components. If the
            components were calculated from signal and noise sim bandpowers,
            then this argument should be the mean of the signal bandpowers.
            Shape should be (nspec,nbin).

        Returns
        -------
        None

        """

        # Record covariance matrix components
        keys = ['sig', 'noi', 'sn0', 'sn1', 'sn2', 'sn3']
        N = self.nspec() * self.nbin
        for (i,key) in enumerate(keys):
            # Check that each component has the right size.
            assert components[i].shape[0] == N
            assert components[i].shape[1] == N
            self.matrix[key] = components[i]
        # Record signal model bandpowers
        assert sig_model.shape == (self.nspec(), self.nbin)
        self.sig_model = sig_model
        # Update noise mask
        self.noise_mask()

    def noise_mask(self):
        """
        Constructs masks for covariance matrix terms with zero expectation value.
        
        TODO: Need to add the ability to turn on noise correlations between
              specified maps.
        
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        
        # Construct masks for a single ell bin.
        nspec = self.nspec()
        mask_sig = np.zeros(shape=(nspec,nspec))
        mask_noi = np.zeros(shape=(nspec,nspec))
        mask_sn0 = np.zeros(shape=(nspec,nspec))
        mask_sn1 = np.zeros(shape=(nspec,nspec))
        mask_sn2 = np.zeros(shape=(nspec,nspec))
        mask_sn3 = np.zeros(shape=(nspec,nspec))
        for (i,m0,m1) in specgen(self.nmap()):
            for (j,m2,m3) in specgen(self.nmap()):
                # Keep all signal-only terms
                mask_sig[i,j] = 1.0
                # Keep noise-only terms if
                #   (m0 == m2 and m1 == m3) *or*
                #   (m0 == m3 and m1 == m2)
                if ((m0 == m2) and (m1 == m3)) or ((m0 == m3) and (m1 == m2)):
                    mask_noi[i,j] = 1.0
                # Keep sn0 terms if (m1 == m3)
                if (m1 == m3):
                    mask_sn0[i,j] = 1.0
                # Keep sn1 terms if (m1 == m2)
                if (m1 == m2):
                    mask_sn1[i,j] = 1.0
                # Keep sn2 terms if (m0 == m3)
                if (m0 == m3):
                    mask_sn2[i,j] = 1.0
                # Keep sn3 terms if (m0 == m2)
                if (m0 == m2):
                    mask_sn3[i,j] = 1.0
        # Repeat these masks for all ell bins.
        nbin = self.nbin
        self.mask['sig'] = np.tile(mask_sig, (nbin,nbin))
        self.mask['noi'] = np.tile(mask_noi, (nbin,nbin))
        self.mask['sn0'] = np.tile(mask_sn0, (nbin,nbin))
        self.mask['sn1'] = np.tile(mask_sn1, (nbin,nbin))
        self.mask['sn2'] = np.tile(mask_sn2, (nbin,nbin))
        self.mask['sn3'] = np.tile(mask_sn3, (nbin,nbin))
        
    def get(self, sig_model=None, noffdiag=None, mask_noise=True):
        """
        Returns an array containing bandpower covariance matrix.

        Parameters
        ----------
        sig_model : array, optional
            If this argument is specified, then the signal terms of the
            bandpower covariance matrix are rescaled to match the provided
            signal model, Shape should be (nspec,nbin).
        noffdiag : int, optional
            If set to a non-negative integer, then this parameter defines the
            maximum range in ell bins that is retained for offdiagonal
            bandpower covariance. For example, if noffdiag=1 then we keep the
            covariances between bin i and bins i-1, i, and i+1, but zero out
            the covariances with any other bins. Default value is None, which
            means to keep covariances between *all* ell bins.
        mask_noise : bool, optional
            If True (default), then set to zero covariance matrix terms that
            have zero expectation value under the assumption that noise is
            independent between maps.

        Returns
        -------
        M : array, shape (N,N)
            Bandpower covariance matrix. The shape of this array is (N,N),
            where N = nbin * nmap * (nmap + 1) / 2. If the noffdiag argument
            is set, then some offdiagonal blocks of this matrix may be set to
            zero.

        """

        # Some convenience factors
        nmap = self.nmap()
        nspec = self.nspec()
        nbin = self.nbin
        keys = ['sig', 'noi', 'sn0', 'sn1', 'sn2', 'sn3']

        # Calculate signal rescale factors.
        scale = {}
        if sig_model is None:
            # No rescaling. Set all scale arrays to one.
            for key in keys:
                scale[key] = 1.0
        else:
            # Allocate arrays for scale factors.
            for key in keys:
                scale[key] = np.ones(self.matrix[key].shape)
            # Double loop over spectra to calculate scale factors for diagonal
            # blocks. There is no rescaling for noise-only terms, so these are
            # left as ones.
            for (i,m0,m1) in specgen(nmap):
                for (j,m2,m3) in specgen(nmap):
                    i02 = specind(nmap, m0, m2)
                    i13 = specind(nmap, m1, m3)
                    i03 = specind(nmap, m0, m3)
                    i12 = specind(nmap, m1, m2)
                    # Scale factor for signal-only term.
                    num = (sig_model[i02,:] * sig_model[i13,:] +
                           sig_model[i03,:] * sig_model[i12,:])
                    den = (self.sig_model[i02,:] * self.sig_model[i13,:] +
                           self.sig_model[i03,:] * self.sig_model[i12,:])
                    scale['sig'][i::nspec,j::nspec] = num / den
                    # Scale factor for signal x noise terms.
                    scale['sn0'][i::nspec,j::nspec] = sig_model[i02,:] / self.sig_model[i02,:]
                    scale['sn1'][i::nspec,j::nspec] = sig_model[i03,:] / self.sig_model[i03,:]
                    scale['sn2'][i::nspec,j::nspec] = sig_model[i12,:] / self.sig_model[i12,:]
                    scale['sn3'][i::nspec,j::nspec] = sig_model[i13,:] / self.sig_model[i13,:]
            # Scale factor for off-diagonal blocks is geometric mean of scale
            # factors for on-diagonal blocks.
            for i in range(nbin):
                # Get blocks for ell bin i
                sigi = scale['sig'][i*nspec:(i+1)*nspec,i*nspec:(i+1)*nspec]
                sn0i = scale['sn0'][i*nspec:(i+1)*nspec,i*nspec:(i+1)*nspec]
                sn1i = scale['sn1'][i*nspec:(i+1)*nspec,i*nspec:(i+1)*nspec]
                sn2i = scale['sn2'][i*nspec:(i+1)*nspec,i*nspec:(i+1)*nspec]
                sn3i = scale['sn3'][i*nspec:(i+1)*nspec,i*nspec:(i+1)*nspec]
                # Second loop to cover upper triangle
                for j in range(i+1,nbin):
                    scale['sig'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec] = np.sqrt(
                        sigi * scale['sig'][j*nspec:(j+1)*nspec,j*nspec:(j+1)*nspec])
                    scale['sn0'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec] = np.sqrt(
                        sn0i * scale['sn0'][j*nspec:(j+1)*nspec,j*nspec:(j+1)*nspec])
                    scale['sn1'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec] = np.sqrt(
                        sn1i * scale['sn1'][j*nspec:(j+1)*nspec,j*nspec:(j+1)*nspec])
                    scale['sn2'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec] = np.sqrt(
                        sn2i * scale['sn2'][j*nspec:(j+1)*nspec,j*nspec:(j+1)*nspec])
                    scale['sn3'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec] = np.sqrt(
                        sn3i * scale['sn3'][j*nspec:(j+1)*nspec,j*nspec:(j+1)*nspec])
                    # ...and the lower triangle
                    scale['sig'][j*nspec:(j+1)*nspec,i*nspec:(i+1)*nspec] = (
                        scale['sig'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec])
                    scale['sn0'][j*nspec:(j+1)*nspec,i*nspec:(i+1)*nspec] = (
                        scale['sn0'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec])
                    scale['sn1'][j*nspec:(j+1)*nspec,i*nspec:(i+1)*nspec] = (
                        scale['sn1'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec])
                    scale['sn2'][j*nspec:(j+1)*nspec,i*nspec:(i+1)*nspec] = (
                        scale['sn2'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec])
                    scale['sn3'][j*nspec:(j+1)*nspec,i*nspec:(i+1)*nspec] = (
                        scale['sn3'][i*nspec:(i+1)*nspec,j*nspec:(j+1)*nspec])

        # Rescale matrix components, optionally apply masks, and add them up.
        matrix = np.zeros(self.matrix['sig'].shape)
        for key in keys:
            if mask_noise:
                matrix += self.matrix[key] * scale[key] * self.mask[key]
            else:
                matrix += self.matrix[key] * scale[key]
        # Return matrix with masking of offdiagonal blocks.
        return matrix * self.mask_ell(noffdiag=noffdiag)
    
    def select(self, maplist=None, ellind=None):
        """
        Make a new BpCov_signoi object with selected maps and/or ell bins.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new BpCov object. Defaults to None,
            which means that the new BpCov object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new BpCov object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        bpcov_new : BpCov_signoi
            New BpCov_signoi object with updated maps and ell bins.        

        """

        # Process maplist argument.
        if maplist is None:
            # Not a deep copy, but I don't expect anyone to change the
            # MapDef objects out from under me.
            maplist = self.maplist.copy()
        # Process ellind argument.
        if ellind is None:
            ellind = range(self.nbin)

        # Determine mapping from old to new bandpower ordering.
        nmap_new = len(maplist)
        nspec_new = nmap_new * (nmap_new + 1) // 2
        ind0 = np.zeros(nspec_new, dtype=int)
        for (i,m0,m1) in specgen(nmap_new):
            i0 = self.maplist.index(maplist[m0])
            i1 = self.maplist.index(maplist[m1])
            ind0[i] = specind(self.nmap(), i0, i1)
        # Expand over selected ell bins.
        nbin_new = len(ellind)
        ind1 = np.zeros(nspec_new * nbin_new, dtype=int)
        for (i,x) in enumerate(ellind):
            ind1[i*nspec_new:(i+1)*nspec_new] = ind0 + x * self.nspec()

        # Apply this selection to all matrix components.
        ix_ = np.ix_(ind1, ind1)
        components = [self.matrix['sig'][ix_], self.matrix['noi'][ix_],
                      self.matrix['sn0'][ix_], self.matrix['sn1'][ix_],
                      self.matrix['sn2'][ix_], self.matrix['sn3'][ix_]]
        # And to the covariance matrix signal model.
        sig_model = self.sig_model[np.ix_(ind0,ellind)]

        # Create new BpCov_signoi object
        return BpCov_signoi(maplist, nbin_new, components, sig_model)
