"""
==========================
Bandpower window functions
==========================

Bandpower window functions describe the mapping of power from true sky signals
to measured bandpowers, including the effects of beams, filtering, partial sky
coverage, map apodization, etc.

We use a very general treatment of bandpower window functions, so that any
measured bandpower can have up to six window functions describing the transfer
functions from true sky TT, EE, BB, TE, EB, and TB. In practice, many of these
functions will be zero, e.g. TT->BB window function is zero in the absence of
temperature-to-polarization leakage.

Some conventions followed here:
* Expectation values are calculated as the theory spectrum multiplied by the
  window function and summed over ell values. The range of the summation is
  set by the ell values for which the window function is defined. All window
  functions are stored internally with ell_min=0, but they can have different
  ell_max.
* We do not enforce any normalization for the bandpower window functions, so
  window functions with the same shape but different amplitudes will yield
  different expectation values. The purpose of this choice is so that the
  overall signal suppression factor can be encoded as the normalization.
* We assume that all binned output spectra share a common set of ell bins.
  This assumption could be relaxed if someone has a good reason.

"""

import numpy as np
from .util import specind, specgen

class BPWF():
    """
    A BPWF object contains the bandpower window functions corresponding to the
    set of binned output spectra that can be calculated from a set of maps.

    The purpose of this object is to calculate bandpower expectation values
    from unbinned theory spectra.

    Window functions are defined by the output spectrum (cross between two
    maps), the input spectrum (one of the six possible spectra from input
    TQU sky maps), and the ell bin.

    """
    
    def __init__(self, maplist, nbin, strict=False):
        """
        Creates a BPWF structure for the specified list of maps.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define possible output spectra.
        nbin : int
            Number of ell bins, assumed to be the same for all spectra.
        strict : bool, optional
            If True, then the expv function will throw a KeyError if you
            request a window function which has not been defined (via the
            add_windowfn method). If False (default value), then expectation
            values will be zero for any undefined window function.

        """
        
        self.maplist = maplist
        self.nmap = len(maplist)
        self.nspec = self.nmap * (self.nmap + 1) // 2
        self.nbin = nbin
        self.bpwf = {}
        self.strict = strict

    @classmethod
    def tophat(cls, maplist, bin_edges, lmax=None, strict=False):
        """
        Creates BPWF object with tophat window functions defined by ell bin
        edges.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define possible output spectra.
        bin_edges : list of int
            A list of bin edges for the tophat ell bins. The number of ell bins
            will be one less than the number of entries in this list.
        lmax : int, optional
            The maximum ell value over which to define window functions.
            Default behavior is to use the upper edge of the last ell bin.
        strict : bool, optional
            Defines behavior for the new BPWF object. Default is False.

        Returns
        -------
        wf : BPWF object
            New object containing tophat window functions for the specified
            spectra.

        """
        
        # Make tophat window functions
        if lmax is None:
            lmax = bin_edges[-1]
        nbin = len(bin_edges) - 1
        fn = np.zeros(shape=(nbin,lmax+1))
        for i in range(len(bin_edges) - 1):
            i0 = bin_edges[i]
            i1 = bin_edges[i+1]
            fn[i,i0:i1] = 1 / (i1 - i0)
        # Create BPWF object
        wf = cls(maplist, nbin, strict=strict)
        for (i,m0,m1) in specgen(wf.nmap):
            spectype = maplist[m0].field + maplist[m1].field
            if spectype == 'ET': spectype = 'TE'
            if spectype == 'BE': spectype = 'EB'
            if spectype == 'BT': spectype = 'TB'
            wf.add_windowfn(spectype, i, fn)
        # Done
        return wf
        
    def add_windowfn(self, spectype, specout, windowfn, lmin=0):
        """
        Add a set of bandpower window functions to the BPWF object.

        Parameters
        ----------
        spectype : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying the input spectrum for the window functions
        specout : int
            Index of output spectrum by vecp ordering.
        windowfn : array, shape (N,M)
            Two-dimensional array containing window functions for N ell bins,
            where N=self.nbin. 
        lmin : int, optional
            Specify lmin value for the window functions. Default value is 0.

        Returns
        -------
        None

        """

        # Check specin argument.
        if spectype not in ['TT','EE','BB','TE','EB','TB']:
            raise ValueError('invalid specin argument')
        # Check that windowfn argument has the right shape.
        assert windowfn.shape[0] == self.nbin
        # Extend window function down to ell=0, if necessary.
        fn = np.zeros(shape=(self.nbin,lmin+windowfn.shape[1]))
        fn[:,lmin:] = windowfn
        # Add window functions to the BPWF object.
        if specout not in self.bpwf.keys():
            self.bpwf[specout] = {}
        self.bpwf[specout][spectype] = fn

    def adjust_windowfn(self, spectype, specout, scalefac):
        """
        Multiply bandpower window functions by a set of scale factors to
        adjust their normalization.

        Parameters
        ----------
        spectype : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying the input spectrum for the window functions
        specout : int
            Index of output spectrum by vecp ordering.
        scalefac : float or array
            Scale factor by which to multiply bandpower window functions.
            If this is a float, then the same factor will be applied to all
            ell bins. If this is an array, then it should have shape=(N,) where
            N is the number of ell bins.

        Returns
        -------
        None

        """

        # Check whether the specified window functions are defined.
        if not self.valid_windowfn(spectype, specout):
            raise KeyError('requested window function not defined')
        # Apply scale factor(s).
        # Slightly convoluted multiplication, but this works for either
        # scalar or vector scalefac.
        self.bpwf[specout][spectype] = (
            self.bpwf[specout][spectype].transpose() * scalefac).transpose()

    def lmax(self):
        """
        Returns the max ell value used by any window function.

        Parameters
        ----------
        None

        Returns
        -------
        lmax : int
            Maximum ell value found after searching through all window
            functions.

        """

        lmax = 0
        # Loop over output spectra
        for specout in range(self.nspec):
            # Check all six possible input spectra.
            for spectype in ['TT','EE','BB','TE','EB','TB']:
                if self.valid_windowfn(spectype, specout):
                    lmax = max(lmax, self.bpwf[specout][spectype].shape[1] - 1)
        return lmax

    def valid_windowfn(self, spectype, specout):
        """
        Check whether specified window function exists for this BPWF object.

        Parameters
        ----------
        spectype : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying input spectrum type
        specout : int
            Index of output spectrum by vecp ordering

        Returns
        -------
        valid : bool
            True if specified window function exists. False otherwise.

        """

        # Check specin argument.
        if spectype not in ['TT','EE','BB','TE','EB','TB']:
            raise ValueError('invalid specin argument')
        # Check that we have the requested window functions.
        if specout not in self.bpwf.keys():
            return False
        if spectype not in self.bpwf[specout].keys():
            return False
        # If we didn't fail previous two tests, then window fn exists.
        return True
        
    def expv(self, specin, spectype, specout, lmin=0):
        """
        Calculates window-fn weighted sums for input spectrum.

        Parameters
        ----------
        specin : array
            Input spectrum in the form of an array with power specified at each
            ell value (delta-ell=1) and starting from lmin, which is set to 0 by
            default. It is the responsibility of the user to make sure that
            input spectrum and window functions both follow the same Cl vs Dl
            convention.
        spectype : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying the type of the input spectrum
        specout : int
            Index of output spectrum by vecp ordering
        lmin : int, optional
            Starting ell value for the specin argument. Default value is 0.

        Returns
        -------
        expv : array, shape (N,)
            Array of bandpower expectation values for the N ell bins.

        """
        
        # Check that window function is defined.
        if not self.valid_windowfn(spectype, specout):
            if self.strict:
                raise KeyError('requested window function not defined')
            else:
                return np.zeros(self.nbin)        
        # Calculate window-function weighted integrals.
        wf = self.bpwf[specout][spectype]
        lmax = min(wf.shape[1], lmin + specin.shape[0])
        expvals = np.zeros(self.nbin)
        for i in range(self.nbin):
            expvals[i] = np.sum(wf[i,lmin:lmax] * specin[0:lmax-lmin])
        return expvals

    def ell_eff(self, spectype, specout):
        """
        Calculates effective ell values for bandpower window functions.

        Parameters
        ----------
        spectype : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying the input spectrum for window functions.
        specout : int
            Index of output spectrum by vecp ordering

        Returns
        -------
        lval : array, shape (N,)
            Array of effective ell values for the N bins.        

        """

        if self.valid_windowfn(spectype, specout):
            lmax = self.bpwf[specout][spectype].shape[1]
            # We need to explicitly normalize the bpwf integral
            norm = self.expv(np.ones(lmax), spectype, specout)
            expval = self.expv(np.arange(lmax), spectype, specout)
            return (expval / norm)
        else:
            raise ValueError('requested window function not defined')

    def select(self, maplist=None, ellind=None):
        """
        Make a new BPWF object with selected maps and/or ell bins.

        This function can be used to downselect maps or ell bins from a BPWF
        object. Window functions will be copied over from the existing BPWF
        object to the new BPWF object.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new BPWF object. Defaults to None,
            which means that the new BPWF object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new BPWF object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        wf_new : BPWF
            New BPWF object with updated maps and ell bins.

        """

        # Process mapind argument.
        if maplist is None:
            # Not a deep copy, but I don't expect anyone to change the
            # MapDef objects out from under me.
            maplist = self.maplist.copy()
        # Find mapping between new and old map lists.
        # Newly added maps are marked with None.
        mapind = []
        for m in maplist:
            mapind.append(self.maplist.index(m))

        # Create new BPWF object.
        if ellind is not None:
            nbin = len(ellind)
        else:
            nbin = self.nbin
        wf_new = BPWF(maplist, nbin, strict=self.strict)

        # Copy window functions to new object.
        for (i, m0, m1) in specgen(len(maplist)):
            # Find the index of this spectra in old BPWF object.
            i0 = specind(len(self.maplist), mapind[m0], mapind[m1])
            # Copy BPWF -- have to do this manually to avoid duplicate
            # references to window function np arrays.
            wf_new.bpwf[i] = {}
            for spectype in ['TT','EE','BB','TE','EB','TB']:
                if spectype in self.bpwf[i0].keys():
                    wf_new.bpwf[i][spectype] = self.bpwf[i0][spectype].copy()
            # If ellind argument is specified, keep only those ell bins.
            if ellind is not None:
                for key in wf_new.bpwf[i].keys():
                    wf_new.bpwf[i][key] = wf_new.bpwf[i][key][ellind,:]
        # Done
        return wf_new
