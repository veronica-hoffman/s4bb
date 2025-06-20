"""
=================
Utility functions
=================

"""

import numpy as np
from . import bandpass

def specind(nmap, m0, m1):
    """
    Calculates the index of the specified spectrum in vecp ordering.

    Parameters
    ----------
    nmap : int
        Number of maps in the cross-spectral analysis
    m0 : int
        Index of map0 in the map list
    m1 : int
        Index of map1 in the map list

    Returns
    -------
    specind : int
        Index of the map0 x map1 spectrum in the list of spectra

    Notes
    -----
    Return value doesn't depend on the ordering of i0 vs i1.

    """

    # Check arguments
    if (nmap < 0) or (m0 < 0) or (m1 < 0):
        raise ValueError('arguments must be non-negative')
    if (m0 >= nmap) or (m1 >= nmap):
        raise ValueError('arguments m0 and m1 must be less than nmap')
    # Sort the map indices
    a = min(m0, m1)
    b = max(m0, m1)
    # Calculate spectrum index in vecp ordering
    return (sum(range(nmap, nmap - (b - a), -1)) + a)

def mapind(nspec, s0):
    """
    Calculates the map indices for a particular spectrum.

    Parameters
    ----------
    nspec : int
        Number of spectra in the cross-spectral analysis
    s0 : int
        Index of the spectrum

    Returns
    -------
    m0 : int
        Index of map0
    m1 : int
        Index of map1

    """

    # Check arguments
    if (nspec < 0) or (s0 < 0):
        raise ValueError('arguments must be non-negative')
    if (s0 >= nspec):
        raise ValueError('argument i must be less than nspec')
    # Calculate map indices
    nmap = int(-0.5 + np.sqrt(0.25 + 2 * nspec))
    for i in range(nmap):
        if s0 >= (nmap - i):
            s0 = s0 - (nmap - i)
        else:
            m0 = s0
            m1 = s0 + i
            break
    return (m0, m1)

def specgen(nmap):
    """
    Generator function for iterating over spectra in vecp ordering.

    Parameters
    ----------
    nmap : int
        Number of maps to iterate over.

    Yields
    ------
    i : int
        Spectrum index, starts at 0 and increments by 1.
    m0 : int
        Map 1 index
    m1 : int
        Map 2 index

    Example
    -------
    >>> nmap = 3
    >>> for (i, m0, m1) in specgen(nmap):
            print((i, m0, m1))
        (0, 0, 0)
        (1, 1, 1)
        (2, 2, 2)
        (3, 0, 1)
        (4, 1, 2)
        (5, 0, 2)
    
    """

    # Create lists of map ordering
    m0 = []
    m1 = []
    for lag in range(nmap):
        for i in range(nmap - lag):
            m0.append(i)
            m1.append(i + lag)
    # Iterate through spectra
    nspec = nmap * (nmap + 1) // 2
    for i in range(nspec):
        yield (i, m0[i], m1[i])

def vecp_to_matrix(vecp):
    """Converts bandpowers from vecp to matrix ordering"""

    # Determine size of the array
    nspec = vecp.shape[0]
    nmap = int((-1 + np.sqrt(1 + 8 * nspec)) / 2)
    assert (nmap * (nmap + 1) // 2) == nspec
    nbin = vecp.shape[1]
    matrix = np.zeros(shape=(nmap,nmap,nbin))
    # Copy values from vecp to matrix (which is symmetric)
    for (i,m0,m1) in specgen(nmap):
        matrix[m0,m1,:] = vecp[i,:]
        matrix[m1,m0,:] = vecp[i,:]
    return matrix

def matrix_to_vecp(matrix):
    """Converts bandpowers from matrix to vecp ordering"""

    # Determine the size of the array
    nmap = matrix.shape[0]
    assert matrix.shape[1] == nmap
    nspec = nmap * (nmap + 1) // 2
    nbin = matrix.shape[2]
    vecp = np.zeros(shape=(nspec,nbin))
    # Copy values from matrix to vecp
    for (i,m0,m1) in specgen(nmap):
        vecp[i,:] = matrix[m0,m1,:]
    return vecp
        
class MapDef():
    """
    The MapDef object describes the properties of a map (or maps).
    It is also used to describe the map inputs to auto and cross spectra.

    """

    def __init__(self, name, field, bandpass=None, Bl=None, fwhm_arcmin=None,
                 lensing_template=False, simtype=None):
        """
        Create a new MapDef object.

        Parameters
        ----------
        name : string
            Name of the map.
        field : string
            Specifies which field the map represents. Valid options are 'T',
            'E', 'B', 'QU', or 'TQU'.
        bandpass : Bandpass object, optional
            Object describing the bandpass of the map.
        Bl : array, dtype=float, optional
            Array specifying beam window function for this map. It is assumed
            that this function starts at ell = 0 and extends up to
            ell_max = len(Bl) - 1. If Bl is defined, then it supersedes the
            `fwhm_arcmin` argument.
        fwhm_arcmin : float, optional
            Full-width at half maximum, in arc-minutes, to define a Gaussian
            beam for this map. If the `Bl` argument is specified, then it
            supersedes this value.
        lensing_template : bool, optional
            If True, indicates that this map is a lensing template so signal
            expectation values should contain lensing B modes only. Default is
            False.
        simtype : string, optional
            Use this parameter to mark maps as 'signal' or 'noise' sims. This
            property is used by the BpCov_signoi.from_xspec() class method.
            Default is None, i.e. unspecified simtype.

        """
        
        self.name = name
        assert field.upper() in ['T','E','B','QU','TQU']
        self.field = field.upper()
        self.bandpass = bandpass
        self.Bl = Bl
        self.fwhm_arcmin = fwhm_arcmin
        self.lensing_template = lensing_template
        self.simtype = simtype

    def __str__(self):
        """Short description of the map"""
        
        # Map name and field
        mapstr = '{}_{}'.format(self.name, self.field)
        # Additional notes:
        #  - Is this a lensing template?
        #  - Is the sim type specified?
        extras = []
        if self.lensing_template:
            extras.append('lensing_template')
        if self.simtype is not None:
            extras.append(self.simtype)
        if len(extras) > 0:
            mapstr += ' [{}]'.format(','.join(extras))
        return mapstr

    def __eq__(self, value):
        """
        MapDef objects are considered equivalent if the name and field match.
        Also check that lensing template and simtype properties match.
        Don't check whether the bandpasses or beams match.
        
        """

        # For simtype, note that None == None evaluates as True.
        return ((self.name == value.name) and (self.field == value.field) and
                (self.lensing_template == value.lensing_template) and
                (self.simtype == value.simtype))

    def copy(self, update_field=None):
        """
        Returns a copy of the MapDef object.

        Parameters
        ----------
        update_field : str, optional
            By default, the returned MapDef object will have the same field
            type as the current object. Use this optional argument to update to
            a new field type.

        Returns
        -------
        map_new : MapDef object
            Copy of the current object (not a deep copy!).

        """

        if update_field is not None:
            return MapDef(self.name, update_field, bandpass=self.bandpass,
                          Bl=self.Bl, fwhm_arcmin=self.fwhm_arcmin,
                          lensing_template=self.lensing_template,
                          simtype=self.simtype)
        else:
            return MapDef(self.name, self.field, bandpass=self.bandpass,
                          Bl=self.Bl, fwhm_arcmin=self.fwhm_arcmin,
                          lensing_template=self.lensing_template,
                          simtype=self.simtype)

    def to_dict(self):
        """
        Returns a dict containing the information about this map.

        Parameters
        ----------
        None

        Returns
        -------
        map_dict : dict
            Dictionary object contain all of the information about the map.
            This dict can be more easily serialized to a pickle file, etc.
            Can convert back to a MapDef object using the from_dict class
            method.

        """

        map_dict = {}
        map_dict['name'] = self.name
        map_dict['field'] = self.field
        map_dict['bandpass'] = np.stack((self.bandpass.nu, self.bandpass.wgt))
        map_dict['Bl'] = self.Bl
        map_dict['fwhm_arcmin'] = self.fwhm_arcmin
        map_dict['lensing_template'] = self.lensing_template
        map_dict['simtype'] = self.simtype
        return map_dict

    @classmethod
    def from_dict(cls, map_dict):
        """
        Creates MapDef object from a dict.
        
        Parameters
        ----------
        map_dict : dict
            Dictionary object containing information about the map. See
            the MapDef.to_dict method for details about keys/values.

        Returns
        -------
        new_map : MapDef
            MapDef object containing map information.

        """

        # The name and field keys are required; other keys are optional.
        try:
            bp = bandpass.Bandpass(map_dict['bandpass'][0,:],
                                   map_dict['bandpass'][1,:])
        except KeyError:
            bp = None
        try:
            Bl = map_dict['Bl']
        except KeyError:
            Bl = None
        try:
            fwhm_arcmin = map_dict['fwhm_arcmin']
        except KeyError:
            fwhm_arcmin = None
        try:
            lensing_template = map_dict['lensing_template']
        except KeyError:
            lensing_template = False
        try:
            simtype = map_dict['simtype']
        except KeyError:
            simtype = None
        return cls(map_dict['name'], map_dict['field'], bandpass=bp, Bl=Bl,
                   fwhm_arcmin=fwhm_arcmin, lensing_template=lensing_template,
                   simtype=simtype)
    
    def beam(self, ell_max, Bl_min=0.0):
        """
        Returns the beam window function (Bl) for this map.

        Parameters
        ----------
        ell_max : int
            Maximum ell value for the beam window function.
        Bl_min : float, optional
            Minimum value for the beam window function. In the case of large
            beam sizes, Bl can reach extremely small values at high ell, which
            leads to problems when dividing by Bl to correct bandpowers. This
            argument allows you to specify a floor to the beam window function.
            Default value is 0, i.e. no floor.

        Returns
        -------
        Bl : array, shape=(ell_max+1,), dtype=float
            Beam window function defined starting at ell=0 and extending up to
            ell=ell_max. If a beam is not defined for this map, the window
            function returned will be all ones.
        
        """

        # If Bl is defined, use that
        if self.Bl is not None:
            if len(self.Bl) > ell_max:
                Bl = self.Bl[0:ell_max+1].copy()
            else:
                Bl = np.zeros(ell_max + 1)
                Bl[0:len(self.Bl)] = self.Bl.copy()
        # otherwise, calculate Gaussian Bl from FWHM
        elif self.fwhm_arcmin is not None:
            ell = np.arange(ell_max + 1)
            sigma_rad = np.radians(self.fwhm_arcmin / 60) / np.sqrt(8 * np.log(2))
            Bl = np.exp(-0.5 * ell**2 * sigma_rad**2)
        # no beam defined
        else:
            Bl = np.ones(ell_max + 1)
        # Apply floor to Bl
        Bl[Bl < Bl_min] = Bl_min
        # Done
        return Bl

    def to_hdf5(self, fh, group):
        """
        Record map definition to HDF5 file

        Parameters
        ----------
        fh : h5py File object
            h5py File object should be opened in write mode.
        group : string
            HDF5 group specifier where map information will be recorded.

        Returns
        -------
        None

        """

        # Store map name and field as string attributes.
        fh.create_group(group)
        fh[group].attrs['name'] = self.name
        fh[group].attrs['field'] = self.field
        # Store bandpass, if it is defined
        if self.bandpass is not None:
            self.bandpass.to_hdf5(fh, group + '/bandpass')
        # Store Bl, if it is defined
        if self.Bl is not None:
            fh[group + 'Bl'] = self.Bl
        # Set fwhm_arcmin as attribute, if it is defined
        if self.fwhm_arcmin is not None:
            fh[group].attrs['fwhm_arcmin'] = self.fwhm_arcmin
        # Store lensing_template as boolean attribute.
        fh[group].attrs['lensing_template'] = self.lensing_template
        # Store simtype as string attribute, if it is defined
        if self.simtype is not None:
            fh[group].attrs['simtype'] = self.simtype

    @classmethod
    def from_hdf5(cls, fh, group):
        """
        Read map definition from HDF5 file

        Parameters
        ----------
        fh : h5py File object
            h5py File object should be opened in read mode.
        group : string
            HDF5 group specifier where map information is stored.

        Returns
        -------
        mdef : MapDef object
            Object containing map information read from HDF5 file.

        """
        
        # Get map name and field
        name = fh[group].attrs['name']
        field = fh[group].attrs['field']
        # Get bandpass, if it is defined.
        if 'bandpass' in fh[group].keys():
            bpass = bandpass.Bandpass.from_hdf5(fh, group + '/bandpass')
        else:
            bpass = None
        # Get Bl, if it is defined
        if 'Bl' in fh[group].keys():
            Bl = np.array(fh[group + 'Bl'])
        else:
            Bl = None
        # Get fwhm_arcmin, if it is defined
        if 'fwhm_arcmin' in fh[group].attrs.keys():
            fwhm_arcmin = fh[group].attrs['fwhm_arcmin']
        else:
            fwhm_arcmin = None
        # Get lensing template flag
        lensing_template = fh[group].attrs['lensing_template']
        # Get simtype, if it is defined
        if 'simtype' in fh[group].attrs.keys():
            simtype = fh[group].attrs['simtype']
        else:
            simtype = None
        # Return MapDef object
        return cls(name, field, bandpass=bpass, Bl=Bl,
                   fwhm_arcmin=fwhm_arcmin, lensing_template=lensing_template,
                   simtype=simtype)

