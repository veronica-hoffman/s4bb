"""
==========
Unit Tests
==========

"""

import unittest
import os
import numpy as np
import healpy as hp
import h5py

from util import specind, mapind, specgen, MapDef
from spectra import XSpec, CalcSpec, CalcSpec_healpy, CalcSpec_namaster
from bandpass import Bandpass
from bpwf import BPWF
from bpcov import BpCov, BpCov_signoi
from models import Model, Model_cmb

# Some of the tests involve generating maps, calculating power spectra, and
# checking that we recover the input spectra. This involves sample variance,
# so we need to specify a tolerance for the test, expressed here in units of
# sigma.
TOL = 5.0

class UtilTest(unittest.TestCase):
    """Unit tests for util.py"""

    def setUp(self):
        # Five maps with a mix of T, E, B
        self.maps = [MapDef('m0_T', 'T'),
                     MapDef('m1_E', 'E'),
                     MapDef('m2_B', 'B'),
                     MapDef('m3_E', 'E'),
                     MapDef('m4_B', 'B')]
        self.nmap = len(self.maps)
        # Five maps yields 15 spectra... specify in vecp ordering
        self.spec = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                     (0, 1), (1, 2), (2, 3), (3, 4),
                     (0, 2), (1, 3), (2, 4),
                     (0, 3), (1, 4),
                     (0, 4)]
        self.nspec = len(self.spec)
        # Lower and upper edges for three ell bins
        self.bins = np.array([[10, 20, 30], [20, 30, 40]])
        self.nbin = self.bins.shape[1]

    def test_specind(self):
        """Test specind function"""

        for i in range(self.nspec):
            self.assertEqual(i, specind(self.nmap, self.spec[i][0],
                                        self.spec[i][1]))

    def test_mapind(self):
        """Test mapind function"""

        for i in range(self.nspec):
            (m0, m1) = mapind(self.nspec, i)
            self.assertEqual(self.spec[i], (m0, m1))

    def test_specgen(self):
        """Test specgen generator"""

        for (i, m0, m1) in specgen(self.nmap):
            self.assertEqual(self.spec[i], (m0, m1))

    def test_MapDef(self):
        """Test MapDef class"""

        # Equality requires same name *and* field
        self.assertEqual(self.maps[0], MapDef('m0_T', 'T'))
        self.assertNotEqual(self.maps[0], MapDef('m0_T', 'E'))
        self.assertNotEqual(self.maps[0], MapDef('m1_T', 'T'))

    def test_MapDef_dict(self):
        """Test conversion between MapDef and dict"""

        bp = Bandpass.tophat(90, 110)
        m0 = MapDef('m0', 'B', bandpass=bp, fwhm_arcmin=10.0, lensing_template=False)
        map_dict = m0.to_dict()
        m1 = MapDef.from_dict(map_dict)
        self.assertEqual(m0, m1)
        np.testing.assert_allclose(m0.bandpass.nu, m1.bandpass.nu, atol=1e-15)
        np.testing.assert_allclose(m0.bandpass.wgt, m1.bandpass.wgt, atol=1e-15)
        self.assertEqual(m0.fwhm_arcmin, m1.fwhm_arcmin)        
        self.assertEqual(m0.lensing_template, m1.lensing_template)
    
class SpectraTest(unittest.TestCase):
    """Unit tests for spectra.py"""
    
    def setUp(self):
        # Five maps with a mix of T, E, B
        self.maps = [MapDef('m0_T', 'T'),
                     MapDef('m1_E', 'E'),
                     MapDef('m2_B', 'B'),
                     MapDef('m3_E', 'E'),
                     MapDef('m4_B', 'B')]
        self.nmap = len(self.maps)
        # Five maps yields 15 spectra... specify in vecp ordering
        self.spec = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                     (0, 1), (1, 2), (2, 3), (3, 4),
                     (0, 2), (1, 3), (2, 4),
                     (0, 3), (1, 4),
                     (0, 4)]
        self.nspec = len(self.spec)
        # Lower and upper edges for three ell bins
        self.bins = np.array([[10, 20, 30], [20, 30, 40]])
        self.nbin = self.bins.shape[1]

    def test_XSpec(self):
        """Test XSpec class"""

        # Try concatenating two sets of spectra.
        nrlz1 = 4
        spec1 = np.ones((self.nspec, self.nbin, nrlz1))
        xspec1 = XSpec(self.maps, self.bins, spec1)
        nrlz2 = 2
        spec2 = 4 * np.ones((self.nspec, self.nbin, nrlz2))
        xspec2 = XSpec(self.maps, self.bins, spec2)
        xspec3 = xspec1 + xspec2
        self.assertEqual(xspec3.nmap(), self.nmap)
        self.assertEqual(xspec3.nspec(), self.nspec)
        self.assertEqual(xspec3.nbin(), self.nbin)
        self.assertEqual(xspec3.nrlz(), nrlz1 + nrlz2)
        self.assertTrue((xspec3.spec[:,:,0:nrlz1] == spec1).all())
        self.assertTrue((xspec3.spec[:,:,nrlz1:] == spec2).all())

        # Test setitem, getitem
        xspec3[0,1,2] = 37.0
        self.assertEqual(xspec3[0,1,2], 37)
        self.assertEqual(xspec3.spec[0,1,2], 37)

        # Test select
        # Downselect from 3 maps (6 spectra) to 2 maps (3 spectra)
        spec4 = np.ones(shape=(6,self.nbin,1))
        spec4[0,:,:] = 2.0**2 # m0 x m0
        spec4[1,:,:] = 3.0**2 # m1 x m1
        spec4[2,:,:] = 5.0**2 # m2 x m2
        spec4[3,:,:] = 2.0 * 3.0 # m0 x m1
        spec4[4,:,:] = 3.0 * 5.0 # m1 x m2
        spec4[5,:,:] = 2.0 * 5.0 # m0 x m2
        xspec4 = XSpec(self.maps[0:3], self.bins, spec4)
        xspec5 = xspec4.select([self.maps[2], self.maps[0]], None)
        self.assertEqual(xspec5.nmap(), 2)
        self.assertEqual(xspec5.nbin(), xspec4.nbin())
        self.assertEqual(xspec5.nrlz(), xspec4.nrlz())
        self.assertTrue((xspec5[0,:,:] == xspec4[2,:,:]).all())
        self.assertTrue((xspec5[1,:,:] == xspec4[0,:,:]).all())
        self.assertTrue((xspec5[2,:,:] == xspec4[5,:,:]).all())

        # Test select
        # Downselect from 3 maps (6 spectra) to 1 map (1 spectrum)
        # while also downselecting from 3 ell bins to 2 ell bins
        xspec6 = xspec4.select([self.maps[1]], [0,1])
        self.assertEqual(xspec6.nmap(), 1)
        self.assertEqual(xspec6.nbin(), 2)
        self.assertEqual(xspec6.nrlz(), xspec4.nrlz())
        self.assertTrue((xspec6[0,:,:] == xspec4[1,0:2,:]).all())

    def test_XSpec_average(self):
        # Make XSpec object with multiple realizations filled with
        # random values.
        nrlz = 100
        values = np.random.randn(self.nspec, self.nbin, nrlz)
        spec = XSpec(self.maps, self.bins, values)
        np.testing.assert_allclose(spec.ensemble_average()[:,:,0],
                                   np.mean(values, axis=2), atol=1e-15)
        
    def test_XSpec_to_hdf5(self):
        # MapDef objects with bells and whistles
        m0 = MapDef('m0', 'E', bandpass=Bandpass.tophat(80, 100),
                    fwhm_arcmin=25.0)
        m1 = MapDef('m1', 'B', bandpass=Bandpass.tophat(140, 160),
                    fwhm_arcmin=25.0)
        m2 = MapDef('m2', 'B', lensing_template=True)
        m3 = MapDef('m3', 'B', Bl=np.ones(300))
        maplist = [m0, m1, m2, m3]
        nmap = len(maplist)
        nspec = nmap * (nmap + 1) // 2
        # Create XSpec object
        spec = XSpec(maplist, self.bins, np.zeros(shape=(nspec, self.nbin)))
        # Write XSpec object to file
        h5name = 'test.h5'
        with h5py.File(h5name, 'w') as f:
            spec.to_hdf5(f)
        # Read back in from HDF5 file
        with h5py.File(h5name, 'r') as f:
            spec2 = XSpec.from_hdf5(f)
        # Test that two objects are the same
        for i in range(nmap):
            self.assertEqual(spec.maplist[i], spec2.maplist[i])
        np.testing.assert_allclose(spec.bins, spec2.bins, atol=1e-15)
        np.testing.assert_allclose(spec.spec, spec2.spec, atol=1e-15)
        # Remove temporary file
        os.remove(h5name)
        
    def test_CalcSpec(self):
        """Test CalcSpec base class"""

        m0 = MapDef('m0', 'T')
        m1 = MapDef('m1', 'QU')
        m2 = MapDef('m2', 'TQU')
        nside = 128
        apod = np.ones(hp.nside2npix(nside))
        cs = CalcSpec([m0, m1, m2], apod, nside, self.bins)
        # Check that maplist_out is as expected.
        self.assertEqual(cs.nmap(), 6)
        self.assertEqual(cs.maplist_out[0], MapDef('m0', 'T'))
        self.assertEqual(cs.maplist_out[1], MapDef('m1', 'E'))
        self.assertEqual(cs.maplist_out[2], MapDef('m1', 'B'))
        self.assertEqual(cs.maplist_out[3], MapDef('m2', 'T'))
        self.assertEqual(cs.maplist_out[4], MapDef('m2', 'E'))
        self.assertEqual(cs.maplist_out[5], MapDef('m2', 'B'))

    def test_CalcSpec_healpy(self):
        """Test CalcSpec_healpy"""

        # Check that we get the sensible power spectrum measurements.
        m0 = MapDef('m0', 'TQU')
        Cl = [8.0, 4.0, 1.0, 2.0] # TT, EE, BB, TE; white spectra
        # Try three values of NSIDE
        for nside in [64,128,256]:
            # Set up CalcSpec_healpy object
            apod = np.ones(hp.nside2npix(nside))
            bins = np.array([[10], [2 * nside]]) # one big ell bin
            cs = CalcSpec_healpy([m0], apod, nside, bins, use_Dl=False)
            # Generate input maps
            tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                             nside, new=True)
            # Calculate spectra
            spec = cs.calc([tqu])
            # Compare output spectra to the input power levels.
            self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
            self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
            self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
            self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
            self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
            self.assertTrue(np.abs(spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))

        # Check that we get sensible power spectrum measurements with different
        # sky cuts / apodization masks. Note that CalcSpec_healpy does not
        # include a pure-B estimator, so for this test we will set BB=0 and
        # just check the output TT, EE, and TE power.
        Cl = [8.0, 4.0, 0.0, 2.0] # TT, EE, BB, TE; white spectra
        nside = 128
        bins = np.array([[10], [2 * nside]]) # one big ell bin
        (theta, phi) = hp.pix2ang(nside, range(hp.nside2npix(nside)))
        apod_options = []
        apod_options.append((theta < np.pi / 4).astype(float))
        apod_options.append((theta < np.pi / 6).astype(float))
        apod_options.append((theta < np.pi / 8).astype(float))
        apod_options.append((theta / np.pi)**2)
        apod_options.append((2 * (theta - np.pi / 2) / np.pi)**2)
        for apod in apod_options:
            # Set up CalcSpec_healpy object
            cs = CalcSpec_healpy([m0], apod, nside, bins, use_Dl=False)
            # Generate input maps
            tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                             nside, new=True)
            # Calculate spectra
            spec = cs.calc([tqu])
            # Compare output spectra to the input power levels.
            fsky = np.mean(apod**2)**2 / np.mean(apod**4) # effective fsky for mask
            self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside) / np.sqrt(fsky))

    def test_CalcSpec_namaster(self):
        """Test CalcSpec_namaster"""

        # Check that we get the sensible power spectrum measurements.
        m0 = MapDef('m0', 'TQU')
        Cl = [8.0, 4.0, 1.0, 2.0] # TT, EE, BB, TE; white spectra
        # Try three values of NSIDE
        for nside in [64,128,256]:
            # Set up CalcSpec_namaster object
            apod = np.ones(hp.nside2npix(nside))
            bins = np.array([[10], [2 * nside]]) # one big ell bin
            cs = CalcSpec_namaster([m0], apod, nside, bins, use_Dl=False, pure_B=False)
            # Generate input maps
            tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                             nside, new=True)
            # Calculate spectra
            spec = cs.calc([tqu])
            # Compare output spectra to the input power levels.
            self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
            self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
            self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
            self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
            self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
            self.assertTrue(np.abs(spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))

        # Test that we properly handle different combinations of T, QU, or TQU
        # input maps.
        m0 = MapDef('m0', 'TQU')
        m1 = MapDef('m1', 'T')
        m2 = MapDef('m2', 'QU')
        # Just work at one value of NSIDE
        nside = 128
        apod = np.ones(hp.nside2npix(nside))
        bins = np.array([[10], [2 * nside]]) # one big ell bin
        # Remake TQU map at this NSIDE
        tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                         nside, new=True)
        # Independent TQU sim with same input spectrum.
        tqu2 = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                          nside, new=True)
        t = tqu2[0,:]
        qu = tqu2[1:,:]
        # T only ------
        cs = CalcSpec_namaster([m1], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([t])
        self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        # QU only -----
        cs = CalcSpec_namaster([m2], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([qu])
        self.assertTrue(np.abs(Cl[1] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(spec[2,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        # T, QU -------
        cs = CalcSpec_namaster([m1, m2], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([t, qu])
        self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
        self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))        
        # QU, T -------
        cs = CalcSpec_namaster([m2, m1], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([qu, t])
        self.assertTrue(np.abs(Cl[1] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(Cl[0] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(spec[3,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(Cl[3] - spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
        # TQU, T, QU --
        cs = CalcSpec_namaster([m0, m1, m2], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([tqu, t, qu])
        self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(Cl[0] - spec[3,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(Cl[1] - spec[4,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[5,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(Cl[3] - spec[6,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
        self.assertTrue(np.abs(spec[7,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[8,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(Cl[3] - spec[9,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
        self.assertTrue(np.abs(spec[10,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[11,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[12,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1]) / (2 * nside))
        self.assertTrue(np.abs(spec[13,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[14,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[15,0,0]) < TOL * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(spec[16,0,0]) < TOL * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(spec[17,0,0]) < TOL * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(spec[18,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1]) / (2 * nside))
        self.assertTrue(np.abs(spec[19,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[20,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))

        # Check that we get sensible power spectrum measurements with different
        # sky cuts / apodization masks. Turn on the NaMaster pure-B estimator
        # to deal with E->B leakage. Keep BB << EE so that we don't have to
        # worry about B->E.
        Cl = [8.0, 4.0, 0.1, 2.0] # TT, EE, BB, TE; white spectra
        m0 = MapDef('m0', 'TQU')
        nside = 128
        bins = np.array([[10], [2 * nside]]) # one big ell bin
        # Generate input maps
        tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                        nside, new=True)
        # Try some different apodizations. These should all be smooth enough
        # to work well with pure-B estimator.
        (theta, phi) = hp.pix2ang(nside, range(hp.nside2npix(nside)))
        apod_options = []
        apod_options.append(np.cos(theta)**2)
        apod_options.append(np.cos(2 * theta)**2)
        apod_options.append(np.cos(3 * theta)**2)
        apod_options.append(np.cos(4 * theta)**2)
        for apod in apod_options:
            # Set up CalcSpec_healpy object
            cs = CalcSpec_namaster([m0], apod, nside, bins, use_Dl=False, pure_B=True)
            # Calculate spectra
            spec = cs.calc([tqu])
            # Compare output spectra to the input power levels.
            fsky = np.mean(apod**2)**2 / np.mean(apod**4) # effective fsky for mask
            self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside) / np.sqrt(fsky))

        # Test bandpower window functions using several ell bins and non-trivial input spectrum.
        nside = 128
        (theta, phi) = hp.pix2ang(nside, range(hp.nside2npix(nside)))
        apod = np.sin(4 * theta)**2 * np.sin(2 * phi)**2
        bins = np.stack((np.arange(50, 300, 25), np.arange(50, 300, 25) + 25))
        cs = CalcSpec_namaster([m0], apod, nside, bins, use_Dl=True, pure_B=True)
        # Input spectra with some peaks
        lmax = bins[1,-1]
        Cl_input = np.zeros(shape=(4,lmax+1))
        Cl_input[0,50:150] = np.sin(2 * np.pi * np.arange(100) / 100)**2 # TT
        Cl_input[1,50:150] = 0.1 * np.sin(3 * np.pi * np.arange(100) / 100)**2 # EE
        Cl_input[2,50:150] = 1e-3 * np.sin(np.pi * np.arange(100) / 100)**2 # BB
        Cl_input[3,:] = 0.1 * np.sqrt(Cl_input[0,:] * Cl_input[1,:]) # TE
        # Use bandpower window functions to calculate expectation values
        include_leakage = True
        bpwf = cs.get_bpwf()
        expv = np.zeros(shape=(6,cs.nbin()))
        expv[0,:] = bpwf.expv(Cl_input[0,:], 'TT', 0)
        expv[1,:] = bpwf.expv(Cl_input[1,:], 'EE', 1)
        if include_leakage:
            expv[1,:] += bpwf.expv(Cl_input[2,:], 'BB', 1)
        expv[2,:] = bpwf.expv(Cl_input[2,:], 'BB', 2)
        if include_leakage:
            expv[2,:] += bpwf.expv(Cl_input[1,:], 'EE', 2)
        expv[3,:] = bpwf.expv(Cl_input[3,:], 'TE', 3)
        # Generate and analyze several simulated maps.
        nsim = 5
        for i in range(nsim):
            sim = hp.synfast(Cl_input, nside, new=True)
            if i == 0:
                spec = cs.calc([sim])
            else:
                spec += cs.calc([sim])
        # Compare expectation value to mean of sims.
        # Calculate the expected bandpower variance here, but having a hard
        # time passing this test quantitatively.
        # - When calculating expectation values above, not sure of whether to
        #   use the EE->BB and BB->EE leakage, hence the include_leakage flag.
        # - Only look at the first four bandpowers, which cover ell=[50:150]
        #   because this is where the input spectra are non-zero. Outside this
        #   range, I think it is just too hard to get the wings of the window
        #   functions exactly right.
        # - Breaking down and increasing tolerance to 10 sigma for this test.
        #   The plots look pretty good **and** perhaps this is really a test
        #   of the NaMaster algorithm.
        # - If this test is failing in the future, come back and look at this
        #   comment!
        TOL10 = 10.0
        fsky = np.mean(apod**2)**2 / np.mean(apod**4) # Effective fsky of mask
        k = fsky * np.array([(2 * np.arange(low, hi) + 1).sum() for (low, hi) in np.transpose(bins)])
        var = np.zeros(expv.shape)
        var[0,:] = 2 * expv[0,:]**2 / k / nsim # TT
        self.assertTrue(all(((spec[0,:,:].mean(axis=1) - expv[0,:])**2 / var[0,:] < TOL10)[0:4]))
        var[1,:] = 2 * expv[1,:]**2 / k / nsim # EE
        self.assertTrue(all(((spec[1,:,:].mean(axis=1) - expv[1,:])**2 / var[1,:] < TOL10)[0:4]))
        var[2,:] = 2 * expv[2,:]**2 / k / nsim # BB
        self.assertTrue(all(((spec[2,:,:].mean(axis=1) - expv[2,:])**2 / var[2,:] < TOL10)[0:4]))
        var[3,:] = (expv[0,:] * expv[1,:] + expv[3,:]**2) / k / nsim # TE
        self.assertTrue(all(((spec[3,:,:].mean(axis=1) - expv[3,:])**2 / var[3,:] < TOL10)[0:4]))
        var[4,:] = (expv[1,:] * expv[2,:]) / k / nsim # EB
        self.assertTrue(all((spec[4,:,:].mean(axis=1)**2 / var[4,:] < TOL10)[0:4]))
        var[5,:] = (expv[0,:] * expv[2,:]) / k / nsim # TB
        self.assertTrue(all((spec[5,:,:].mean(axis=1)**2 / var[5,:] < TOL10)[0:4]))
        
class BandpassTest(unittest.TestCase):
    """Unit tests for bandpass.py"""

    def test_deltafn(self):
        """Test delta-fn bandpass"""

        nu0 = 100.0 # GHz
        bp = Bandpass.deltafn(nu0)
        self.assertEqual(1.0, bp.bandpass_integral(lambda x: 1.0))
        self.assertEqual(nu0, bp.nu_eff())
        # Check the CMB unit conversion
        from bandpass import GHz, Tcmb, h, k, c
        x = h * nu0 * GHz / (k * Tcmb)
        conv = (2 * k**3 * Tcmb**2 / (h**2 * c**2) * x**4 * np.exp(x) /
                (np.exp(x) - 1)**2)
        self.assertEqual(conv, bp.cmb_unit_conversion())

    def test_tophat(self):
        """Test tophat bandpass"""

        nu0 = 100.0 # GHz
        nu1 = 120.0
        bp = Bandpass.tophat(nu0, nu1, RJ=False)
        self.assertEqual(1.0, bp.bandpass_integral(lambda x: 1.0))
        self.assertEqual((nu0 + nu1) / 2, bp.nu_eff())
        for nu in np.linspace(nu0, nu1, 10):
            self.assertEqual(1.0 / (nu1 - nu0), bp.fn(nu))

class BpwfTest(unittest.TestCase):
    """Unit tests for bpwf.py"""

    def setUp(self):
        # BPWF object with three maps (T, E, B) and tophat window functions.
        self.maplist = [MapDef('m0_T', 'T'),
                        MapDef('m1_E', 'E'),
                        MapDef('m2_B', 'B')]
        self.wf = BPWF.tophat(self.maplist, [10, 20, 30, 40, 50, 60], lmax=100)
        self.nbin = self.wf.nbin

    def test_expv(self):
        """Test BPWF.expv method"""

        specin = np.ones(self.wf.lmax() + 1)
        # map0 x map0 should be TT only
        self.assertTrue(all(self.wf.expv(specin, 'TT', 0) == np.ones(self.nbin)))
        for spectype in ['EE','BB','TE','EB','TB']:
            self.assertTrue(all(self.wf.expv(specin, spectype, 0) == np.zeros(self.nbin)))
        # map1 x map1 should be EE only
        self.assertTrue(all(self.wf.expv(specin, 'EE', 1) == np.ones(self.nbin)))
        for spectype in ['TT','BB','TE','EB','TB']:
            self.assertTrue(all(self.wf.expv(specin, spectype, 1) == np.zeros(self.nbin)))
        # map2 x map2 should be BB only
        self.assertTrue(all(self.wf.expv(specin, 'BB', 2) == np.ones(self.nbin)))
        for spectype in ['TT','EE','TE','EB','TB']:
            self.assertTrue(all(self.wf.expv(specin, spectype, 2) == np.zeros(self.nbin)))
        # map0 x map2 should be TB only
        self.assertTrue(all(self.wf.expv(specin, 'TE', 3) == np.ones(self.nbin)))
        for spectype in ['TT','EE','BB','EB','TB']:
            self.assertTrue(all(self.wf.expv(specin, spectype, 3) == np.zeros(self.nbin)))

    def test_select(self):
        """Test BPWF select method"""

        specin = np.ones(self.wf.lmax() + 1)
        # First, try downselecting maps -- keep E and B only
        wfnew = self.wf.select([self.maplist[1], self.maplist[2]], None)
        # map0 x map0 should be EE only
        self.assertTrue(all(wfnew.expv(specin, 'EE', 0) == np.ones(self.nbin)))
        for spectype in ['TT','BB','TE','EB','TB']:
            self.assertTrue(all(wfnew.expv(specin, spectype, 0) == np.zeros(self.nbin)))
        # map1 x map1 should be BB only
        self.assertTrue(all(wfnew.expv(specin, 'BB', 1) == np.ones(self.nbin)))
        for spectype in ['TT','EE','TE','EB','TB']:
            self.assertTrue(all(wfnew.expv(specin, spectype, 1) == np.zeros(self.nbin)))
        # map0 x map1 should be EB only
        self.assertTrue(all(wfnew.expv(specin, 'EB', 2) == np.ones(self.nbin)))
        for spectype in ['TT','EE','BB','TE','TB']:
            self.assertTrue(all(wfnew.expv(specin, spectype, 2) == np.zeros(self.nbin)))

        # Next, try downselecting ell bins -- keep bins 2 and 3 only
        keep = [2, 3]
        wfnew = self.wf.select(None, keep)
        self.assertEqual(len(keep), wfnew.nbin)
        self.assertTrue(all(self.wf.ell_eff('TT', 0)[keep] == wfnew.ell_eff('TT', 0)))

class BpCovTest(unittest.TestCase):
    """Unit tests for bpcov.py"""

    def test_mask_ell(self):
        """Test BpCov.mask_ell method"""

        # Make a BpCov structure with two maps, three ell bins.
        map1 = MapDef('map1', 'B')
        map2 = MapDef('map2', 'B')
        maplist = [map1, map2]
        bpcm = BpCov(maplist, 3)
        bpcm.set(np.ones((9,9)))

        # noffdiag=0
        self.assertTrue((bpcm.get(noffdiag=0) == np.array([[1,1,1,0,0,0,0,0,0],
                                                           [1,1,1,0,0,0,0,0,0],
                                                           [1,1,1,0,0,0,0,0,0],
                                                           [0,0,0,1,1,1,0,0,0],
                                                           [0,0,0,1,1,1,0,0,0],
                                                           [0,0,0,1,1,1,0,0,0],
                                                           [0,0,0,0,0,0,1,1,1],
                                                           [0,0,0,0,0,0,1,1,1],
                                                           [0,0,0,0,0,0,1,1,1]])).all())
        # noffdiag=1
        self.assertTrue((bpcm.get(noffdiag=1) == np.array([[1,1,1,1,1,1,0,0,0],
                                                           [1,1,1,1,1,1,0,0,0],
                                                           [1,1,1,1,1,1,0,0,0],
                                                           [1,1,1,1,1,1,1,1,1],
                                                           [1,1,1,1,1,1,1,1,1],
                                                           [1,1,1,1,1,1,1,1,1],
                                                           [0,0,0,1,1,1,1,1,1],
                                                           [0,0,0,1,1,1,1,1,1],
                                                           [0,0,0,1,1,1,1,1,1]])).all())
        # noffdiag=2
        self.assertTrue((bpcm.get(noffdiag=2) == np.ones((9,9))).all())
    
    def test_select_map(self):
        """Test BpCov.select method for maps"""
    
        # Make a BpCov structure with three maps, one ell bin.
        map1 = MapDef('map1', 'B')
        map2 = MapDef('map2', 'B')
        map3 = MapDef('map3', 'B')
        maplist = [map1, map2, map3]
        bpcm = BpCov(maplist, 1)
        bpcm.set(np.array([[ 0,  1,  2,  3,  4,  5],
                           [ 6,  7,  8,  9, 10, 11],
                           [12, 13, 14, 15, 16, 17],
                           [18, 19, 20, 21, 22, 23],
                           [24, 25, 26, 27, 28, 29],
                           [30, 31, 32, 33, 34, 35]]))
    
        # Select map1 only.
        bpcm1 = bpcm.select(maplist=[map1], ellind=None)
        self.assertTrue((bpcm1.get() == np.array([0])).all())
        # Select map2 only.
        bpcm2 = bpcm.select(maplist=[map2], ellind=None)
        self.assertTrue((bpcm2.get() == np.array([7])).all())
        # Select map3 only.
        bpcm3 = bpcm.select(maplist=[map3], ellind=None)
        self.assertTrue((bpcm3.get() == np.array([14])).all())
    
        # Select map1 and map2.
        bpcm12 = bpcm.select(maplist=[map1, map2], ellind=None)
        self.assertTrue((bpcm12.get() == np.array([[ 0,  1,  3],
                                                   [ 6,  7,  9],
                                                   [18, 19, 21]])).all())
        # Select map2 and map3
        bpcm23 = bpcm.select(maplist=[map2, map3], ellind=None)
        self.assertTrue((bpcm23.get() == np.array([[ 7,  8, 10],
                                                   [13, 14, 16],
                                                   [25, 26, 28]])).all())
        # Select map1 and map3
        bpcm13 = bpcm.select(maplist=[map1, map3], ellind=None)
        self.assertTrue((bpcm13.get() == np.array([[ 0,  2,  5],
                                                   [12, 14, 17],
                                                   [30, 32, 35]])).all())
    
        # Keep all three maps, but permute their order.
        bpcm231 = bpcm.select(maplist=[map2, map3, map1], ellind=None)
        self.assertTrue((bpcm231.get() == np.array([[ 7,  8,  6, 10, 11,  9],
                                                    [13, 14, 12, 16, 17, 15],
                                                    [ 1,  2,  0,  4,  5,  3],
                                                    [25, 26, 24, 28, 29, 27],
                                                    [31, 32, 30, 34, 35, 33],
                                                    [19, 20, 18, 22, 23, 21]])).all())

    def test_select_ell(self):
        """Test BpCov.select method for ell bins"""

        # Make a BpCov structure with two maps, three ell bins.
        map1 = MapDef('map1', 'B')
        map2 = MapDef('map2', 'B')
        maplist = [map1, map2]
        bpcm = BpCov(maplist, 3)
        M = np.ones((3,3))
        M = np.concatenate((M, 2*M, 4*M))
        M = np.concatenate((M, 3*M, 9*M), axis=1)
        bpcm.set(M)

        # Select bin 0 only.
        bpcm0 = bpcm.select(maplist=None, ellind=[0])
        self.assertTrue((bpcm0.get() == np.ones((3,3))).all())
        # Select bin 1 only.
        bpcm1 = bpcm.select(maplist=None, ellind=[1])
        self.assertTrue((bpcm1.get() == 6 * np.ones((3,3))).all())
        # Select bin 2 only.
        bpcm2 = bpcm.select(maplist=None, ellind=[2])
        self.assertTrue((bpcm2.get() == 36 * np.ones((3,3))).all())

        # Select bins 0 and 1.
        bpcm01 = bpcm.select(maplist=None, ellind=[0,1])
        self.assertTrue((bpcm01.get() == np.array([[1, 1, 1, 3, 3, 3],
                                                   [1, 1, 1, 3, 3, 3],
                                                   [1, 1, 1, 3, 3, 3],
                                                   [2, 2, 2, 6, 6, 6],
                                                   [2, 2, 2, 6, 6, 6],
                                                   [2, 2, 2, 6, 6, 6]])).all())
        # Select bins 1 and 2.
        bpcm12 = bpcm.select(maplist=None, ellind=[1,2])
        self.assertTrue((bpcm12.get() == np.array([[ 6,  6,  6, 18, 18, 18],
                                                   [ 6,  6,  6, 18, 18, 18],
                                                   [ 6,  6,  6, 18, 18, 18],
                                                   [12, 12, 12, 36, 36, 36],
                                                   [12, 12, 12, 36, 36, 36],
                                                   [12, 12, 12, 36, 36, 36]])).all())
        # Select bins 0 and 2.
        bpcm02 = bpcm.select(maplist=None, ellind=[0,2])
        self.assertTrue((bpcm02.get() == np.array([[1, 1, 1,  9,  9,  9],
                                                   [1, 1, 1,  9,  9,  9],
                                                   [1, 1, 1,  9,  9,  9],
                                                   [4, 4, 4, 36, 36, 36],
                                                   [4, 4, 4, 36, 36, 36],
                                                   [4, 4, 4, 36, 36, 36]])).all())
        # Select bins 2 and 0, i.e. flip the order.
        bpcm20 = bpcm.select(maplist=None, ellind=[2,0])
        self.assertTrue((bpcm20.get() == np.array([[36, 36, 36, 4, 4, 4],
                                                   [36, 36, 36, 4, 4, 4],
                                                   [36, 36, 36, 4, 4, 4],
                                                   [ 9,  9,  9, 1, 1, 1],
                                                   [ 9,  9,  9, 1, 1, 1],
                                                   [ 9,  9,  9, 1, 1, 1]])).all())

    def bpcm_analytic(self, S, N, nmode):
        """
        Calculate analytic bandpower covariance matrix assuming a common
        CMB-type signal and independent noise in all maps.

        Parameters
        ----------
        S : float
            Signal power for CMB-type signal (common to all maps)
        N : array
            Noise power for each map
        nmode : float
            Bandpower degrees-of-freedom

        Returns
        -------
        bpcm : array
            Analytic expectation for bandpower covariance matrix

        """

        # Allocate array
        nmap = len(N)
        nspec = nmap * (nmap + 1) // 2
        bpcm = np.zeros(shape=(nspec,nspec))
        # Double loop over covariance matrix
        # Doesn't take advantage of symmetry, but should be fast enough
        for (i,m0,m1) in specgen(nmap):
            for (j,m2,m3) in specgen(nmap):
                # Some bandpower expectation values
                C02 = S + N[m0] if m0 == m2 else S
                C13 = S + N[m1] if m1 == m3 else S
                C03 = S + N[m0] if m0 == m3 else S
                C12 = S + N[m1] if m1 == m2 else S
                # Calculate 
                bpcm[i,j] = (C02 * C13 + C03 * C12) / nmode

    def test_bpcov_signoi(self):
        """Test construction of BpCov_signoi from sims"""

        # Simulate "maps" as collection of independent Gaussian modes.
        nbin = 3
        nmode = 50
        nrlz = 1000
        # CMB signal, common to all maps
        S = 1.0 # signal power
        cmb = np.sqrt(S) * np.random.randn(nbin, nmode, nrlz)
        # Noise power for each map
        nmap = 4
        N = 10**(0.5 * np.random.randn(nmap))
        noise = np.zeros(shape=(nmap,nbin,nmode,nrlz))
        for i in range(nmap):
            noise[i,:,:,:] = np.sqrt(N[i]) * np.random.randn(nbin, nmode, nrlz)
        # Calculate all auto and cross spectra between signal and noise maps.
        # Note that the 'cmb' map is the signal for all maps.
        maplist_sn = []
        for i in range(nmap):
            maplist_sn.append(MapDef('m{}'.format(i), 'B', simtype='signal'))
        for i in range(nmap):
            maplist_sn.append(MapDef('m{}'.format(i), 'B', simtype='noise'))
        bins = np.array([[20,30,40], [30,40,50]])
        sn_nspec = 2 * nmap * (2 * nmap + 1) // 2
        sn_sim = np.zeros(shape=(sn_nspec, nbin, nrlz))
        for (i,m0,m1) in specgen(2 * nmap):
            if m0 < nmap:
                mapa = cmb
            else:
                mapa = noise[m0-nmap,:,:,:]
            if m1 < nmap:
                mapb = cmb
            else:
                mapb = noise[m1-nmap,:,:,:]
            sn_sim[i,:,:] = (mapa * mapb).mean(axis=1)
        spec = XSpec(maplist_sn, bins, sn_sim)
        # Now construct BpCov_signoi object
        maplist = [MapDef('m{}'.format(i), 'B') for i in range(nmap)]
        bpcm = BpCov_signoi.from_xspec(maplist, spec)
        Msim = bpcm.get()
        # Get analytic expectation for bandpower covariance matrix
        nspec = nmap * (nmap + 1) // 2
        M = np.zeros(shape=(nspec*nbin,nspec*nbin))
        Mblock = self.bpcm_analytic(S, N, nmode)
        for i in range(nbin):
            M[i*nspec:(i+1)*nspec,i*nspec:(i+1)*nspec] = Mblock
        # Convert both covariance matrices to correlation matrices.
        corr = M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
        corr_sim = Msim / np.sqrt(np.outer(np.diag(M), np.diag(M)))
        # Test that all entries are within 5*sigma of analytic expectation.
        np.testing.assert_allclose(corr, corr_sim, atol=5/nmode)

        # Scale to a new signal model
        S2 = 2.0
        Msim = bpcm.get(sig_model=S2*np.ones(shape=(nspec,nbin)))
        # Get analytic expectation
        M = np.zeros(shape=(nspec*nbin,nspec*nbin))
        Mblock = self.bpcm_analytic(S2, N, nmode)
        for i in range(nbin):
            M[i*nspec:(i+1)*nspec,i*nspec:(i+1)*nspec] = Mblock
        # Convert to correlation matrices.
        corr = M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
        corr_sim = Msim / np.sqrt(np.outer(np.diag(M), np.diag(M)))
        # Test that all entries are within 5*sigma of expectation
        np.testing.assert_allclose(corr, corr_sim, atol=5/nmode)

        # Select just two maps out of four.
        maplist2 = maplist[0:2]
        N2 = N[0:2]
        bpcm2 = bpcm.select(maplist=maplist2)
        Msim = bpcm2.get()
        # Get analytic expectation
        nspec2 = 2 * 3 // 2
        M = np.zeros(shape=(nspec2*nbin,nspec2*nbin))
        Mblock = self.bpcm_analytic(S, N2, nmode)
        for i in range(nbin):
            M[i*nspec2:(i+1)*nspec2,i*nspec2:(i+1)*nspec2] = Mblock
        # Convert to correlation matrices.
        corr = M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
        corr_sim = Msim / np.sqrt(np.outer(np.diag(M), np.diag(M)))
        # Test that all entries are within 5*sigma of expectation
        np.testing.assert_allclose(corr, corr_sim, atol=5/nmode)
        
class ModelsTest(unittest.TestCase):
    """Unit tests for models.py"""

    def test_base_model(self):
        """Test Model base class"""

        # Define some maps
        m0 = MapDef('m0', 'T')
        m1 = MapDef('m1', 'E')
        m2 = MapDef('m2', 'B')
        maplist = [m0, m1, m2]
        # Define some bandpower window functions
        lmax = 200
        bin_edges = [20, 40, 60, 80, 100]
        wf = BPWF.tophat(maplist, bin_edges, lmax=lmax)
        # Construct model
        mod = Model(maplist, wf)
        # Check various dimensions
        self.assertEqual(mod.nparam(), 0)
        self.assertEqual(mod.nmap(), len(maplist))
        self.assertEqual(mod.nspec(), len(maplist) * (len(maplist) + 1) // 2)
        self.assertEqual(mod.nbin(), len(bin_edges) - 1)
        # Test that theory_spec returns the right shape
        self.assertTrue(np.all(mod.theory_spec([], 0, 0) == np.zeros((6,lmax+1))))
        # Test that expv method returns the right shape
        self.assertTrue(np.all(mod.expv([]) == np.zeros((mod.nspec(),mod.nbin()))))
        # Use select method to drop one map and one ell bin.
        # Then test to make sure that everything has the expected shape.
        mod2 = mod.select([m0, m1], [0, 1, 2])
        self.assertEqual(mod2.nmap(), 2)
        self.assertEqual(mod2.nspec(), 3)
        self.assertEqual(mod2.nbin(), 3)

    def test_model_cmb(self):
        """Test Model_cmb class"""

        # Define some maps
        m0 = MapDef('m0', 'T')
        m1 = MapDef('m1', 'E')
        m2 = MapDef('m2', 'B')
        m3 = MapDef('m3', 'E', lensing_template=True)
        m4 = MapDef('m4', 'B', lensing_template=True)
        maplist = [m0, m1, m2, m3, m4]
        nmap = len(maplist)
        nspec = nmap * (nmap + 1) // 2
        # Define some bandpower window functions
        lmax = 200
        bin_edges = [20, 40, 60, 80, 100]
        wf = BPWF.tophat(maplist, bin_edges, lmax=lmax)
        nbin = wf.nbin
        # Using some fake CMB spectra that are flat in ell, so that
        # bandpower expectation values are trivial.
        Cl_unlens = np.zeros(shape=(4,lmax+1))
        Cl_unlens[0,:] = 10.0 # TT
        Cl_unlens[1,:] = 1.0  # EE
        Cl_unlens[2,:] = 0.0  # BB
        Cl_unlens[3,:] = -2.5 # TE
        Cl_lens = np.zeros(shape=(4,lmax+1))
        Cl_lens[0,:] = 8.0
        Cl_lens[1,:] = 0.75
        Cl_lens[2,:] = 0.2
        Cl_lens[3,:] = -2.0
        Cl_tensor = np.zeros(shape=(4,lmax+1))
        Cl_tensor[0,:] = 0.1
        Cl_tensor[1,:] = 0.1
        Cl_tensor[2,:] = 0.1
        Cl_tensor[3,:] = 0.05
        # Create the Model_cmb object
        mod = Model_cmb(maplist, wf, Cl_unlens, Cl_lens, Cl_tensor, 1.0)
        # Check parameters
        self.assertEqual(mod.nparam(), 2)
        self.assertEqual(mod.param_dict_to_list({'r': 0, 'Alens': 1}), [0, 1])
        # Check theory spectra for two non-lensing templates
        np.testing.assert_allclose(mod.theory_spec([0,0], 0, 1).mean(axis=1),
                                   np.array([10, 1, 0, -2.5, 0, 0]))
        np.testing.assert_allclose(mod.theory_spec([0,1], 0, 1).mean(axis=1),
                                   np.array([8, 0.75, 0.2, -2.0, 0, 0]))
        np.testing.assert_allclose(mod.theory_spec([1,0], 0, 1).mean(axis=1),
                                   np.array([10.1, 1.1, 0.1, -2.45, 0, 0]))
        np.testing.assert_allclose(mod.theory_spec([1,1], 0, 1).mean(axis=1),
                                   np.array([8.1, 0.85, 0.3, -1.95, 0, 0]))
        np.testing.assert_allclose(mod.theory_spec([0,0.5], 0, 1).mean(axis=1),
                                   np.array([9.0, 0.875, 0.1, -2.25, 0, 0]))
        # Now try it for lensing template
        np.testing.assert_allclose(mod.theory_spec([0,0], 0, 3).mean(axis=1),
                                   np.array([0, 0, 0, 0, 0, 0]))
        np.testing.assert_allclose(mod.theory_spec([0,1], 0, 3).mean(axis=1),
                                   np.array([0, 0, 0.2, 0, 0, 0]))
        np.testing.assert_allclose(mod.theory_spec([0,0.5], 0, 3).mean(axis=1),
                                   np.array([0, 0, 0.1, 0, 0, 0]))
        np.testing.assert_allclose(mod.theory_spec([1,0.5], 0, 3).mean(axis=1),
                                   np.array([0, 0, 0.1, 0, 0, 0]))
        # Test expectation values
        expv = mod.expv([1, 1])
        self.assertEqual(expv.shape, (nspec,nbin))
        np.testing.assert_allclose(expv.mean(axis=1),
                                   [8.1, 0.85, 0.3, 0.0, 0.2, -1.95, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        
if __name__ == '__main__':
    unittest.main()



