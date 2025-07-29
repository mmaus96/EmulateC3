import numpy as np

from classy import Class
from linear_theory import f_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from shapefit import shapefit_factor

# k vector to use:



class direct_fit_theory():
    
    def __init__(self, pars = None):
        if pars != None:
            w, omega_b,omega_cdm, h, logA, ns = pars
        else:
            w, omega_b,omega_cdm, h, logA, ns = [-1., 0.02237, 0.12, 0.6736, np.log(1e10 * 2.0830e-9), 0.9649]

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106
        
        As =  np.exp(logA)*1e-10
        w0 = w
        wa = 0.

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'm_ncdm': mnu,
            'tau_reio': 0.0568,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        fid_class = Class()
        fid_class.set(pkparams)
        fid_class.compute()
        
        self.theta_star = fid_class.theta_star_100()
        
        self.fid_class = fid_class
        
        self.kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )
        
    def get_fid_dists(self,z):
        
        speed_of_light = 2.99792458e5
        
        fid_class = self.fid_class
        h = fid_class.h()
        
        Hz_fid = fid_class.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz_fid = fid_class.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        fid_dists = (Hz_fid, chiz_fid)
        
        return fid_dists


    def compute_pell_tables_wcdm(self, pars, z, fid_dists= (None,None), ap_off=False, w0wa = False ):

        if w0wa:
            w0,wa, omega_b,omega_cdm, h, logA = pars
        else:
            w, omega_b,omega_cdm, h, logA = pars
            w0 = w
            wa = 0.
        Hzfid, chizfid = fid_dists
        speed_of_light = 2.99792458e5

        # omega_b = 0.02242
        
        As =  np.exp(logA)*1e-10#2.0830e-9
        ns = 0.9649

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'tau_reio': 0.0544,
            'z_reio': 7.,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        # Caluclate AP parameters
        Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        apar, aperp = Hzfid / Hz, chiz / chizfid

        if ap_off:
            apar, aperp = 1.0, 1.0

        # Calculate growth rate
        # fnu = pkclass.Omega_nu / pkclass.Omega_m()
        # f   = f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)
        f = pkclass.scale_independent_growth_factor_f(z)

        # Calculate and renormalize power spectrum
        ki = np.logspace(-3.0,1.0,200)
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] )
        # pi = (sigma8/pkclass.sigma8())**2 * pi

        # Now do the RSD
        modPT = LPT_RSD(ki, pi, kIR=0.2,use_Pzel = False,\
                    cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
        modPT.make_pltable(f, kv=self.kvec, apar=apar, aperp=aperp, ngauss=3)

        return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable, pkclass.sigma8(), pkclass.scale_independent_growth_factor(z),f
    
    def compute_pell_tables_w0wacdm(self, pars, z, fid_dists= (None,None), ap_off=False ):

        w0,wa,ns,omega_b,omega_cdm, h, logA = pars
        Hzfid, chizfid = fid_dists
        speed_of_light = 2.99792458e5

        # omega_b = 0.02242
        # w0 = w
        # wa = 0.
        As =  np.exp(logA)*1e-10#2.0830e-9
        ns = 0.9649

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 10.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'm_ncdm': mnu,
            # 'tau_reio': 0.0568,
            'z_reio': 7.,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        # Caluclate AP parameters
        Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        apar, aperp = Hzfid / Hz, chiz / chizfid

        if ap_off:
            apar, aperp = 1.0, 1.0

        # Calculate growth rate
        # fnu = pkclass.Omega_nu / pkclass.Omega_m()
        # f   = f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)
        f = pkclass.scale_independent_growth_factor_f(z)

        # Calculate and renormalize power spectrum
        ki = np.logspace(-3.0,1.0,200)
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] )
        # pi = (sigma8/pkclass.sigma8())**2 * pi

        # Now do the RSD
        modPT = LPT_RSD(ki, pi, kIR=0.2,use_Pzel = False,\
                    cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
        modPT.make_pltable(f, kv=self.kvec, apar=apar, aperp=aperp, ngauss=3)

        return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable, pkclass.sigma8(), pkclass.scale_independent_growth_factor(z),f
    
    def compute_pell_tables_SF(self,pars, z ):
    
        # OmegaM, h, sigma8 = pars
        # Hzfid, chizfid = fid_dists
        f_sig8,apar,aperp,m = pars
        
        pkclass = self.fid_class
        h = self.fid_class.h()

        sig8_z = pkclass.sigma(8,z,h_units=True)
        f = f_sig8 / sig8_z

        # Calculate and renormalize power spectrum
        ki = np.logspace(-3.0,1.0,200)
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] ) * np.exp( shapefit_factor(ki,m) )
        # pi = (sig8_z/pkclass.sigma8())**2 * pi

        # Now do the RSD
        modPT = LPT_RSD(ki, pi, kIR=0.2,use_Pzel = False,\
                    cutoff=10, extrap_min = -4, extrap_max = 3, N = 1000, threads=8, jn=5)
        modPT.make_pltable(f, kv=self.kvec, apar=apar, aperp=aperp, ngauss=3)

        return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable
    
    def compute_fsigma_s8(self,pars,z):
        
        f_sig8,_,_,m = pars
        
        h_fid = self.fid_class.h()
        rd_fid = self.fid_class.rs_drag()
        
        return f_sig8*np.exp(m/(1.2) * np.tanh(0.6*np.log((rd_fid*h_fid)/(8.0*h_fid)) ))
    
    def compute_theta_star_w0waCDM(self,pars):
        
        
        w0,wa, omega_b,omega_cdm, h, logA = pars
        
        As =  np.exp(logA)*1e-10#2.0830e-9
        ns = 0.9649

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

        pkparams = {
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'tau_reio': 0.0544,
            'z_reio': 7.,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        return pkclass.theta_star_100()