import numpy as np
import time
import json
import yaml

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
from scipy.integrate import simps

from taylor_approximation import taylor_approximate
# from Compute_zParams_class import Compute_zParams
from compute_pell_tables_direct import direct_fit_theory
from linear_theory import D_of_a,f_of_a

# from make_pkclass_aba import make_pkclass
from copy import deepcopy

# Class to have a shape-fit likelihood for a bunch of pieces of data from both galactic caps in the same z bin
# Currently assumes all data have the same fiducial cosmology etc.
# If not I suggest changing the theory class so that instead of being labelled by "zstr" it gets labelled by sample name.
# And each sample name indexing the fiducial cosmology numbers (chi, Hz etc) in dictionaries. For another time...

class PkLikelihood(Likelihood):
    
    zfids: list
    #sig8: float
    
    basedir: str
    
    fs_sample_names: list

    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool 

    fs_datfns: list

    covfn: str
    
    fs_kmins: list
    fs_mmaxs: list
    fs_qmaxs: list
    fs_hmaxs: list
    fs_matMfns: list
    fs_matWfns: list
    # pypower_kin_fn: list
    # pypower_kout_fn: list
    
    w_kin_fn: str
    kmax_spline: list
    cov_fac: float
    inv_cov_fac: float
    hexa: bool
    jeff: bool
    ext_prior: bool
    redef: bool
    
    compute_p0: bool
    compute_p2: bool
    compute_p4: bool
    


    def initialize(self):
        """Sets up the class."""
        # Redshift Label for theory classes
        # self.zstr = "%.2f" %(self.zfid)
        print(self.fs_sample_names,self.fs_datfns)

	# Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.linear_param_dict = yaml.load(open(self.basedir+self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict)
        
        print("We are here!")
        
        self.compute_p0 = True
        self.compute_p2 = True
        self.compute_p4 = True
        
        self.pconv = {}
        self.xith = {}
        
#         self.sp_kmax = {}
        
#         for ll, fs_sample_name in enumerate(self.fs_sample_names):
#             self.sp_kmax[fs_sample_name] = self.kmax_spline[ll]

        if self.ext_prior:
            raise Exception('External prior not setup in this likelihood script')
            
        if not self.redef:
            raise Exception('Only redef is possible in this likelihood script')    
        
        self.loadData()
        #

    def get_requirements(self):
        
        req = {'taylor_pk_ell_mod': None,\
               'zPars': None,\
               'w0': None,\
               'wa': None,\
               'ns': None,\
               'H0': None,\
               'sigma8': None,\
               'omegam': None,\
               'omega_b': None,\
               'omega_cdm': None,\
               'logA': None}
        
        for fs_sample_name in self.fs_sample_names:
            req_bias = { \
                   'bsig8_' + fs_sample_name: None,\
                   # 'b1_' + fs_sample_name: None,\
                   'b2sig8_' + fs_sample_name: None,\
                   'bssig8_' + fs_sample_name: None,\
                   'b3sig8_' + fs_sample_name: None,\
                  # 'alpha0_' + fs_sample_name: None,\
                  #'alpha2_' + fs_sample_name: None,\
                  # 'SN0_' + fs_sample_name: None,\
                   #'SN2_' + fs_sample_name: None\
                   }
            req = {**req, **req_bias}

        return(req)

    def full_predict(self, thetas=None):
        
        thy_obs = []

        if thetas is None:
            thetas = self.linear_param_means
        
        for zfid,fs_sample_name in zip(self.zfids,self.fs_sample_names):
            fs_thy  = self.fs_predict(fs_sample_name,zfid,thetas=thetas)
            fs_obs  = self.fs_observe(fs_thy, fs_sample_name)
            thy_obs = np.concatenate( (thy_obs,fs_obs) )
        
        return thy_obs
    
    def logp(self,**params_values):
        """Return a log-likelihood."""

        # Compute the theory prediction with lin. params. at prior mean
        #t1 = time.time()
        thy_obs_0 = self.full_predict()
        self.Delta = self.dd - thy_obs_0
        #t2 = time.time()
        
        # Now compute template
        self.templates = []
        for param in self.linear_param_dict.keys():
            thetas = self.linear_param_means.copy()
            thetas[param] += 1.0
            self.templates += [ self.full_predict(thetas=thetas) - thy_obs_0 ]
        
        self.templates = np.array(self.templates)
        #t3 = time.time()
        
        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1./self.linear_param_stds**2)
        #self.Va = np.einsum('ij,jk,k', self.templates, self.cinv, self.Delta)
        #self.Lab = np.einsum('ij,jk,lk', self.templates, self.cinv, self.templates) + np.diag(1./self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)
        #t4 = time.time()
        
        # Compute the modified chi2
        lnL  = -0.5 * np.dot(self.Delta,np.dot(self.cinv,self.Delta)) # this is the "bare" lnL
        lnL +=  0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va)) # improvement in chi2 due to changing linear params
        self.detFish = 0.5 * np.log( np.linalg.det(self.Lab) )
        if not self.optimize:
            if self.jeff:
                lnL += 0.5 * self.Nlin * np.log(2*np.pi)
            else:
                lnL += - 0.5 * np.log( np.linalg.det(self.Lab) ) + 0.5 * self.Nlin * np.log(2*np.pi) # volume factor from the determinant
        
        #t5 = time.time()
        
        #print(t2-t1, t3-t2, t4-t3, t5-t4)
        
        return lnL
    
    def get_best_fit(self):
        try:
            self.p0_nl  = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.p0_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.p0_nl + self.p0_lin
        except:
            print("Make sure to first compute the posterior.")
        
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        # First load the data
        
        self.kdats = {}
        self.p0dats = {}
        self.p2dats = {}
        self.p4dats = {}
        self.fitiis = {}
        
        for ii, fs_datfn in enumerate(self.fs_datfns):
            fs_sample_name = self.fs_sample_names[ii]
            fs_dat = np.loadtxt(self.basedir+fs_datfn)
            self.kdats[fs_sample_name] = fs_dat[:,0]
            self.p0dats[fs_sample_name] = fs_dat[:,1]
            self.p2dats[fs_sample_name] = fs_dat[:,2]
            self.p4dats[fs_sample_name] = fs_dat[:,3]
            
            # Make a list of indices for the monopole and quadrupole only in Fourier space
            # This is specified to each sample in case the k's are different.
            yeses = self.kdats[fs_sample_name] > 0
            nos   = self.kdats[fs_sample_name] < 0
            self.fitiis[fs_sample_name] = np.concatenate( (yeses, nos, yeses, nos, yeses ) )
        

        
        # Join the data vectors together
        self.dd = []        
        for fs_sample_name in self.fs_sample_names:
            self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name], self.p4dats[fs_sample_name]) )

        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.basedir+self.covfn)/self.cov_fac
        
        # We're only going to want some of the entries in computing chi^2.
        
        # this is going to tell us how many indices to skip to get to the nth multipole
        startii = 0
        
        for ss, fs_sample_name in enumerate(self.fs_sample_names):
            
            kcut = (self.kdats[fs_sample_name] > self.fs_mmaxs[ss])\
                          | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:     # FS Monopole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_qmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_hmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Hexadecapole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)*self.inv_cov_fac
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        # Finally load the window function matrix.
        self.matMs = {}
        self.matWs = {}
        # self.pyp_kin = {}
        # self.pyp_kout = {}
        for ii, fs_sample_name in enumerate(self.fs_sample_names):
            self.matMs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matMfns[ii])
            self.matWs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matWfns[ii])
            # self.pyp_kin[fs_sample_name] = np.loadtxt(self.basedir+self.pypower_kin_fn[ii])
            # self.pyp_kout[fs_sample_name] = np.loadtxt(self.basedir+self.pypower_kout_fn[ii])
        
        self.w_kin = np.loadtxt(self.basedir+self.w_kin_fn)
        
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        return p0, p2, p4
    
    def fs_predict(self, fs_sample_name,zfid, thetas=None):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        zstr = "%.2f" %(zfid)
        taylorPTs = pp.get_result('taylor_pk_ell_mod')
        kv, p0ktable, p2ktable, p4ktable = taylorPTs[zstr]
        
        if not hasattr(self, "w0waCDM_multipoles"):
            # self.w0waCDM_multipoles = {}
            self.w0waCDM_multipoles = np.zeros((59,19,3))

        if self.compute_p0:
            self.w0waCDM_multipoles[ :,:, 0] = p0ktable
        if self.compute_p2:
            self.w0waCDM_multipoles[ :,:, 1] = p2ktable
        if self.compute_p4:
            self.w0waCDM_multipoles[ :,:, 2] = p4ktable
        
        # pkclass = make_pkclass(self.zfid)

        #
        sig8 = pp.get_param('sigma8')
        Om = pp.get_param('omegam')
        
        zPars = pp.get_result('zPars')
        D_z = zPars[zstr][0]
        f_z = zPars[zstr][1]
        sig8_z = sig8*D_z
        # sig8_z = sig8*D_of_a(1./(1.+zfid),OmegaM=Om)
        # f_z = f_of_a(1./(1.+zfid),OmegaM=Om)
        #sig8 = pp.get_result('sigma8')
        b1   = pp.get_param('bsig8_' + fs_sample_name)/sig8_z - 1
        # b1   = pp.get_param('b1_' + fs_sample_name)
        b2   = pp.get_param('b2sig8_' + fs_sample_name)/(sig8_z**2)
        bs   = pp.get_param('bssig8_' + fs_sample_name)/(sig8_z**2)
        b3   = pp.get_param('b3sig8_' + fs_sample_name)/(sig8_z**3)


       # Instead of calling the linear parameters directly we will now analytically marginalize over them
        
        if thetas is None:
            alp0_tilde = self.linear_param_means['alpha0_' + fs_sample_name]
            alp2_tilde = self.linear_param_means['alpha2_' + fs_sample_name]
            sn0 = self.linear_param_means['SN0_' + fs_sample_name]
            sn2 = self.linear_param_means['SN2_' + fs_sample_name]
            if self.hexa:
                alp4_tilde = self.linear_param_means['alpha4_' + fs_sample_name]
                sn4 = self.linear_param_means['SN4_' + fs_sample_name]
            else: alp4_tilde,sn4 = 0.,0.
        else:
            alp0_tilde = thetas['alpha0_' + fs_sample_name]
            alp2_tilde = thetas['alpha2_' + fs_sample_name]
            sn0 = thetas['SN0_' + fs_sample_name]
            sn2 = thetas['SN2_' + fs_sample_name]
            
            if self.hexa:
                alp4_tilde = thetas['alpha4_' + fs_sample_name]
                sn4 = thetas['SN4_' + fs_sample_name]
            else: alp4_tilde,sn4 = 0.,0.
            
        alp0 = (1+b1)**2 * alp0_tilde
        alp2 = f_z*(1+b1)*(alp0_tilde+alp2_tilde)
        alp4 = f_z*(f_z*alp2_tilde+(1+b1)*alp4_tilde)
        alp6 = f_z**2*alp4_tilde
 
        bias = [b1, b2, bs, b3]
        cterm = [alp0,alp2,alp4,alp6]
        stoch = [sn0, sn2, sn4]
        bvec = bias + cterm + stoch
        self.bvec = bvec
        
        #print(self.zstr, b1, sig8)
        
        p0, p2, p4 = self.combine_bias_terms_pkell(bvec, p0ktable, p2ktable, p4ktable)
        
        #np.savetxt('pells_' + self.zstr + '_' + self.sample_name + '.txt',[kv,p0,p2,p4])
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([0.0,],p0)
        p2 = np.append([0.,],p2)
        p4 = np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        
#         if np.any(np.isnan(tt)):
#             w0 = self.provider.get_param('w0') 
#             wa = self.provider.get_param('wa')
            
#             H0 = self.provider.get_param('H0') 
#             ns = self.provider.get_param('ns')
#             OmM = self.provider.get_param('omegam')
#             print("NaN's encountered. Parameter values are: ", str([w0,wa,ns,H0,OmM]))
        
        return(tt)
        #

    def fs_observe(self,tt,fs_sample_name):
        """Apply the window function matrix to get the binned prediction."""
        
        
        # print(fs_sample_name)
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        # kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        # maxk = self.sp_kmax[fs_sample_name]
        # kv  = np.linspace(0.0,maxk,int(maxk/0.001),endpoint=False) + 0.0005
        kv  = self.w_kin
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,1],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        
#         if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
#             w0 = self.provider.get_param('w0') 
#             wa = self.provider.get_param('wa')
            
#             H0 = self.provider.get_param('H0') 
#             ns = self.provider.get_param('ns')
#             OmM = self.provider.get_param('omegam')
#             print("NaN's encountered. Parameter values are: ", str([w0,wa,ns,H0,OmM]))
        
        # wide angle
        # expanded_model = np.matmul(self.matMs[fs_sample_name], thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        # Multiply by ad-hoc factor
        # convolved_model = np.matmul(self.matWs[fs_sample_name], expanded_model )
        convolved_model = np.matmul(self.matWs[fs_sample_name], thy )
        
        # keep only the monopole and quadrupole
        # convolved_model = convolved_model[self.fitiis[fs_sample_name]]
        
        # Save the model:
        self.pconv[fs_sample_name] = convolved_model
    
        return convolved_model


class Taylor_pk_theory_zs(Theory):
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    zfids: list
    # pk_filenames: list
    # s8_filenames: list
    # basedir: str
    omega_nu: float
    
    def initialize(self):
        """Sets up the class by loading the derivative matrices."""
        
        print("Loading Taylor series.")
        
        # Load sigma8
        # self.compute_sigma8 = Compute_Sigma8_wCDM(self.basedir + self.s8_filename)
        # self.compute_zpars = Compute_zParams(self.basedir + self.s8_filenames)
        
        # Load clustering
        # self.taylors_pk = {}
        # self.s8_emus = {}
        self.compute_theory = direct_fit_theory()
        
        self.fid_dists = {}
        
        for zfid in self.zfids:
            zstr = "%.2f"%(zfid)
            
            self.fid_dists[zstr] =  self.compute_theory.get_fid_dists(zfid)
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {\
               'w0': None,\
               'wa': None,\
               'ns': None,\
               'omega_b': None,\
               'omega_cdm': None,\
               'H0': None,\
               'logA': None,\
              }
        
        return(req)
    
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_pk_ell_mod','zPars']
    
    def get_can_provide_params(self):
        return ['sigma8','omegam']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        
        w0 = pp.get_param('w0')
        wa = pp.get_param('wa')
        ns = pp.get_param('w0')
        hub = pp.get_param('H0') / 100.
        logA = pp.get_param('logA')
        omega_b = pp.get_param('omega_b')
        omega_cdm = pp.get_param('omega_cdm')
        
        OmM = (omega_cdm + omega_b + self.omega_nu)/hub**2
        #sig8 = pp.get_param('sigma8')
        # OmM = pp.get_param('omegam')
        # sig8 = self.compute_sigma8.compute_sigma8(w,OmM,hub,logA)
        cosmopars = [w0,wa,ns,omega_b, omega_cdm, hub, logA]
        
        ptables = {}
        zPars = {}
        
        for zfid in self.zfids:
            zstr = "%.2f" %(zfid)
            
            # Load pktables
#             x0s = self.taylors_pk[zstr]['x0']
#             derivs0 = self.taylors_pk[zstr]['derivs_p0']
#             derivs2 = self.taylors_pk[zstr]['derivs_p2']
#             derivs4 = self.taylors_pk[zstr]['derivs_p4']
            
#             kv = self.taylors_pk[zstr]['kvec']
#             p0ktable = taylor_approximate(cosmopars, x0s, derivs0, order=4)
#             p2ktable = taylor_approximate(cosmopars, x0s, derivs2, order=4)
#             p4ktable = taylor_approximate(cosmopars, x0s, derivs4, order=4)

            fid_dists = self.fid_dists[zstr]
    
            kv = self.compute_theory.kvec
        
            try:
                p0ktable, p2ktable, p4ktable, sig8, Dz, fz = self.compute_theory.compute_pell_tables_w0wacdm(cosmopars, zfid, fid_dists)
            except Exception as e:  # Catch any exception using the general Exception class
                print("An error in compute_pell_tables occurred:", e)
                p0ktable, p2ktable, p4ktable = np.ones((59,19)),np.ones((59,19)),np.ones((59,19))
                p0ktable[:,:3] *= 1e10
                p2ktable[:,:3] *= 1e10
                p4ktable[:,:3] *= 1e10
                sig8, Dz, fz = 1.0,1.0,1.0
            
            # if w0+wa >= 0:
            #     p0ktable, p2ktable, p4ktable = np.ones((59,19)),np.ones((59,19)),np.ones((59,19))
            #     p0ktable[:,:3] *= 1e10
            #     p2ktable[:,:3] *= 1e10
            #     p4ktable[:,:3] *= 1e10
            #     sig8, Dz, fz = 1.0,1.0,1.0
            # else:
            #     p0ktable, p2ktable, p4ktable, sig8, Dz, fz = self.compute_theory.compute_pell_tables_w0wacdm(cosmopars, zfid, fid_dists)
            
            ptables[zstr] = (kv, p0ktable, p2ktable, p4ktable)
            
            
            zPars[zstr] = [Dz,fz]
            
        #state['sigma8'] = sig8
        # state['derived'] = {'sigma8': sig8,'omegam':OmM}
        state['derived']['sigma8'] = sig8
        state['derived']['omegam'] = OmM
        state['taylor_pk_ell_mod'] = ptables
        state['zPars'] = zPars
