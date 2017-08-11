# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

"""
Tests for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, sysconfig, sys, os

from .utilities import random, check, make_test, setup_module, setup_method, teardown_module

def test_pycosmo_1():
    def fkt_pycosmo(y, lna, k, eta, hubble_a, tdot, omega_gam, omega_neu, omega_dm_0, omega_b_0, ha, rh, r_bph_a, xc_damp):
        a=np.exp(lna)
        psi=-y[0]-12./(rh*k*a)**2*(omega_gam*y[9]+omega_neu*y[6+2*10+3])
        dphidlna=psi-k**2/(3.*a**2*ha**2)*y[0]+0.5/(ha*rh)**2*(omega_dm_0*a**(-3)*y[1]+omega_b_0*a**(-3)*y[3]+4.*omega_gam*a**(-4)*y[5]+4.*omega_neu*a**(-4)*y[6+2*10+1])
        ppi=y[9]+y[6]+y[10]
        n_y=8+3*10
        dydlna=np.zeros(n_y, dtype=np.float32)
        dydlna[0]=dphidlna
        dydlna[1]=-k/(a*ha)*y[2]-3.*dphidlna
        dydlna[2]=-y[2]+k/(a*ha)*psi
        dydlna[3]=-k/(a*ha)*y[4]-3.*dphidlna
        dydlna[4]=-y[4]+k/(a*ha)*psi+tdot/r_bph_a/(a*ha)*(y[4]-3.*y[7])
        dydlna[5]=-k/(a*ha)*y[7]-dphidlna
        dydlna[6]=k/(a*ha)*(-y[8])+tdot/(a*ha)*(y[6]-ppi/2.)
        dydlna[7]=k/(3.*a*ha)*(y[5]-y[9]+psi)+tdot/(a*ha)*(y[7]-y[4]/3.)
        dydlna[8]=k/(a*ha)/3.*(y[6]-2.*y[10])+tdot/(a*ha)*(y[8])
        dydlna[9]=k/(5.*a*ha)*(2.*y[7]-3.*y[11])+tdot/(a*ha)*(y[9]-ppi/10.)
        dydlna[10]=k/(a*ha)/5.*(2.*y[8]-3.*y[12])+tdot/(a*ha)*(y[10]-ppi/5.)
        for i in range(3,10):
            dydlna[5+2*i]=k/(a*ha)/(2.*i+1.)*(i*y[5+2*(i-1)]-(i+1.)*y[5+2*(i+1)])+tdot/(a*ha)*y[5+2*i]
            dydlna[6+2*i]=k/(a*ha)/(2.*i+1.)*(i*y[6+2*(i-1)]-(i+1.)*y[6+2*(i+1)])+tdot/(a*ha)*y[6+2*i]
        dydlna[5+2*10]=1./(a*ha)*(k*y[5+2*(10-1)]-((10+1.)/eta-tdot)*y[5+2*10])
        dydlna[6+2*10]=1./(a*ha)*(k*y[6+2*(10-1)]-((10+1.)/eta-tdot)*y[6+2*10])
        dydlna[6+2*10+1]=-k/(a*ha)*y[6+2*10+2]-dphidlna
        dydlna[6+2*10+2]=k/(3.*a*ha)*(y[6+2*10+1]-y[6+2*10+3]+psi)
        for j in range(2,10):
            dydlna[6+2*10+1+j]=k/(a*ha)/(2.*j+1.)*(j*y[6+2*10+1+j-1]-(j+1.)*y[6+2*10+1+j+1])
        dydlna[6+2*10+1+10]=1./(a*ha)*(k*y[6+2*10+1+10-1]-(10+1.)/eta*y[6+2*10+1+10])
        if xc_damp>0:
            dydlna[5:n_y-1]=y[5:n_y-1]*0.5*(1.-np.tanh((k*eta-xc_damp)/50.))
        return dydlna
    y = np.array([  \
       6.99174287e-01,  9.02477138e-01, -1.39934020e-06,  9.02477138e-01, \
      -1.39934020e-06,  3.00825713e-01,  0.00000000e+00, -4.66446734e-07, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.00825713e-01, \
      -4.66446734e-07, -4.33950717e-13,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,], dtype=np.float32)
    lna, k, eta, hubble_a, tdot = -16.118095651, 0.0001, 0.0465166421846, 6.44541327136e+13, -58991230.1681
    omega_gam, omega_neu, omega_dm_0, omega_b_0 = 5.04081632653e-05, 3.43441882421e-05, 0.255, 0.045
    ha, rh, r_bph_a, xc_damp = 214995844.604, 4282.7494, 6.69534412955e-05, 1

    hfkt = hope.jit(fkt_pycosmo)
    ro = fkt_pycosmo(y, lna, k, eta, hubble_a, tdot, omega_gam, omega_neu, omega_dm_0, omega_b_0, ha, rh, r_bph_a, xc_damp)
    rh = hfkt(y, lna, k, eta, hubble_a, tdot, omega_gam, omega_neu, omega_dm_0, omega_b_0, ha, rh, r_bph_a, xc_damp)
    assert check(ro, rh)

def test_pycosmo_1_opt():
    def fkt_pycosmo_opt(y, lna, k, eta, hubble_a, tdot, omega_gam, omega_neu, omega_dm_0, omega_b_0, ha, rh, r_bph_a, xc_damp):
        a=np.exp(lna)
        psi=-y[0]-12./(rh*k*a)**2*(omega_gam*y[9]+omega_neu*y[6+2*10+3])
        dphidlna=psi-k**2/(3.*a**2*ha**2)*y[0]+0.5/(ha*rh)**2*(omega_dm_0*a**(-3)*y[1]+omega_b_0*a**(-3)*y[3]+4.*omega_gam*a**(-4)*y[5]+4.*omega_neu*a**(-4)*y[6+2*10+1])
        ppi=y[9]+y[6]+y[10]
        n_y=8+3*10
        dydlna=np.zeros(n_y, dtype=np.float32)
        dydlna[0]=dphidlna
        dydlna[1]=-k/(a*ha)*y[2]-3.*dphidlna
        dydlna[2]=-y[2]+k/(a*ha)*psi
        dydlna[3]=-k/(a*ha)*y[4]-3.*dphidlna
        dydlna[4]=-y[4]+k/(a*ha)*psi+tdot/r_bph_a/(a*ha)*(y[4]-3.*y[7])
        dydlna[5]=-k/(a*ha)*y[7]-dphidlna
        dydlna[6]=k/(a*ha)*(-y[8])+tdot/(a*ha)*(y[6]-ppi/2.)
        dydlna[7]=k/(3.*a*ha)*(y[5]-y[9]+psi)+tdot/(a*ha)*(y[7]-y[4]/3.)
        dydlna[8]=k/(a*ha)/3.*(y[6]-2.*y[10])+tdot/(a*ha)*(y[8])
        dydlna[9]=k/(5.*a*ha)*(2.*y[7]-3.*y[11])+tdot/(a*ha)*(y[9]-ppi/10.)
        dydlna[10]=k/(a*ha)/5.*(2.*y[8]-3.*y[12])+tdot/(a*ha)*(y[10]-ppi/5.)
        for i in range(3,10):
            dydlna[5+2*i]=k/(a*ha)/(2.*i+1.)*(i*y[5+2*(i-1)]-(i+1.)*y[5+2*(i+1)])+tdot/(a*ha)*y[5+2*i]
            dydlna[6+2*i]=k/(a*ha)/(2.*i+1.)*(i*y[6+2*(i-1)]-(i+1.)*y[6+2*(i+1)])+tdot/(a*ha)*y[6+2*i]
        dydlna[5+2*10]=1./(a*ha)*(k*y[5+2*(10-1)]-((10+1.)/eta-tdot)*y[5+2*10])
        dydlna[6+2*10]=1./(a*ha)*(k*y[6+2*(10-1)]-((10+1.)/eta-tdot)*y[6+2*10])
        dydlna[6+2*10+1]=-k/(a*ha)*y[6+2*10+2]-dphidlna
        dydlna[6+2*10+2]=k/(3.*a*ha)*(y[6+2*10+1]-y[6+2*10+3]+psi)
        for j in range(2,10):
            dydlna[6+2*10+1+j]=k/(a*ha)/(2.*j+1.)*(j*y[6+2*10+1+j-1]-(j+1.)*y[6+2*10+1+j+1])
        dydlna[6+2*10+1+10]=1./(a*ha)*(k*y[6+2*10+1+10-1]-(10+1.)/eta*y[6+2*10+1+10])
        if xc_damp>0:
            dydlna[5:n_y-1]=y[5:n_y-1]*0.5*(1.-np.tanh((k*eta-xc_damp)/50.))
        return dydlna
    y = np.array([  \
       6.99174287e-01,  9.02477138e-01, -1.39934020e-06,  9.02477138e-01, \
      -1.39934020e-06,  3.00825713e-01,  0.00000000e+00, -4.66446734e-07, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.00825713e-01, \
      -4.66446734e-07, -4.33950717e-13,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, \
       0.00000000e+00,  0.00000000e+00,], dtype=np.float32)
    lna, k, eta, hubble_a, tdot = -16.118095651, 0.0001, 0.0465166421846, 6.44541327136e+13, -58991230.1681
    omega_gam, omega_neu, omega_dm_0, omega_b_0 = 5.04081632653e-05, 3.43441882421e-05, 0.255, 0.045
    ha, rh, r_bph_a, xc_damp = 214995844.604, 4282.7494, 6.69534412955e-05, 1
    hope.config.optimize = True
    hfkt = hope.jit(fkt_pycosmo_opt)
    ro = fkt_pycosmo_opt(y, lna, k, eta, hubble_a, tdot, omega_gam, omega_neu, omega_dm_0, omega_b_0, ha, rh, r_bph_a, xc_damp)
    rh = hfkt(y, lna, k, eta, hubble_a, tdot, omega_gam, omega_neu, omega_dm_0, omega_b_0, ha, rh, r_bph_a, xc_damp)
    assert check(ro, rh)
    hope.config.optimize = False

def test_pycosmo_2():
    def fkt_pycosmo_2(y, lna, dydlna, k, ha_lna, eta_lna, taudot_lna, lmax, H0, omega_r_0, omega_m_0, omega_k_0, omega_l_0, rh, omega_gam, omega_neu, omega_dm_0, omega_b_0, xc_damp, a_tca):
        a = np.exp(lna)
        r_bph_a = 3./4.*omega_b_0/omega_gam*a
        idx_f = (lna - ha_lna[0]) / (ha_lna[2] - ha_lna[0]) * 2.
        idx_i = np.floor(idx_f)
        idx = np.int_(idx_i)
        ratio = idx_f - idx_i
        eta = (1 - ratio) * eta_lna[idx] + ratio * eta_lna[idx + 1]
        ha = (H0*(omega_r_0*a**-4+omega_m_0*a**-3+omega_k_0*a**-2+omega_l_0)**0.5)/H0/rh
        tdot = (1 - ratio) * taudot_lna[idx] + ratio * taudot_lna[idx + 1]
        psi=-y[0]-12./(rh*k*a)**2*(omega_gam*y[9]+omega_neu*y[6+2*lmax+3])
        dphidlna=psi-k**2/(3.*a**2*ha**2)*y[0]+0.5/(ha*rh)**2*(omega_dm_0*a**(-3)*y[1]+omega_b_0*a**(-3)*y[3]+4.*omega_gam*a**(-4)*y[5]+4.*omega_neu*a**(-4)*y[6+2*lmax+1])
        ppi=y[9]+y[6]+y[10]
        n_y=8+3*lmax
        dydlna[:]=0
        dydlna[0]=dphidlna 
        dydlna[1]=-k/(a*ha)*y[2]-3.*dphidlna  
        dydlna[2]=-y[2]+k/(a*ha)*psi  
        dydlna[3]=-k/(a*ha)*y[4]-3.*dphidlna  
        dydlna[5]=-k/(a*ha)*y[7]-dphidlna                                   
        dydlna[6]=k/(a*ha)*(-y[8])+tdot/(a*ha)*(y[6]-ppi/2.)                
        if a<a_tca:   
            dh_dlna=-1./(2.*ha)*(4.*omega_r_0*a**-4+3.*omega_m_0*a**-3+2.*omega_k_0*a**-2)/rh**2   
            slip=2./(1.+r_bph_a)*(y[4]-3.*y[7])+1./tdot/(1+1./r_bph_a)*( (2.*a*ha + a*dh_dlna)*y[4]+k*(2.*y[5]+psi) + k*dydlna[5] ) 
            dydlna[4]=-y[4]/(1.+1./r_bph_a)+k/(a*ha)*((y[5]-2.*y[9])/(1.+r_bph_a)+ psi)+slip/(1.+r_bph_a)  
            dydlna[7]=k/(3.*a*ha)*(y[5]-2.*y[9]+(1.+r_bph_a)*psi)-r_bph_a/3.*(dydlna[4]+y[4])   
            dydlna[8]=k/(a*ha)/3.*(y[6]-2.*y[10])+tdot/(a*ha)*(y[8])            
        if a>=a_tca:   
            dydlna[4]=-y[4]+k/(a*ha)*psi+tdot/r_bph_a/(a*ha)*(y[4]-3.*y[7])  
            dydlna[7]=k/(3.*a*ha)*(y[5]-2.*y[9]+psi)+tdot/(a*ha)*(y[7]-y[4]/3.)    
            dydlna[8]=k/(a*ha)/3.*(y[6]-2.*y[10])+tdot/(a*ha)*(y[8])            
            dydlna[9]=k/(5.*a*ha)*(2.*y[7]-3.*y[11])+tdot/(a*ha)*(y[9]-ppi/10.) 
            dydlna[10]=k/(a*ha)/5.*(2.*y[8]-3.*y[12])+tdot/(a*ha)*(y[10]-ppi/5.)
            for i in range(3,lmax):   
                dydlna[5+2*i]=k/(a*ha)/(2.*i+1.)*(i*y[5+2*(i-1)]-(i+1.)*y[5+2*(i+1)])+tdot/(a*ha)*y[5+2*i] 
                dydlna[6+2*i]=k/(a*ha)/(2.*i+1.)*(i*y[6+2*(i-1)]-(i+1.)*y[6+2*(i+1)])+tdot/(a*ha)*y[6+2*i] 
            dydlna[5+2*lmax]=1./(a*ha)*(k*y[5+2*(lmax-1)]-((lmax+1.)/eta-tdot)*y[5+2*lmax]) 
            dydlna[6+2*lmax]=1./(a*ha)*(k*y[6+2*(lmax-1)]-((lmax+1.)/eta-tdot)*y[6+2*lmax]) 

        dydlna[6+2*lmax+1]=-k/(a*ha)*y[6+2*lmax+2]-dphidlna                 
        dydlna[6+2*lmax+2]=k/(3.*a*ha)*(y[6+2*lmax+1]-y[6+2*lmax+3]+psi)    
        for j in range(2,lmax):   
            dydlna[6+2*lmax+1+j]=k/(a*ha)/(2.*j+1.)*(j*y[6+2*lmax+1+j-1]-(j+1.)*y[6+2*lmax+1+j+1]) 
        dydlna[6+2*lmax+1+lmax]=1./(a*ha)*(k*y[6+2*lmax+1+lmax-1]-(lmax+1.)/eta*y[6+2*lmax+1+lmax]) 
        if xc_damp>0: 
            tanhArg = (k*eta-xc_damp)/50.
            tanhA = np.fabs(tanhArg)
            tanhB = 1.26175667589988239 + tanhA * (-0.54699348440059470 + tanhA * 2.66559097474027817)
            damping = (1.- tanhB * tanhArg / (tanhB * tanhA + 1)) / 2.
            dydlna[5:n_y-1]=dydlna[5:n_y-1]*damping 
        return dydlna 
    y = np.array([  6.99174287e-01,   9.02477138e-01,  -9.79336020e-07, \
             9.02477138e-01,  -9.79336020e-07,   3.00825713e-01, \
             0.00000000e+00,  -3.26445340e-07,   0.00000000e+00, \
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
             3.00825713e-01,  -3.26445340e-07,  -2.12548108e-13, \
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
             0.00000000e+00,   0.00000000e+00])
    lna = -16.11809565095832
    dydlna = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., \
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., \
            0.,  0.,  0.])
    k = 0.0001
    ha_lna = np.array([-16.11809565, -16.10233994, -16.08658422])
    eta_lna = np.array([3.25549306e-02,   3.30717641e-02,   3.35968002e-02])
    taudot_lna = np.array([ -9.36368733e+06,  -9.07322465e+06,  -8.79177215e+06])
    lmax = 7
    H0 = 70.0
    omega_r_0 = 8.475235150739534e-05
    omega_m_0 = 1.0
    omega_k_0 = 0.0
    omega_l_0 = -8.475235150739534e-05
    rh = 2997.9245799999994
    omega_gam = 5.040816326530613e-05
    omega_neu = 3.434418824208921e-05
    omega_dm_0 = 0.995
    omega_b_0 = 0.005
    xc_damp = 1000.0
    a_tca = 2.82992247647206e-05

    hope.config.optimize = True
    hfkt = hope.jit(fkt_pycosmo_2)
    ro = fkt_pycosmo_2(y, lna, dydlna, k, ha_lna, eta_lna, taudot_lna, lmax, H0, omega_r_0, omega_m_0, omega_k_0, omega_l_0, rh, omega_gam, omega_neu, omega_dm_0, omega_b_0, xc_damp, a_tca)
    rh = hfkt(y, lna, dydlna, k, ha_lna, eta_lna, taudot_lna, lmax, H0, omega_r_0, omega_m_0, omega_k_0, omega_l_0, rh, omega_gam, omega_neu, omega_dm_0, omega_b_0, xc_damp, a_tca)
    assert check(ro, rh)
    hope.config.optimize = False
