"""Calculation sheet for annular valve.

Created on 14.12.2021

@author: Timothy Loayza
"""

import numpy as np
import math
from matplotlib import pyplot as plt


class MRF:
    """Class to keep all awesome magneto things."""

    def __init__(self, step_t=10**(-6), nb_step=10**5,
                 d=1*10**(-4), g=3*10**(-4), length=6*10**(-3), fi=0.02, d_eff_disp=10**(-4),
                 R=4.1, N_e=100, N_s=100, b=0.1, t_0_agglo = 0.0,
                 i=np.ones(10**5), dP=np.zeros(10**5)):

        self.record_audio = None

        # SIMULATION
        self.step_t = step_t
        self.nb_step = nb_step

        self.mu_0 = 1.25663*10**(-6)
        self.R = R
        self.N_e = N_e
        self.N_s = N_s
        self.H0 = 16_000 # approximated for the LORD MRF 140 

        self.d = d
        self.g = g
        self.len = length
        self.fi = fi
        self.radius = 5*10**(-7)

        self.fi_max = 0.85 # for MRF 132,  0.74 in theory
        self.r_min = (2+self.fi)/3
        self.r_max = (2+self.fi_max)/3
        
        self.E_mag = 0

        self.epsilon = self.radius*((self.fi_max/self.fi)**(1/3) - 1)
        self.viscosity_oil = 0.06
        self.viscosity_mrf = 0.112 # for LORD MRF 132
        self.nb_particles = self.fi*self.d*self.g*self.len/(4/3*math.pi*self.radius**3)
        
        self.d_eff_disp = d_eff_disp

        self.t_0_agglo = t_0_agglo # 0.0 if else
        self.t_0_disp = 0.0

        self.mu_min = self.fi/(1-self.r_min) + 1
        self.mu_max = self.fi/(1-self.r_max) + 1

        self.b = b
        self.c = 2.5 # yield stress pressure drop coefficient

        # parameters during time
        self.i = i
        self.v = np.zeros(self.nb_step)
        self.H = self.N_e * self.i / self.g

        self.dP = dP
        self.Q = np.zeros(self.nb_step)
        self.tau = np.zeros(self.nb_step)

        self.r = np.zeros(self.nb_step)
        self.r_goal = np.zeros(self.nb_step)
        self.r_goal_dP = np.zeros(self.nb_step)

        self.mu = np.zeros(self.nb_step)
        self.perm = np.zeros(self.nb_step)
        self.L = np.zeros(self.nb_step)
        
        self.R_fluidic = 6*self.viscosity_mrf*(2*self.len)/(self.g**3*(0.004415+self.g)*math.pi)

# ============================================================================
# SPECIFIC TO MODEL 14.12.2021
# ============================================================================
    def get_induced_voltage(self, L_0, L_1, i_0, i_1):
        return (L_0*i_0 - L_1*i_1) / self.step_t

    def get_H(self, i_0):
        return self.N_e * i_0 / self.g

    def get_tau(self, H):
        """Calculate the differential pressure drop"""
        # for LORD 140
        # 50kPa for 100kA/m
        # min 3kPa, diff 60 kPa
        
        #yield_stress = 3_000 + 1.2*(60_000 - 60_000*math.exp(-H/120_000))
        #return yield_stress * self.c * self.len / self.g
    
        # for LORD 132
        # 50kPa for 100kA/m
        # min 3kPa, diff 60 kPa
        yield_stress = 0.66*(60_000 - 60_000*math.exp(-H/120_000))
        return yield_stress * self.c * self.len / self.g

    def get_mu(self, r):
        return self.fi/(1-r) + 1

    def get_r(self, H):
        return (1/3)*(self.fi_max + 2 - (self.fi_max-self.fi)*math.exp(-H/self.H0))

    def get_r_dP(self, H, dP, tau):
        if dP > tau:
            return self.r_min
        k = self.b*dP**2/tau**2
        return 1 - self.fi / (self.get_mu(self.get_r(H)) - k*(self.get_mu(self.get_r(H)) - self.mu_min) - 1)

    def get_r_t(self, r_goal, r_1, t_0):
        return r_1 + self.step_t*(r_goal - r_1)/t_0

    def get_perm(self, mu, S, L):
        """Calculate the magnetic permitivity."""
        return mu * self.mu_0 * S / L

    def get_L(self, perm):
        return self.N_s**2 * perm

    def get_t_0_agglo(self, H):
        diff_E_mag = H**2*self.g**2/2*(self.get_perm(self.mu_max, self.len*self.d, self.g) -
                                       self.get_perm(self.mu_min, self.len*self.d, self.g))
        self.E_mag = diff_E_mag
        return 6*math.pi*self.epsilon**2*self.viscosity_oil*self.radius*self.nb_particles/diff_E_mag
    
    def get_t_0_disp(self, dP, tau):
        return self.d_eff_disp * self.R_fluidic * self.g * self.d / (dP - tau)

    def run(self):
        # Initial calculations
        self.tau[0] = self.get_tau(self.H[0])
        
        if self.t_0_agglo == 0.0:
            self.t_0_agglo = self.get_t_0_agglo(self.H[0])
        if self.tau[0] < self.dP[0]:
            self.t_0_disp = self.get_t_0_disp(self.dP[0], self.tau[0])
        
        self.r_goal[0] = self.get_r(self.H[0])
        self.r_goal_dP[0] = self.r_goal[0]
        self.r[0] = self.r_goal[0]
        
        self.mu[0] = self.get_mu(self.r[0])
        self.perm[0] = self.get_perm(self.mu[0], self.len*self.d, self.g)
        self.L[0] = self.get_L(self.perm[0])

        # step by step calculation
        for step in range(1, self.nb_step):
            self.tau[step] = self.get_tau(self.H[step])
            
            if self.tau[step] < self.dP[step]:
                self.t_0_disp = self.get_t_0_disp(self.dP[step], self.tau[step])
            
            self.r_goal[step] = self.get_r(self.H[step])
            self.r_goal_dP[step] = self.get_r_dP(self.H[step], self.dP[step], self.tau[step])
            
            if self.tau[step] < self.dP[step]:
                self.r[step] = self.get_r_t(self.r_goal_dP[step], self.r[step-1], self.t_0_disp)
            else:
                self.r[step] = self.get_r_t(self.r_goal_dP[step], self.r[step-1], self.t_0_agglo)
            
            self.mu[step] = self.get_mu(self.r[step])
            self.perm[step] = self.get_perm(self.mu[step], self.len*self.d, self.g)
            self.L[step] = self.get_L(self.perm[step])
            # i is firstly considered constant
            self.v[step] = self.get_induced_voltage(self.L[step-1], self.L[step], self.i[step-1], self.i[step])
