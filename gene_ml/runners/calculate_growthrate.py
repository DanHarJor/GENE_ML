from ...IFS_scripts.fieldlib import fieldfile
from ...IFS_scripts.ParIO import Parameters
import os
import numpy as np

import numpy as np
import optparse as op
import sys
import matplotlib.pyplot as plt
# from .IFS_scripts.fieldlib import 
from ...IFS_scripts.get_nrg import get_nrg0
# from .IFS_scripts.ParIO import 
# from .IFS_scripts.finite_differences import 

from ...gene_ml.parsers.parser_timeseries import ParserTimeseries

parser_ts = ParserTimeseries()

def calculate_growthrate2(scanfiles_dir, suffix, calc_from_apar=False):
    calc_from_apar=calc_from_apar
    suffix = '_'+suffix
    parameters_path = os.path.join(scanfiles_dir, f'parameters{suffix}')
    nrg_path = os.path.join(scanfiles_dir, f'nrg{suffix}')

    Qes, time = parser_ts.Qes_history(nrg_path)
    
    Qes_e = Qes['1'] # Only if electrons are first in the parameters file
    # due to float precision gene drops the heat flux to order 10 after it reaches order 10^30
    # This causes a blip in the grothrate too. so we gather the times where the blip happens here.
    deltaQ = Qes_e - np.roll(Qes_e, 1)
    # plt.figure()
    # plt.semilogy(time, Qes_e)
    # plt.xlim(0,100)
    # print(Qes_e)
    # plt.title('Qes')

    # plt.figure()
    # plt.semilogy(time, deltaQ)
    # plt.title('delta Qes')
    ignore_times = time[np.argwhere(deltaQ < 1e-10)]
    # plt.vlines(ignore_times, np.min(deltaQ), np.max(deltaQ), color='red')
    
    par = Parameters()
    par.Read_Pars(parameters_path)
    pars = par.pardict
    

    field = fieldfile(os.path.join(scanfiles_dir,'field'+suffix),pars)
    field.set_time(field.tfld[-1])
    imax = np.unravel_index(np.argmax(abs(field.phi()[:,0,:])),(field.nz,field.nx))
    phi = np.empty(0,dtype='complex128')
    if pars['n_fields'] > 1:
        imaxa = np.unravel_index(np.argmax(abs(field.apar()[:,0,:])),(field.nz,field.nx))
        apar = np.empty(0,dtype='complex128')

    time = np.empty(0)
    for i in range(len(field.tfld)):
        field.set_time(field.tfld[i])
        phi = np.append(phi,field.phi()[imax[0],0,imax[1]])
        if pars['n_fields'] > 1:
            apar = np.append(apar,field.apar()[imaxa[0],0,imaxa[1]])
        time = np.append(time,field.tfld[i])
    
    if len(phi) < 2.0:
        output_zeros = True
        omega = 0.0+0.0J
    else:
        output_zeros = False
        if calc_from_apar:
            print( "Calculating omega from apar")
            if pars['n_fields'] < 2:
                NotImplemented
                #stop
            omega = np.log(apar/np.roll(apar,1))
            dt = time - np.roll(time,1)
            omega /= dt
            omega = np.delete(omega,0)
            time = np.delete(time,0)
        else:
            omega = np.log(phi/np.roll(phi,1))
            dt = time - np.roll(time,1)
            omega /= dt
            omega = np.delete(omega,0)
            time = np.delete(time,0)

    gamma = np.real(omega)
    omega = np.imag(omega)
    gam_avg = np.average(gamma)
    om_avg = np.average(omega)
    
    # plt.figure()
    # plt.title('before delete')
    # plt.vlines(ignore_times, np.min(gamma), np.max(gamma), color='red')
    # plt.plot(time, gamma)

    for t in ignore_times:
        tdif = np.abs(time-t)
        arg_t_closest = np.argmin(tdif)
        # sometimes the closes to the ignore time is not the blip, so we take the min within 5 steps away
        if arg_t_closest < 5:
            arg_delete = arg_t_closest + np.argmin(gamma[0: arg_t_closest+5])
        else:
            arg_delete = arg_t_closest - 5 + np.argmin(gamma[arg_t_closest-5: arg_t_closest+5])
        gamma = np.delete(gamma, arg_delete)
        time = np.delete(time, arg_delete)

    # plt.figure()
    # plt.title('after delete')
    # plt.vlines(ignore_times, np.min(gamma), np.max(gamma), color='red')
    # plt.plot(time, gamma)

    return gamma, time

def calculate_growthrate(field_file_path, parameters_path):
    par = Parameters()
    par.Read_Pars(parameters_path)
    pars = par.pardict
    field = fieldfile(field_file_path, pars)
    imax = np.unravel_index(np.argmax(abs(field.phi()[:,0,:])),(field.nz,field.nx))
    phi = np.empty(0,dtype='complex128')
    time = np.empty(0)
    for i in range(len(field.tfld)):
        phi = np.append(phi,field.phi()[imax[0],0,imax[1]])
        # if pars['n_fields']>1: print('N fields:', pars['n_fields'])
        time = np.append(time, field.tfld[i])
    print( "phi_0,phi_f",phi[0],phi[-1])
    print('time',time[-1])
    if len(phi)<2: raise ValueError('Daniel says: David had a clause for this in his script, I am not sure of its purpose. len(phi)<2')
    print('phi',phi)
    omega = np.log(phi/np.roll(phi,1))
    print('OMEGA', omega)
    dt = time - np.roll(time,1)
    omega /= dt
    # print( 'omega',omega)
    omega = np.delete(omega,0)
    time = np.delete(time,0)

    gamma = np.real(omega)
    omega = np.imag(omega)

    gam_avg = np.average(np.real(omega))
    om_avg = np.average(np.imag(omega))
    print( "Gamma:",gam_avg)
    print( "Omega:",om_avg)

    return np.array(gamma).astype('float'), np.array(time).astype('float')

