import xarray as xr
import numpy as np 
import pandas as pd
import metpy.calc as mpc
import pickle as pickle 
from metpy.units import units

; ------------- File names: -----------
dataDir  = '/Users/mdfowler/Documents/Analysis/CLUBB_initial/data/'

z3file   = dataDir+'daily/f.e20.FHIST.f09_f09.cesm2_1.001.cam.h1.2000_Z3.nc'
varsFile = dataDir+'daily/f.e20.FHIST.f09_f09.cesm2_1.001.cam.h1.2000_RiVars.nc'
topoFile = dataDir+'fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc'
pressFile = dataDir+'daily/f.e20.FHIST.f09_f09.cesm2_1.001.cam.h1.2000_PresLevs.nc'
fileOut   = dataDir+'daily/GradientRichardsonNumber_2000.nc'   

; ---------- Read in data and limit to lower atmospheric levels ------- 
# Open files into datasets 
z3_ds         = xr.open_dataset(z3file, decode_times=True)
z3_ds['time'] = z3_ds.indexes['time'].to_datetimeindex()

varsDS         = xr.open_dataset(varsFile, decode_times=True)
varsDS['time'] = varsDS.indexes['time'].to_datetimeindex()

pressDS         = xr.open_dataset(pressFile, decode_times=True)
pressDS['time'] = pressDS.indexes['time'].to_datetimeindex()

topo_ds         = xr.open_dataset(topoFile, decode_times=True)

# Read in a land mask at the same resolution 
testName = '/Users/mdfowler/Documents/Analysis/CLUBB_initial/data/f.e20.FHIST.f09_f09.cesm2_1.001.clm2.h0.1989-12.nc'
testDF   = xr.open_dataset(testName)

# Make land mask
landMask              = testDF.landmask.values
landMask[landMask==0] = np.nan

np.shape(landMask)

# Limit to levels with pressures >850.0 hPa 
iLev = np.where(z3_ds.lev.values > 850.0)[0]

z3_ds   = z3_ds.isel(lev=iLev)
varsDS  = varsDS.isel(lev=iLev)
pressDS = pressDS.isel(lev=iLev)

; ------------- Compute height above ground and potential temperature ------ 
# Get height (above ground level)
Z3   = z3_ds.Z3.values*landMask
PHIS = topo_ds.PHIS.values*landMask

# Convert geopotential into geopotential height of surface
PHIS = PHIS/9.81

# Force PHIS to have same dimensions as Z3 
PHISnew1 = np.repeat(PHIS[np.newaxis, :,:],np.shape(Z3)[1],axis=0)
PHISnew  = np.repeat(PHISnew1[np.newaxis,:,:,:],np.shape(Z3)[0],axis=0)
print('Shape of Z3:   ', np.shape(Z3))
print('Shape of PHIS: ', np.shape(PHISnew))

# Height agl is then just Z3 - PHISnew
z_agl = (Z3 - PHISnew) * units.meters

# Get potential temperature 
P = (pressDS.PRESSURE.values*landMask) * units('Pa')
T = (varsDS.T.values*landMask)         * units('K')

potTemp = mpc.potential_temperature(P,T)

# Clear variables from memory that aren't needed anymore
del PHISnew
del PHIS 
del Z3 
del P
del T 
del pressDS
del z3_ds
del topo_ds
del testDF

; --------- Compute gradient richardson number ------
# Compute gradient richardson number
U = (varsDS.U.values) * units('m/s')
V = (varsDS.V.values) * units('m/s')

print('Computing Ri...')

Ri = mpc.gradient_richardson_number(z_agl, potTemp, U, V, 1)

; --------- Save Ri to pickle file ---------
pickle.dump( Ri, open(fileOut, "wb") )

