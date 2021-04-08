import xarray as xr
import numpy as np 
import pandas as pd
import metpy.calc as mpc
import pickle as pickle 
from metpy.units import units

# ------------- File names: -----------
#dataDir  = '/Users/mdfowler/Documents/Analysis/CLUBB_initial/data/'
dataDir = '/glade/work/mdfowler/data/temp_staging/'

yearChoice = '2009'

z3file   = dataDir+'f.e20.FHIST.f09_f09.cesm2_1.001.cam.h1.'+yearChoice+'_Z3.nc'
varsFile = dataDir+'f.e20.FHIST.f09_f09.cesm2_1.001.cam.h1.'+yearChoice+'_RiVars.nc'
#topoFile = 'fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc'
topoFile = '/glade/p/cesmdata/cseg/inputdata/atm/cam/topo/fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc'
pressFile = dataDir+'f.e20.FHIST.f09_f09.cesm2_1.001.cam.h1.'+yearChoice+'_PresLevs.nc'

fileOut   = dataDir+'GradientRichardsonNumber_'+yearChoice+'.nc'   

# ---------- Read in data and limit to lower atmospheric levels ------- 
# Open files into datasets 
z3_ds         = xr.open_dataset(z3file, decode_times=True)
z3_ds['time'] = z3_ds.indexes['time'].to_datetimeindex()

varsDS         = xr.open_dataset(varsFile, decode_times=True)
varsDS['time'] = varsDS.indexes['time'].to_datetimeindex()

pressDS         = xr.open_dataset(pressFile, decode_times=True)
pressDS['time'] = pressDS.indexes['time'].to_datetimeindex()

topo_ds         = xr.open_dataset(topoFile, decode_times=True)

# Read in a land mask at the same resolution 
#testName = '/Users/mdfowler/Documents/Analysis/CLUBB_initial/data/f.e20.FHIST.f09_f09.cesm2_1.001.clm2.h0.1989-12.nc'
testName = '/glade/p/cgd/amp/amwg/runs/f.e20.FHIST.f09_f09.cesm2_1.001/lnd/hist/f.e20.FHIST.f09_f09.cesm2_1.001.clm2.h0.1972-01.nc'
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

# ------------- Compute height above ground and potential temperature ------ 
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

# --------- Compute gradient richardson number ------
# Compute gradient richardson number
U = (varsDS.U.values) * units('m/s')
V = (varsDS.V.values) * units('m/s')

print('Computing Ri...')

Ri = mpc.gradient_richardson_number(z_agl, potTemp, U, V, 1)

# --------- Save Ri to pickle file ---------
# pickle.dump( Ri, open(fileOut, "wb") )
#ds_out = xr.Dataset( 
#            data_vars=dict(
#                       Ri=(['time','lev','lat','lon'], Ri)),
#            coords=dict(
#                      time=(['time'],varsDS.time.values),
#                      lev=(['lev'],varsDS.lev.values),
#                      lat=(['lat'],varsDS.lat.values),
#                      lon=(['lon'],varsDS.lon.values)) )
print('Creating DataArray and writing out to netCDF...')
ds_out = xr.DataArray(np.asarray(Ri), coords=[varsDS.time.values, varsDS.lev.values,varsDS.lat.values, varsDS.lon.values], dims=['time','lev','lat','lon']) 

ds_out.to_netcdf(path=fileOut,mode='w')
     


