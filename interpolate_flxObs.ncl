begin

; ----------------------------------
; Set file names of data to be read in and file to save things out as
fileName_ObsIn = "/Users/meganfowler/Documents/NCAR/Analysis/CLUBB_initial/GBAF_Benchmark.nc"
fileName_out = "/Users/meganfowler/Documents/NCAR/Analysis/CLUBB_initial/data/processed_LHFLX_RegriddedObs.nc" 

; Set file name with grid I actually want 
fileName_GridIn = "/Users/meganfowler/Documents/NCAR/Analysis/CLUBB_initial/data/f.e20.FHIST.f09_f09.cesm2_1.001.cam.h0.1951-59_PS.nc"

; Open file of observations and gridded data
fObs = addfile(fileName_ObsIn, "r")
fModel = addfile(fileName_GridIn, "r") 

; -----------------------------------
; Read in target lat and lon
latModel = fModel->lat
lonModel = fModel->lon 

; Read in obs data, lat, and lon
latObs = fObs->lat
lonObs = fObs->lon
timeInt_LH = fObs->timeint_of_hfls

; Need to convert obs lon to be 0-360, not -180 to 180 
timeInt_LH = lonFlip(timeInt_LH)
;lonObs = lonFlip(lonObs)
lonObs1 = fspan(0.25,359.75,720)   ; Create evenly increasing lon array corresponding to re-ordered obs

; -------------------------------------
; Interpolate observations of flux to model grid using NCL
regridLH = linint2(lonObs1, latObs, timeInt_LH, True, lonModel, latModel, 0) 

; ------------------------------------
; Write out to netCDF file...   
;   Using code from: https://www.ncl.ucar.edu/Applications/method_1.shtml

system("/bin/rm -f " + fileName_out)   ; remove any pre-existing file
ncdf = addfile(fileName_out,"c")  ; open output netCDF file

;===================================================================
; create global attributes of the file (optional)
;===================================================================
fAtt               = True            ; assign file attributes
fAtt@title         = "Using NCL Simple Approach to netCDF Creation for saving regridded data"
fAtt@source_file   =  "GBAF_Benchmark.nc"
fAtt@Conventions   = "None"
fAtt@creation_date = systemfunc ("date")
fileattdef( ncdf, fAtt )            ; copy file attributes

;===================================================================
; output variables directly; NCL will call appropriate functions
; to write the meta data associated with each variable
;===================================================================

regridLH!0 = "lat"
regridLH!1 = "lon"

regridLH&lat = latModel
regridLH&lon = lonModel

regridLH@long_name = "LH from GBAF_Benchmark (via ILAMB) regridded to match model grid." 
regridLH@units = "W/m2" 
ncdf->LH_obs = regridLH 


end
