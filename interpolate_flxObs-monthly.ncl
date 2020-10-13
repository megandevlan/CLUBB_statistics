;  Date: 9 Sep 2020 
;  Purpose: almost identical to previous script (interpolate_flxObs.ncl), but accounts for time component in monthly LH obs. 


begin

; ----------------------------------
; Set file names of data to be read in and file to save things out as
fileName_ObsIn = "/Users/mdfowler/Documents/Analysis/CLUBB_initial/data/obs/sh_0.5x0.5.nc"
fileName_out = "/Users/mdfowler/Documents/Analysis/CLUBB_initial/data/processed_SHFLXmonthly_RegriddedObs.nc" 

; Set file name with grid I actually want 
fileName_GridIn = "/Users/mdfowler/Documents/Analysis/CLUBB_initial/data/f.e20.FHIST.f09_f09.cesm2_1.001.cam.h0.1951-59_PS.nc"

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
timeObs = fObs->time
SHobs  = fObs->sh

; Need to convert obs lon to be 0-360, not -180 to 180 
SHobs = lonFlip(SHobs)
;lonObs = lonFlip(lonObs)
lonObs1 = fspan(0.25,359.75,720)   ; Create evenly increasing lon array corresponding to re-ordered obs

; -------------------------------------
; Interpolate observations of flux to model grid using NCL
regridSH = linint2(lonObs1, latObs, SHobs, True, lonModel, latModel, 0) 

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
fAtt@source_file   =  "sh_0.5x0.5.nc"
fAtt@Conventions   = "None"
fAtt@creation_date = systemfunc ("date")
fileattdef( ncdf, fAtt )            ; copy file attributes

;===================================================================
; output variables directly; NCL will call appropriate functions
; to write the meta data associated with each variable
;===================================================================

regridSH!0 = "time"
regridSH!1 = "lat"
regridSH!2 = "lon"

regridSH&time = timeObs
regridSH&lat  = latModel
regridSH&lon  = lonModel

regridSH@long_name = "SH from GBAF regridded to match model grid." 
regridSH@units = "W/m2" 
ncdf->SH_obs = regridSH 


end
