'''
This script is for running SIMION simulations in parallel.
Simulation args are retardation, mid1_ratio, mid2_ratio, ke_fwhm, numParticles, baseDir


'''

import os
import shutil
import tempfile
import numpy as np
from run_simion_edited import runSimion

import sys
sys.path.insert(0, os.path.abspath('..'))
from voltage_generator import calculateVoltage_NedlerMeade

ResultsDir = r"C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_data"
SimionDir = r"C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_simulation"
iobFileLoc = SimionDir + "//TOF_simulation.iob"
recordingFile = SimionDir + "//TOF_simulation.rec"
potArrLoc = SimionDir + "//copiedArray.PA0"
lua_path = SimionDir + "//TOF_simulation.lua"
Fly2File = SimionDir + "//TOF_simulation.fly2"F


def generate_fly2File(fileNameToWriteTo, ke_fwhm, numParticles = None):
    # Check if .fly2 file with this name already exists
    fileExists = os.path.isfile(fileNameToWriteTo)
    # Delete previous copy if it exists
    if fileExists:
        os.remove(fileNameToWriteTo)
        
    # Open up file to write to
    with open(fileNameToWriteTo, 'w') as fileOut:
        fileOut.write('particles {\n')
        fileOut.write('  coordinates = 0,\n')
        fileOut.write('  standard_beam{\n')
        fileOut.write(f'    n = {numParticles},\n')
        fileOut.write('    tob = 0,\n')
        fileOut.write('    mass = 0.000548579903,\n')
        fileOut.write('    charge = -1,\n')
        fileOut.write('    ke = gaussian_distribution {\n')
        fileOut.write('    mean = 600,\n')
        fileOut.write('    fwhm = '+str(ke_fwhm)+'\n')
        fileOut.write('    },\n')
        fileOut.write("    az =  0,\n")
        fileOut.write("    el =  0, \n")
        fileOut.write("    cwf = 1,\n")
        fileOut.write("    color = 0,\n")
        fileOut.write("    position = vector(12.2, 0, 0)\n")
        fileOut.write("  }\n")
        fileOut.write("}")
        
        
def run_simulation(simArgs):
    retardation, mid1_ratio, mid2_ratio, ke_fwhm, numParticles, baseDir = simArgs
    
    # Create a temporary directory for this simulation
    temp_dir = tempfile.mkdtemp()
    try:
        # Copy necessary files to the temporary directory
        iobFileLoc = shutil.copy(os.path.join(SimionDir, "TOF_simulation.iob"), temp_dir)
        recordingFile = shutil.copy(os.path.join(SimionDir, "TOF_simulation.rec"), temp_dir)
        lua_path = shutil.copy(os.path.join(SimionDir, "TOF_simulation.lua"), temp_dir)
        Fly2File = shutil.copy(os.path.join(SimionDir, "TOF_simulation.fly2"), temp_dir)
        
        generate_fly2File(fly2File, foat(ke_fwhm), numParticles)
        
        # Copy all PA files to the temporary directory
        for pa_file in os.listfir(SimionDir):
            if pa_file.startswith('copiedArray.PA'):
                shutil.copy(os.path.join(SimionDir, pa_file), temp_dir)
                
        potArrLoc = os.path.join(temp_dir, 'copiedArray.PA0')
        
        newVoltages, resistorValues = calculateVoltage_NedlerMeade(
            retardation,
            voltage_front=0,
            mid1_ratio=mid1_ratio
            mid2_ratio=mid2_ratio
            )
        
        mid1 = np.abs(mid1_ratio) if mid1_ratio < 0 else mid1_ratio
        m1sign = 'neg' if mid1_ratio < 0 else 'pos'
        mid2 = np.abs(mid2_ratio) if mid2_ratio < 0 else mid2_ratio
        m2sign = 'neg' if mid2_ratio < 0 else 'pos'
        r = np.abs(retardation) if retardation < 0 else retardation
        rsign = 'neg' if retardation < 0 else 'pos'
        
        outputDir = os.path.join(baseDir, f'R{np.abs(retardation)}')
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        
        simion_output_path = os.path.join(outputDir,
                                            f'sim_{rsign}_R{r}_{m1sign}_{mid1}_{m2sign}_{mid2}_{ke_fwhm}_n{numParticles}.txt')
        
        runSimion(Fly2File, newVoltages, simion_output_path, recordingFile, iobFileLoc, potArrLoc, temp_dir)

            
        
        
