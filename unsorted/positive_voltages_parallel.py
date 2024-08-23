import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import sys
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
#from run_simion import parse_and_process_data, runSimion #, generate_fly2File_lognorm
from run_simion_edited import parse_and_process_data, runSimion
#from ltspice_runner import run_spice_simulation, modify_cir_file, check_currents
sys.path.insert(0, os.path.abspath('..'))
from voltage_generator import calculateVoltage_NelderMeade


#ResultsDir = r"C:\\Users\\proxi\\Documents\\coding\\TOF_data"
ResultsDir = r"C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_data"
#SimionDir = r"C:\\Users\\proxi\\Documents\\coding\\TOF_ML_backup\\simulations\\TOF_simulation"
SimionDir = r"C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_simulation"
iobFileLoc = SimionDir + "//TOF_simulation.iob"
recordingFile = SimionDir + "//TOF_simulation.rec"
potArrLoc = SimionDir + "//copiedArray.PA0"
lua_path = SimionDir + "//TOF_simulation.lua"
Fly2File = SimionDir + "//TOF_simulation.fly2"
#ltspice_path = r"C:\\Users\\proxi\\AppData\\Local\\Programs\\ADI\\LTspice\\LTspice.exe"
#spice_sim_path = r"C:\\Users\\proxi\\Documents\\coding\\TOF_ML\\simulations\\ltspice"



def generate_fly2File2(filenameToWriteTo, ke_fwhm, numParticles):

    # Check if .fly2 file with this name already exists
    fileExists = os.path.isfile(filenameToWriteTo)

    # Delete previous copy, if there is one
    if fileExists:
        os.remove(filenameToWriteTo)

    # Open up file to write to
    with open(filenameToWriteTo, "w") as fileOut:
        fileOut.write("particles {\n")
        fileOut.write("  coordinates = 0,\n")
        fileOut.write("  standard_beam {\n")
        fileOut.write(f"    n = {numParticles},\n")
        fileOut.write("    tob = 0,\n")
        fileOut.write("    mass = 0.000548579903,\n")
        fileOut.write("    charge = -1,\n")
        fileOut.write("    ke =  gaussian_distribution {\n")
        fileOut.write("      mean = 600,\n")
        fileOut.write("      fwhm = " + str(ke_fwhm) + "\n")
        fileOut.write("    },\n")
        fileOut.write("    az =  0,\n")
        fileOut.write("    el =  0, \n")
        fileOut.write("    cwf = 1,\n")
        fileOut.write("    color = 0,\n")
        fileOut.write("    position = vector(12.2, 0, 0)\n")
        fileOut.write("  }\n")
        fileOut.write("}")


def run_simulation(args):
    retardation, mid1_ratio, mid2_ratio, ke_fwhm, numParticles, baseDir = args
    
    ResultsDir = r"C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_data"
    SimionDir = r"C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_simulation"
    iobFileLoc = SimionDir + "//TOF_simulation.iob"
    recordingFile = SimionDir + "//TOF_simulation.rec"
    potArrLoc = SimionDir + "//copiedArray.PA0"
    lua_path = SimionDir + "//TOF_simulation.lua"
    Fly2File = SimionDir + "//TOF_simulation.fly2"

    # Create a temporary directory for this simulation
    temp_dir = tempfile.mkdtemp()
    try:
        # Copy necessary files to the temporary directory
        #print("generate fly file first")
        
        iobFileLoc = shutil.copy(os.path.join(SimionDir, "TOF_simulation.iob"), temp_dir)
        recordingFile = shutil.copy(os.path.join(SimionDir, "TOF_simulation.rec"), temp_dir)
        lua_path = shutil.copy(os.path.join(SimionDir, "TOF_simulation.lua"), temp_dir)
        Fly2File = shutil.copy(os.path.join(SimionDir, "TOF_simulation.fly2"), temp_dir)
        
        generate_fly2File2(Fly2File, float(ke_fwhm), numParticles)

        # Copy all PA files to the temporary directory
        for pa_file in os.listdir(SimionDir):
            if pa_file.startswith("copiedArray.PA"):
                #print(pa_file)
                shutil.copy(os.path.join(SimionDir, pa_file), temp_dir)

        potArrLoc = os.path.join(temp_dir, "copiedArray.PA0")

        new_voltages, resistor_values = calculateVoltage_NelderMeade(retardation, voltage_front=0,
                                                                        mid1_ratio=mid1_ratio,
                                                                        mid2_ratio=mid2_ratio)
        
        
        mid1 = np.abs(mid1_ratio) if mid1_ratio < 0 else mid1_ratio
        m1sign = 'neg' if mid1_ratio < 0 else 'pos'
        mid2 = np.abs(mid2_ratio) if mid2_ratio < 0 else mid2_ratio
        m2sign = 'neg' if mid2_ratio < 0 else 'pos'
        r = np.abs(retardation) if retardation < 0 else retardation
        rsign = 'neg' if retardation < 0 else 'pos'

        output_dir = os.path.join(baseDir, f"R{np.abs(retardation)}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        simion_output_path = os.path.join(output_dir,
                                            f"sim_{rsign}_R{r}_{m1sign}_{mid1}_{m2sign}_{mid2}_{ke_fwhm}_n{numParticles}.txt")

        runSimion(Fly2File, new_voltages, simion_output_path, recordingFile, iobFileLoc, potArrLoc, temp_dir)
        print('ran Simion')
        #results_df = parse_results(simion_output_path)
        parse_and_process_data(simion_output_path)

        return simion_output_path
    
    finally:
        print('Cleaning up temp dir')
        try: 
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error in simulation: {e}")

# Don't run simulations that are already run. 
def check_existing_simulation(args):
    retardation, mid1_ratio, mid2_ratio, ke_fwhm, numParticles, baseDir = args
    if mid1_ratio < 0:
        mid1 = np.abs(mid1_ratio)
        m1sign = "neg"
    else:
        mid1 = mid1_ratio
        m1sign = "pos"
    if mid2_ratio < 0:
        mid2 = np.abs(mid2_ratio)
        m2sign = "neg"
    else:
        mid2 = mid2_ratio
        m2sign = "pos"
    if retardation < 0:
        r = np.abs(retardation)
        rsign = "neg"
    else:
        r = retardation
        rsign = "pos"

    # Construct the output path based on the structure provided
    output_dir = os.path.join(baseDir, f"R{np.abs(retardation)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simion_output_path = os.path.join(output_dir,
                                          f"sim_{rsign}_R{r}_{m1sign}_{mid1}_{m2sign}_{mid2}_{ke_fwhm}_n{numParticles}.h5")

    if os.path.exists(simion_output_path) and os.path.getsize(simion_output_path) >= 89000:
        return None
    else:
        return args


def record_execution_time(num_cores, simulation_args):
    times = []
    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_simulation, simulation_args), 1):
            if i % num_cores == 0:
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                print(f"{i} simulations completed in {elapsed_time:.2f} seconds")
    return times


def parallel_pool(num_cores, simulation_args):
    results = []
    with Pool(processes=num_cores) as pool:
        results.append(list(pool.imap_unordered(run_simulation, simulation_args)))
    return results


if __name__ == '__main__':
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{string} is not a valid path")

    parser = argparse.ArgumentParser(
        description='code for running Simion simulations for collection efficiency analysis')

    parser.add_argument(
        "--retardation",
        nargs="*",
        type=int,
        default=[-550],
    )
    parser.add_argument(
        "--ke_fwhm",
        nargs="*",
        type=float,
        default = [1] #, 3 , 6, 12, 15, 18, 24, 30, 60],
    )

    parser.add_argument(
        "--mid1_ratio",
        nargs="*",
        type=float,
        default=[0] #, 0.08, 0.11248, 0.2, 0.3, 0.4, 0.6, 0.9],
    )

    parser.add_argument(
        "--mid2_ratio",
        nargs="*",
        type=float,
        default=[0]#, 0.08, 0.1354, 0.2, 0.3, 0.4, 0.6, 0.9],
    )
    
    parser.add_argument(
        "--numParticles",
        nargs='*',
        type=float,
        default=[10000]
    )

    args = parser.parse_args()



    # Prepare list of all combinations
    simulation_args = [(retardation, mid1_ratio, mid2_ratio, ke_fwhm, numParticles, ResultsDir)
                       for retardation in args.retardation
                       for mid1_ratio in args.mid1_ratio
                       for mid2_ratio in args.mid2_ratio
                       for ke_fwhm in args.ke_fwhm
                       for numParticles in args.numParticles]
    print(f'Number of simulation args: {len(simulation_args)}')

    # Filter out existing simulations
    with Pool(processes=cpu_count()) as pool:
        simulation_args = list(filter(None, pool.map(check_existing_simulation, simulation_args)))

    # Determine the number of CPU cores to use
    num_cores = cpu_count()  # Specify the number of cores to use
    print(f'Cores: {num_cores}')

    # Record execution time for parallel execution
    results = parallel_pool(num_cores, simulation_args)
    #print(results)

