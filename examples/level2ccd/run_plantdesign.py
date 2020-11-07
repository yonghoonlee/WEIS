# Yong Hoon Lee, University of Illinois at Urbana-Champaign

from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_General import CaseGen_General
import numpy as np
import os, platform

def run_plantdesign_eval():

    # Paths calling the standard modules of WEIS
    fastBatch = runFAST_pywrapper_batch(FAST_ver='OpenFAST', dev_branch=True)
    run_dir1                    = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep
    run_dir2                    = os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) + os.sep
    fastBatch.FAST_directory    = os.path.join(run_dir2, 'OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files
    fastBatch.FAST_InputFile    = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
    fastBatch.FAST_runDirectory = 'output/iea15mw-semi-lin'
    fastBatch.debug_level       = 2
    fastBatch.read_yaml         = False
    fastBatch.write_yaml        = True

    # User settings
    TMax        = 2000.0    # Length of wind grids and OpenFAST simulations [s]
    cut_in      = 3.        # Cut in wind speed [m/s]
    cut_out     = 25.       # Cut out wind speed [m/s]
    n_ws        = 23        # Number of wind speed bins [-]
    v_rated     = 10.74     # Rated wind speed for the turbine [m/s]
    wind_speeds = np.linspace(int(cut_in), int(cut_out), int(n_ws)) # Wind speeds to run OpenFAST at
    # Ttrans      = max([0., TMax - 60.])  # Start of the transient for DLC with a transient, e.g. DLC 1.4
    # TStart      = max([0., TMax - 600.]) # Start of the recording of the channels of OpenFAST

    # Degrees of freedom
    DOFs = ['GenDOF', 'TwFADOF1', 'PtfmHvDOF', 'PtfmPDOF']

    # Plant design 1: PtfmMass
    PtfmFactor  = np.linspace(0.80, 1.20, 5)
    PtfmMass    = 1.7838E+07*PtfmFactor # mass
    PtfmRIner   = 1.2507E+10*PtfmFactor # Ixx = mr^2
    PtfmPIner   = 1.2507E+10*PtfmFactor # Iyy = mr^2
    PtfmYIner   = 2.3667E+10*PtfmFactor # Izz = mr^2

    # Initial conditions for ElastoDyn
    u_ref       = np.arange(3.,26.) # Wind speed vector to specify the initial conditions
    pitch_ref   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5058525323662666, 5.253759185225932, 7.50413344606208, 9.310153958810268, 10.8972969450052, 12.412247669440042, 13.883219268525659, 15.252012626933068, 16.53735488246438, 17.76456777500061, 18.953261878035104, 20.11055307762722, 21.238680277668898, 22.30705111326602, 23.455462501156205] # Pitch values in deg
    omega_ref   = [2.019140272160114, 2.8047214918577925, 3.594541645994511, 4.359025795823625, 5.1123509774611025, 5.855691196288371, 6.589281196735111, 7.312788026081227, 7.514186181824161, 7.54665511646938, 7.573823812448151, 7.600476033113538, 7.630243938880304, 7.638301051122195, 7.622050377183605, 7.612285710588359, 7.60743945212863, 7.605865650155881, 7.605792924227456, 7.6062185247519825, 7.607153933765292, 7.613179734210654, 7.606737845170748] # Rotor speeds in rpm
    pitch_init = np.interp(wind_speeds, u_ref, pitch_ref)
    omega_init = np.interp(wind_speeds, u_ref, omega_ref)

    # Trim case: torque for below-rated wind, pitch for over-rated wind
    TrimCase = 3*np.ones(len(wind_speeds), dtype=int)
    TrimCase[np.array(wind_speeds) < v_rated] = 2

    # Find the controller
    if platform.system() == 'Windows':
        path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dll')
    elif platform.system() == 'Darwin':
        path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dylib')
    else:
        path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.so')

    # Settings passed to OpenFAST
    case_inputs = {}
    case_inputs[("Fst","TMax")]                 = {'vals':[TMax], 'group':0}
    case_inputs[("Fst","DT")]                   = {'vals':[0.01], 'group':0}
    case_inputs[("Fst","OutFileFmt")]           = {'vals':[2], 'group':0} # 1=Text, 2=Binary, 3 Both
    case_inputs[("Fst","DT_Out")]               = {'vals':['default'], 'group':0}
    case_inputs[("Fst","CompMooring")]          = {'vals':[0], 'group':0} # 0=None, 1=MAP++, 2=FEAMooring, 3=MoorDyn, 4=OrcaFlex
    case_inputs[("Fst","Linearize")]            = {'vals':['True'], 'group':0} # Generate linear files
    case_inputs[("Fst","CalcSteady")]           = {'vals':['True'], 'group':0} # Trim enabled
    case_inputs[("Fst","TrimCase")]             = {'vals':TrimCase.tolist(), 'group':1}
    case_inputs[("Fst","TrimTol")]              = {'vals':[1e-5], 'group':0}
    case_inputs[("Fst","NLinTimes")]            = {'vals':[12], 'group':0} # Azimuth angles
    case_inputs[("ElastoDyn","GenDOF")]         = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","RotSpeed")]       = {'vals': omega_init, 'group': 1}
    case_inputs[("ElastoDyn","BlPitch1")]       = {'vals': pitch_init, 'group': 1}
    case_inputs[("ElastoDyn","BlPitch2")]       = case_inputs[("ElastoDyn","BlPitch1")]
    case_inputs[("ElastoDyn","BlPitch3")]       = case_inputs[("ElastoDyn","BlPitch1")]
    case_inputs[("ElastoDyn","PtfmMass")]       = {'vals':PtfmMass.tolist(), 'group':2}
    case_inputs[("ElastoDyn","PtfmRIner")]      = {'vals':PtfmRIner.tolist(), 'group':2}
    case_inputs[("ElastoDyn","PtfmPIner")]      = {'vals':PtfmPIner.tolist(), 'group':2}
    case_inputs[("ElastoDyn","PtfmYIner")]      = {'vals':PtfmYIner.tolist(), 'group':2}
    case_inputs[("ServoDyn","PCMode")]          = {'vals':[0], 'group':0} # 0=None, 5=DLL
    case_inputs[("ServoDyn","VSContrl")]        = {'vals':[1], 'group':0} # 0=None, 1=Simple VS, 5=DLL
    case_inputs[("ServoDyn","VS_RtGnSp")]       = {'vals':[7.55998713], 'group':0}
    case_inputs[("ServoDyn","VS_RtTq")]         = {'vals':[19624046.0], 'group':0}
    case_inputs[("ServoDyn","VS_Rgn2K")]        = {'vals':[340000.000], 'group':0}
    case_inputs[("ServoDyn","VS_SlPc")]         = {'vals':[10], 'group':0}
    case_inputs[("ServoDyn","DLL_FileName")]    = {'vals':[path2dll], 'group':0}
    case_inputs[("InflowWind","WindType")]      = {'vals':[1], 'group':0}
    case_inputs[("InflowWind","HWindSpeed")]    = {'vals': wind_speeds, 'group': 1}
    case_inputs[("AeroDyn15","AFAeroMod")]      = {'vals':[1], 'group':0} # 1=Steady, 2=Unsteady (should be 1 for linearizing)
    case_inputs[("HydroDyn","ExctnMod")]        = {'vals':[0], 'group':0} # 2
    case_inputs[("HydroDyn","RdtnMod")]         = {'vals':[0], 'group':0} # 2
    case_inputs[("HydroDyn","DiffQTF")]         = {'vals':[0], 'group':0} # 0
    case_inputs[("HydroDyn","WvDiffQTF")]       = {'vals':['False'], 'group':0}

    # Degrees-of-freedom: set all to False & enable those defined in self
    case_inputs[("ElastoDyn","FlapDOF1")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF2")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","EdgeDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TeetDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","DrTrDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","GenDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","YawDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF1")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF2")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF1")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF2")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmSgDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmSwDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmHvDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmRDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmPDOF")] = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmYDOF")] = {'vals':['False'], 'group':0}
    for dof in DOFs:
        case_inputs[("ElastoDyn",dof)] = {'vals':['True'], 'group':0}

    case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=fastBatch.FAST_runDirectory, namebase='iea15mw-semi-lin')
    n_cores = min(os.cpu_count(),len(case_list)) # Number of available cores

    fastBatch.case_list = case_list
    fastBatch.case_name_list = case_name_list

    if n_cores == 1:
        fastBatch.run_serial()
    else:
        fastBatch.run_multi(cores=n_cores)

if __name__ == '__main__':
    run_plantdesign_eval()

