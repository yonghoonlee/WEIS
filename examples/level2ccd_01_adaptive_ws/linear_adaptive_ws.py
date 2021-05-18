# Yong Hoon Lee, University of Illinois at Urbana-Champaign

import os, platform, shutil, pickle, glob, yaml
import numpy as np
import control
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import weis
import weis.control.mbc.mbc3 as mbc
from weis.aeroelasticse.LinearFAST import LinearFAST
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from weis.aeroelasticse.Util.FileTools import save_yaml, load_yaml
from pCrunch import Processing

class FASTmodel(LinearFAST):
    
    SOL_FLG_LINEAR = 1
    SOL_FLG_STEADY = 2
    
    def __init__(self, **kwargs):
        self.casename = ''
        self.outfiles = []
        self.WEIS_root = os.path.dirname(os.path.dirname(os.path.abspath(weis.__file__)))
        self.FAST_steadyDirectory = None
        self.FAST_linearDirectory = None
        self.path2dll = None
        self.case_inputs = {}
        self.DT = 0.025
        self.OutFileFmt = 2
        self.DT_Out = 'default'
        self.CompMooring = 0
        self.TrimCase = []
        self.TrimGain = 4e-5
        self.TrimTol = 1e-5
        self.VS_RtGnSp = None
        self.VS_RtTq = None
        self.VS_Rgn2K = None
        self.VS_SlPc = None
        
        
        super(FASTmodel, self).__init__()
        # LinearFAST class attributes and methods:
        #   CompMooring, DOFs,FAST_InputFile,FAST_directory,FAST_exe,
        #   FAST_runDirectory,FAST_ver,FAST_yamlfile_in,FAST_yamlfile_out,
        #   GBRatio,HydroStates,NLinTimes,TMax,WindSpeeds,case_list,
        #   case_name_list,channels,cores,debug_level,dev_branch,fst_vt,
        #   overwrite,overwrite_outfiles,parallel,post,
        #   postFAST_steady,read_yaml,v_rated,write_yaml,runFAST_linear,
        #   runFAST_steady,run_mpi,run_multi,run_serial
        
        for (k, w) in kwargs.items(): # key word arguments to attributes
            try:
                setattr(self, k, w)
            except:
                pass
        
        self.set_DLLpath() # Obtain platform and set controller DLL path
        
    def runFAST_steady(self):
        # Steady state computation
        if self.parallel:
            self.run_multi(self.cores)
        else:
            self.run_serial()
            
        # Output files
        outfiles = []
        for cname in self.case_name_list:
            if os.path.isfile(self.FAST_runDirectory + os.sep + cname + '.outb'):
                outfiles.append(self.FAST_runDirectory + os.sep + cname + '.outb')
            elif os.path.isfile(self.FAST_runDirectory + os.sep + cname + '.out'):
                outfiles.append(self.FAST_runDirectory + os.sep + cname + '.out')
            else:
                # outfiles.append(FileNotFoundError)
                print('FILE NOT FOUND: ' + self.FAST_runDirectory + os.sep + cname + '.outb')
        self.outfiles = outfiles
        
        # Post processing steady state results
        self.postFAST_steady()
    
    def postFAST_steady(self):
        fp = Processing.FAST_Processing()
        fp.OpenFAST_outfile_list = self.outfiles
        fp.t0 = self.TMax - min(max(min(self.TMax/4.0, 400.0), 200.0), self.TMax)
        fp.parallel_analysis = self.parallel
        fp.parallel_cores = self.cores
        fp.results_dir = os.path.join(self.FAST_runDirectory, 'stats')
        if self.debug_level == 0:
            fp.verbose = False
        else:
            fp.verbose = True
        fp.save_LoadRanking = True
        fp.save_SummaryStats = True
        
        # Load and save statistics and load rankings
        stats, _ = fp.batch_processing()
        if hasattr(stats, '__len__'):
            stats = stats[0]
            
        windSortInd = np.argsort(stats['Wind1VelX']['mean'])
        ssChannels = [['Wind1VelX', 'Wind1VelX'],  
                      ['OoPDefl1',  'OoPDefl'],
                      ['IPDefl1',   'IPDefl'],
                      ['BldPitch1', 'BlPitch1'],
                      ['RotSpeed',  'RotSpeed'],
                      ['TTDspFA',   'TTDspFA'],
                      ['TTDspSS',   'TTDspSS'],
                      ['PtfmSurge', 'PtfmSurge'],
                      ['PtfmSway',  'PtfmSway'],
                      ['PtfmHeave', 'PtfmHeave'],
                      ['PtfmRoll',  'PtfmRoll'],
                      ['PtfmYaw',   'PtfmYaw'],
                      ['PtfmPitch', 'PtfmPitch']]
        ssChanData = {}
        for iChan in ssChannels:
            try:
                ssChanData[iChan[1]] = np.array(stats[iChan[0]]['mean'])[windSortInd].tolist()
            except:
                print('Warning: ' + iChan[0] + ' is is not in OutList')
        
        save_yaml(self.FAST_runDirectory, self.casename + '_steady_ops.yaml', ssChanData)
        
    
    def runFAST_linear(self):
        # Linearization
        if self.parallel:
            self.run_multi(self.cores)
        else:
            self.run_serial()
    
    def run_mpi(*args, **kwargs):
        print('run_mpi method will not be executed.')
        
    def set_GBRatio(self):
        fstR = InputReader_OpenFAST(FAST_ver=self.FAST_ver, dev_branch=self.dev_branch)
        fstR.FAST_InputFile = self.FAST_InputFile
        fstR.FAST_directory = self.FAST_directory
        fstR.execute()
        self.GBRatio = fstR.fst_vt['ElastoDyn']['GBRatio']
        
    def set_DLLpath(self):
        if platform.system() == 'Windows':
            self.path2dll = os.path.join(self.WEIS_root, 'local/lib/libdiscon.dll')
        elif platform.system() == 'Darwin':
            self.path2dll = os.path.join(self.WEIS_root, 'local/lib/libdiscon.dylib')
        else:
            self.path2dll = os.path.join(self.WEIS_root, 'local/lib/libdiscon.so')
    
    class plant_design():
        modulename = ''
        deckname = ''
        value = None
    
    def prepare_case_inputs(self, SolutionStage, plantdesign_list):
        if SolutionStage == self.SOL_FLG_STEADY:
            self.FAST_runDirectory = self.FAST_steadyDirectory
            namebase = self.casename+'_steady'
        elif SolutionStage == self.SOL_FLG_LINEAR:
            self.FAST_runDirectory = self.FAST_linearDirectory
            namebase = self.casename+'_linear'
            
        case_inputs = {}
        if SolutionStage == self.SOL_FLG_STEADY:
            case_inputs[("Fst","TMax")]             = {'vals':[self.TMax], 'group':0}
        elif SolutionStage == self.SOL_FLG_LINEAR:
            case_inputs[("Fst","TMax")]             = {'vals':[8.0*self.TMax], 'group':0}
        case_inputs[("Fst","DT")]                   = {'vals':[self.DT], 'group':0}
        case_inputs[("Fst","OutFileFmt")]           = {'vals':[self.OutFileFmt], 'group':0} # 1=Text, 2=Binary, 3=Both
        case_inputs[("Fst","DT_Out")]               = {'vals':[self.DT_Out], 'group':0}
        case_inputs[("ServoDyn","DLL_FileName")]    = {'vals':[self.path2dll], 'group':0}
        case_inputs[("InflowWind","WindType")]      = {'vals':[1], 'group':0}
        
        if type(self.WindSpeeds) == np.ndarray:
            self.WindSpeeds = self.WindSpeeds.tolist()
        elif type(self.WindSpeeds) in [int, float]:
            self.WindSpeeds = [self.WindSpeeds]
        if not (type(self.WindSpeeds[0]) == float):
            self.WindSpeeds = [float(ws) for ws in self.WindSpeeds]
        case_inputs[("InflowWind","HWindSpeed")]    = {'vals':self.WindSpeeds, 'group':1}
        
        if SolutionStage == self.SOL_FLG_STEADY:
            case_inputs[("Fst","Linearize")]        = {'vals':['False'], 'group':0}
        elif SolutionStage == self.SOL_FLG_LINEAR:
            case_inputs[("Fst","Linearize")]        = {'vals':['True'], 'group':0}
            case_inputs[("Fst","NLinTimes")]        = {'vals':[self.NLinTimes], 'group':0} # Azimuth angles
            case_inputs[("Fst","CalcSteady")]       = {'vals':['True'], 'group':0}
            case_inputs[("Fst","TrimCase")]         = {'vals':self.TrimCase, 'group':1}
            case_inputs[("Fst","TrimGain")]         = {'vals':[self.TrimGain], 'group':0}
            case_inputs[("Fst","TrimTol")]         = {'vals':[self.TrimTol], 'group':0}
            case_inputs[("Fst","CompMooring")]      = {'vals':[self.CompMooring], 'group':0} # 0=None, 1=MAP++, 2=FEAMooring, 3=MoorDyn, 4=OrcaFlex
            if not self.HydroStates:
                case_inputs[("Fst","CompHydro")]    = {'vals':[0], 'group':0}
            case_inputs[("AeroDyn15","AFAeroMod")]  = {'vals':[1], 'group':0}
            case_inputs[("ServoDyn","PCMode")]      = {'vals':[0], 'group':0}
            case_inputs[("ServoDyn","VSContrl")]    = {'vals':[1], 'group':0}
            
            if (type(self.VS_RtGnSp) == float) and (type(self.VS_RtTq) == float) and (type(self.VS_Rgn2K) == float) and (type(self.VS_SlPc) == float):
                case_inputs[("ServoDyn","VS_RtGnSp")] = {'vals':[self.VS_RtGnSp], 'group':0}
                case_inputs[("ServoDyn","VS_RtTq")] = {'vals':[self.VS_RtTq], 'group':0}
                case_inputs[("ServoDyn","VS_Rgn2K")] = {'vals':[self.VS_Rgn2K], 'group':0}
                case_inputs[("ServoDyn","VS_SlPc")] = {'vals':[self.VS_SlPc], 'group':0}
            else:
                raise ValueError('Turbine-specific torque control parameters (VS_RtGnSp, VS_RtTq, VS_Rgn2K, VS_SlPc) should be defined.')
            
            case_inputs[("HydroDyn","ExctnMod")]    = {'vals':[2], 'group':0} # 0 or 2
            case_inputs[("HydroDyn","RdtnMod")]     = {'vals':[2], 'group':0} # 0 or 2
            case_inputs[("HydroDyn","DiffQTF")]     = {'vals':[0], 'group':0} # 0
            case_inputs[("HydroDyn","WvDiffQTF")]   = {'vals':['False'], 'group':0}
            
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
            for dof in self.DOFs:
                case_inputs[("ElastoDyn",dof)] = {'vals':['True'], 'group':0}
                
            # Initial conditions
            ss_ops = load_yaml(os.path.join(self.FAST_steadyDirectory, self.casename + '_steady_ops.yaml'))
            uu = ss_ops['Wind1VelX']
            
            for ic in ss_ops:
                if ic != 'Wind1VelX':
                    case_inputs[("ElastoDyn",ic)] = {'vals':np.interp(case_inputs[("InflowWind","HWindSpeed")]['vals'],uu,ss_ops[ic]).tolist(), 'group':1}
                    
            case_inputs[('ElastoDyn','BlPitch2')] = case_inputs[('ElastoDyn','BlPitch1')]
            case_inputs[('ElastoDyn','BlPitch3')] = case_inputs[('ElastoDyn','BlPitch1')]
        
        # Plant design parameters
        for pd in plantdesign_list:
            modulename = pd.modulename
            deckname = pd.deckname
            value = pd.value
            if type(value) != list:
                if type(value) == np.ndarray:
                    value = value.tolist()
                else:
                    value = [value]
            case_inputs[(modulename, deckname)] = {'vals':value, 'group':0}
        
        # Channels
        channels = {}
        if SolutionStage == self.SOL_FLG_STEADY:
            for var in ["TipDxc1",   "TipDyc1",   "TipDzc1",   "TipDxb1",   "TipDyb1",                \
                        "TipDxc2",   "TipDyc2",   "TipDzc2",   "TipDxb2",   "TipDyb2",                \
                        "TipDxc3",   "TipDyc3",   "TipDzc3",   "TipDxb3",   "TipDyb3",                \
                        "RootMxc1",  "RootMyc1",  "RootMzc1",  "RootMxb1",  "RootMyb1",               \
                        "RootMxc2",  "RootMyc2",  "RootMzc2",  "RootMxb2",  "RootMyb2",               \
                        "RootMxc3",  "RootMyc3",  "RootMzc3",  "RootMxb3",  "RootMyb3",               \
                        "TwrBsMxt",  "TwrBsMyt",  "TwrBsMzt",                                         \
                        "GenPwr",    "GenTq",     "RotThrust", "RtAeroCp",  "RtAeroCt",               \
                        "RotSpeed",  "BldPitch1", "TTDspSS",   "TTDspFA",   "NacYaw",                 \
                        "Wind1VelX", "Wind1VelY", "Wind1VelZ",                                        \
                        "LSSTipMxa", "LSSTipMya", "LSSTipMza", "LSSTipMxs", "LSSTipMys", "LSSTipMzs", \
                        "LSShftFys", "LSShftFzs", "TipRDxr",   "TipRDyr",   "TipRDzr"]:
                channels[var] = True
        elif SolutionStage == self.SOL_FLG_LINEAR:
            for var in ["BldPitch1", "BldPitch2", "BldPitch3",                                        \
                        "IPDefl1",   "IPDefl2",   "IPDefl3",   "OoPDefl1",  "OoPDefl2",  "OoPDefl3",  \
                        "NcIMURAxs", "TipDxc1",   "TipDyc1",                                          \
                        "Spn2MLxb1", "Spn2MLxb2", "Spn2MLxb3", "Spn2MLyb1", "Spn2MLyb2", "Spn2MLyb3"  \
                        "TipDzc1",   "TipDxb1",   "TipDyb1",                                          \
                        "TipDxc2",   "TipDyc2",   "TipDzc2",   "TipDxb2",   "TipDyb2",                \
                        "TipDxc3",   "TipDyc3",   "TipDzc3",   "TipDxb3",   "TipDyb3",                \
                        "RootMxc1",  "RootMyc1",  "RootMzc1",  "RootMxb1",  "RootMyb1",               \
                        "RootMxc2",  "RootMyc2",  "RootMzc2",  "RootMxb2",  "RootMyb2",               \
                        "RootMxc3",  "RootMyc3",  "RootMzc3",  "RootMxb3",  "RootMyb3",               \
                        "TwrBsMxt",  "TwrBsMyt",  "TwrBsMzt",                                         \
                        "GenPwr",    "GenTq",     "RotThrust", "RtAeroCp",  "RtAeroCt",  "RotSpeed",  \
                        "TTDspSS",   "TTDspFA",   "NacYaw",    "Wind1VelX", "Wind1VelY", "Wind1VelZ", \
                        "LSSTipMxa", "LSSTipMya", "LSSTipMza", "LSSTipMxs", "LSSTipMys", "LSSTipMzs", \
                        "LSShftFys", "LSShftFzs", "TipRDxr",   "TipRDyr",   "TipRDzr"                 \
                        "TwstDefl1","TwstDefl2","TwstDefl3"]:
                channels[var] = True
        
        # Case generation
        case_list, case_name_list = CaseGen_General(
            case_inputs, dir_matrix=self.FAST_runDirectory, namebase=namebase
        )
        
        if os.path.isfile(self.FAST_runDirectory + os.sep + 'case_matrix.txt'):
            shutil.copy(
                self.FAST_runDirectory + os.sep + 'case_matrix.txt',
                self.FAST_runDirectory + os.sep + namebase + '_case_matrix.txt'
            )
            os.remove(self.FAST_runDirectory + os.sep + 'case_matrix.txt')
            
        if os.path.isfile(self.FAST_runDirectory + os.sep + 'case_matrix.yaml'):
            shutil.copy(
                self.FAST_runDirectory + os.sep + 'case_matrix.yaml',
                self.FAST_runDirectory + os.sep + namebase + '_case_matrix.yaml'
            )
            os.remove(self.FAST_runDirectory + os.sep + 'case_matrix.yaml')
        
        # Save results
        self.case_inputs = case_inputs
        self.channels = channels
        self.case_list = case_list
        self.case_name_list = case_name_list
        
        
        
def set_IEA_UMaine(mdl, WindSpeeds, PD, Level):
    
    # OpenFAST case
    mdl.FAST_ver = 'OpenFAST'
    
    FAST_dir = 'examples/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi'
    FAST_fst = 'IEA-15-240-RWT-UMaineSemi.fst'
    mdl.FAST_directory = os.path.join(mdl.WEIS_root, FAST_dir)
    mdl.FAST_InputFile = FAST_fst
    
    case_dir = os.path.join(mdl.WEIS_root, 'examples/level2ccd_01_adaptive_ws') + os.sep
    mdl.FAST_steadyDirectory = os.path.join(
        case_dir, 'output', mdl.casename + str('_PD{:+011.4e}_L{:04d}'.format(PD, Level)), 'steady')
    mdl.FAST_linearDirectory = os.path.join(
        case_dir, 'output', mdl.casename + str('_PD{:+011.4e}_L{:04d}'.format(PD, Level)), 'linear')
    os.makedirs(mdl.FAST_steadyDirectory, exist_ok=True)
    os.makedirs(mdl.FAST_linearDirectory, exist_ok=True)
    
    mdl.CompMooring = 0 # 0=None, 1=MAP++, 2=FEAMooring, 3=MoorDyn, 4=OrcaFlex
    mdl.DOFs = ['GenDOF','TwFADOF1','PtfmPDOF']
    mdl.set_GBRatio() # Obtain and set gear box ratio
    mdl.HydroStates = True
    mdl.NLinTimes = 12
    mdl.TMax = 1000.0 #2000.0
    mdl.v_rated = 10.74
    mdl.WindSpeeds = WindSpeeds
    mdl.FAST_yamlfile_in = ''
    mdl.read_yaml = False
    mdl.FAST_yamlfile_out = ''
    mdl.write_yaml = False
    
    mdl.DT = 0.025
    mdl.DT_Out = 'default'
    mdl.OutFileFmt = 2 # 1=Text, 2=Binary, 3=Both
    mdl.TrimGain = 4e-5
    mdl.TrimTol = 1e-5
    mdl.VS_RtGnSp = 7.56*0.95 # Track 95% of rated generator speed to improve convergence near rated condition
    mdl.VS_RtTq = 19.62e6
    mdl.VS_Rgn2K = 3.7e5
    mdl.VS_SlPc = 10.0
    
    # Trim case: torque for below-rated wind, pitch for over-rated wind
    TrimCase = 3*np.ones(len(mdl.WindSpeeds), dtype=int)
    TrimCase[np.array(mdl.WindSpeeds) < mdl.v_rated] = 2
    mdl.TrimCase = TrimCase.tolist()
    
    return mdl

class linear_model():
    def __init__(self):
        self.prefix = ''
        self.wind_speed = 0.0
        self.lin_files = []
        self.num_state = 0
        self.num_control = 0
        self.num_output = 0
        self.desc_state = []
        self.desc_control = []
        self.desc_output = []
        self.SS = None
        self.SSr = None
        self.Hinf = 0.0
        self.Hinf_history = []
        self.xop = None
        self.xdop = None
        self.u_op = None
        self.y_op = None
        self.refine = False

if __name__ == '__main__':
    
    Cores = 120
    PtfmFactor = 1.0
    Hinf_criteria = 1.0
    Max_iteration = 10
    Level = 0
    WindSpeeds = np.linspace(3.0, 25.0, 23).tolist()
    FullWindSpeeds = []
    FullLinearModels = []
    
    fg = plt.figure()
    
    for iter in range(0,Max_iteration):
        
        if len(WindSpeeds) == 0:
            break
    
        mdl = FASTmodel()
        mdl.casename = 'lin_aws'
        mdl = set_IEA_UMaine(mdl, WindSpeeds=WindSpeeds, PD=PtfmFactor, Level=Level)
        mdl.parallel = True
        mdl.cores = Cores
        mdl.debug_level = 2
        pkl_filename = os.path.dirname(mdl.FAST_steadyDirectory) + '_mdl.pkl'
        
        if os.path.isfile(pkl_filename):
            WEIS_root = mdl.WEIS_root
            FAST_InputFile = mdl.FAST_InputFile
            FAST_directory = mdl.FAST_directory
            FAST_exe = mdl.FAST_exe
            FAST_linearDirectory = mdl.FAST_linearDirectory
            FAST_steadyDirectory = mdl.FAST_steadyDirectory
            FAST_yamlfile_in = mdl.FAST_yamlfile_in
            FAST_yamlfile_out = mdl.FAST_yamlfile_out
            path2dll = mdl.path2dll
            with open(pkl_filename, 'rb') as pkl:
                mdl = pickle.load(pkl)
            mdl.WEIS_root = WEIS_root
            mdl.FAST_InputFile = FAST_InputFile
            mdl.FAST_directory = FAST_directory
            mdl.FAST_exe = FAST_exe
            mdl.FAST_linearDirectory = FAST_linearDirectory
            mdl.FAST_steadyDirectory = FAST_steadyDirectory
            mdl.FAST_yamlfile_in = FAST_yamlfile_in
            mdl.FAST_yamlfile_out = FAST_yamlfile_out
            mdl.path2dll = path2dll
            WindSpeeds = mdl.WindSpeeds
        else:
            # Prepare plant design
            plantdesign_list = []
            
            pd = mdl.plant_design()
            pd.modulename = 'ElastoDyn'
            pd.deckname = 'PtfmMass'
            pd.value = 1.7838E+07*PtfmFactor # mass
            plantdesign_list.append(pd)
            
            pd = mdl.plant_design()
            pd.modulename = 'ElastoDyn'
            pd.deckname = 'PtfmRIner'
            pd.value = 1.2507E+10*PtfmFactor # Ixx = mr^2
            plantdesign_list.append(pd)
            
            pd = mdl.plant_design()
            pd.modulename = 'ElastoDyn'
            pd.deckname = 'PtfmPIner'
            pd.value = 1.2507E+10*PtfmFactor # Iyy = mr^2
            plantdesign_list.append(pd)
            
            pd = mdl.plant_design()
            pd.modulename = 'ElastoDyn'
            pd.deckname = 'PtfmYIner'
            pd.value = 2.3667E+10*PtfmFactor # Izz = mr^2
            plantdesign_list.append(pd)
            
            # Steady state solution
            mdl.prepare_case_inputs(mdl.SOL_FLG_STEADY, plantdesign_list)
            mdl.runFAST_steady()
            
            # Linearization
            mdl.prepare_case_inputs(mdl.SOL_FLG_LINEAR, plantdesign_list)
            mdl.runFAST_linear()
            
            # Save
            with open(pkl_filename, 'wb') as pkl:
                pickle.dump(mdl, pkl)
            
        case_list = mdl.case_list
        case_name_list = mdl.case_name_list
        linear_model_list = []
        
        for case_name in case_name_list:
            LM = linear_model() # Create linear model object
            LM.prefix = os.path.join(mdl.FAST_linearDirectory, case_name) # Set full path to case filename as prefix
            LM.lin_files = glob.glob(LM.prefix + '.*.lin') # Find linear model files
            MBC, matData, FAST_linData = mbc.fx_mbc3(LM.lin_files) # MBC3 transformation
            
            A = MBC['AvgA']
            B = MBC['AvgB']
            C = MBC['AvgC']
            D = MBC['AvgD']
            
            num_state = matData['NumStates']
            num_control = matData['NumInputs']
            num_output = matData['NumOutputs']
            desc_state = matData['DescStates']
            desc_control = matData['DescCntrlInpt']
            desc_output = matData['DescOutput']
            
            wind_speed = np.mean(matData['WindSpeed'])
            xop = np.mean(matData['xop'],axis=1)
            xdop = np.mean(matData['xdop'],axis=1)
            u_op = np.mean(matData['u_op'],axis=1)
            y_op = np.mean(matData['y_op'],axis=1)
            
            # Azimuth angle variable removal
            str_removal = 'ED Variable speed generator DOF (internal DOF index = DOF_GeAz), rad'
            idx_removal = desc_state.index(str_removal)
            A = np.delete(A, idx_removal, axis=0)
            A = np.delete(A, idx_removal, axis=1)
            B = np.delete(B, idx_removal, axis=0)
            C = np.delete(C, idx_removal, axis=1)
            desc_state.remove(str_removal)
            
            # Reduced SS for blade pitch and gen speed determination
            str_remain_out = 'ED GenSpeed, (rpm)'
            idx_remain_out = desc_output.index(str_remain_out)
            str_remain_ctr = 'ED Extended input: collective blade-pitch command, rad'
            idx_remain_ctr = desc_control.index(str_remain_ctr)
            Ar = A
            Br = B[:,idx_remain_ctr].reshape(B.shape[0],1)
            Cr = C[idx_remain_out,:].reshape(1,C.shape[1])
            Dr = np.array(D[idx_remain_out,idx_remain_ctr])
            
            LM.SS = control.ss(A, B, C, D)
            LM.SSr = control.ss(Ar, Br, Cr, Dr)
            LM.num_state = num_state
            LM.num_control = num_control
            LM.num_output = num_output
            LM.desc_state = desc_state
            LM.desc_control = desc_control
            LM.desc_output = desc_output
            LM.wind_speed = wind_speed
            LM.xop = xop
            LM.xdop = xdop
            LM.u_op = u_op
            LM.y_op = y_op
            LM.refine = False
            
            linear_model_list.append(LM)
            
        FullWindSpeeds += WindSpeeds
        FullLinearModels += linear_model_list
        sortedIdx = np.argsort(np.array(FullWindSpeeds))
        FullWindSpeeds = [FullWindSpeeds[idx] for idx in sortedIdx]
        FullLinearModels = [FullLinearModels[idx] for idx in sortedIdx]

        # Initialize refine flag to False
        for idx in range(0, len(FullLinearModels)):
            FullLinearModels[idx].refine = False
        
        # Calculate Hinf for full LMs
        for idx in range(1,len(FullLinearModels)-1):
            LM_L = FullLinearModels[idx-1]
            LM_C = FullLinearModels[idx]
            LM_R = FullLinearModels[idx+1]
            interp_wind_speed = np.array([LM_L.wind_speed, LM_R.wind_speed])
            
            interp_Ar = np.empty(shape=(2, LM_C.SSr.A.shape[0], LM_C.SSr.A.shape[1]))
            interp_Ar[0,:,:] = LM_L.SSr.A
            interp_Ar[1,:,:] = LM_R.SSr.A
            interp_Am = interp1d(interp_wind_speed, interp_Ar, axis=0)
            Ar_interp = interp_Am(LM_C.wind_speed)
            
            interp_Br = np.empty(shape=(2, LM_C.SSr.B.shape[0], LM_C.SSr.B.shape[1]))
            interp_Br[0,:,:] = LM_L.SSr.B
            interp_Br[1,:,:] = LM_R.SSr.B
            interp_Bm = interp1d(interp_wind_speed, interp_Br, axis=0)
            Br_interp = interp_Bm(LM_C.wind_speed)
            
            interp_Cr = np.empty(shape=(2, LM_C.SSr.C.shape[0], LM_C.SSr.C.shape[1]))
            interp_Cr[0,:,:] = LM_L.SSr.C
            interp_Cr[1,:,:] = LM_R.SSr.C
            interp_Cm = interp1d(interp_wind_speed, interp_Cr, axis=0)
            Cr_interp = interp_Cm(LM_C.wind_speed)
            
            interp_Dr = np.empty(shape=(2, LM_C.SSr.D.shape[0], LM_C.SSr.D.shape[1]))
            interp_Dr[0,:,:] = LM_L.SSr.D
            interp_Dr[1,:,:] = LM_R.SSr.D
            interp_Dm = interp1d(interp_wind_speed, interp_Dr, axis=0)
            Dr_interp = interp_Dm(LM_C.wind_speed)
            
            # Create SS model of interpolated A, B, C, D matrices
            SSr_interp = control.ss(Ar_interp, Br_interp, Cr_interp, Dr_interp)
            
            # Frequency response of actual and interpolated model
            omega = np.logspace(-3,1,10001)
            FR_actual_mag, FR_actual_phase, FR_actual_omega = LM_C.SSr.freqresp(omega)
            FR_actual_complex = \
                    np.squeeze(FR_actual_mag)*np.cos(FR_actual_phase) + \
                    1j*np.squeeze(FR_actual_mag)*np.sin(FR_actual_phase)
            FR_interp_mag, FR_interp_phase, FR_interp_omega = SSr_interp.freqresp(omega)
            FR_interp_complex = \
                    np.squeeze(FR_interp_mag)*np.cos(FR_interp_phase) + \
                    1j*np.squeeze(FR_interp_mag)*np.sin(FR_interp_phase)

            # Frequency response difference
            FR_diff = FR_actual_complex - FR_interp_complex
            FullLinearModels[idx].Hinf = np.nanmax(np.abs(FR_diff))
            FullLinearModels[idx].Hinf_history.append(FullLinearModels[idx].Hinf)
            if FullLinearModels[idx].Hinf > Hinf_criteria:
                FullLinearModels[idx].refine = True
                FullLinearModels[idx+1].refine = True
        
        # Save Hinf info
        hinf_filename = os.path.dirname(mdl.FAST_steadyDirectory) + '_Hinf.yaml'
        tmp1 = [float(LM.wind_speed) for LM in FullLinearModels]
        tmp2 = [float(LM.Hinf) for LM in FullLinearModels]
        tmp3 = [bool(LM.refine) for LM in FullLinearModels]
        with open(hinf_filename, 'w') as yml:
            yaml.dump({'WindSpeeds':tmp1, 'Hinf':tmp2, 'refine':tmp3}, yml)

        # Refine wind speeds
        WindSpeeds = []
        for idx in range(1,len(FullLinearModels)):
            if FullLinearModels[idx].refine:
                wind_speed_L = FullLinearModels[idx-1].wind_speed
                wind_speed_R = FullLinearModels[idx].wind_speed
                WindSpeeds.append(wind_speed_L + 0.25*(wind_speed_R - wind_speed_L))
                WindSpeeds.append(wind_speed_L + 0.5*(wind_speed_R - wind_speed_L))
                WindSpeeds.append(wind_speed_L + 0.75*(wind_speed_R - wind_speed_L))
        
        Hinf = [LM.Hinf for LM in FullLinearModels]
        plt.plot(FullWindSpeeds,Hinf)
        plt.draw()
        
        Level += 1

    # Save LMs
    LMfilename = os.path.join(
        os.path.dirname(os.path.dirname(mdl.FAST_linearDirectory)),
        mdl.casename + '_LM.pkl'
    )
    if not os.path.isfile(LMfilename):
        with open(LMfilename, 'wb') as pkl:
            pickle.dump(FullLinearModels, pkl)
    
    
