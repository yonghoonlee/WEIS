# Yong Hoon Lee, University of Illinois at Urbana-Champaign

import os, platform, random, shutil
import numpy as np
import weis
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
    
    def prepare_case_inputs(self, SolutionStage, plantdesign_list, fid):
        if SolutionStage == self.SOL_FLG_STEADY:
            self.FAST_runDirectory = self.FAST_steadyDirectory
            namebase = self.casename+'_'+fid+'_steady'
        elif SolutionStage == self.SOL_FLG_LINEAR:
            self.FAST_runDirectory = self.FAST_linearDirectory
            namebase = self.casename+'_'+fid+'_linear'
            
        case_inputs = {}
        if SolutionStage == self.SOL_FLG_STEADY:
            case_inputs[("Fst","TMax")]             = {'vals':[self.TMax], 'group':0}
        elif SolutionStage == self.SOL_FLG_LINEAR:
            case_inputs[("Fst","TMax")]             = {'vals':[4.0*self.TMax], 'group':0}
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
        
        if os.path.isfile(mdl.FAST_runDirectory + os.sep + 'case_matrix.txt'):
            shutil.copy(
                self.FAST_runDirectory + os.sep + 'case_matrix.txt',
                self.FAST_runDirectory + os.sep + namebase + '_case_matrix.txt'
            )
            os.remove(mdl.FAST_runDirectory + os.sep + 'case_matrix.txt')
            
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
        
        
        
def set_IEA_UMaine(mdl):
    
    # OpenFAST case
    mdl.FAST_ver = 'OpenFAST'
    
    FAST_dir = 'examples/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi'
    FAST_fst = 'IEA-15-240-RWT-UMaineSemi.fst'
    mdl.FAST_directory = os.path.join(mdl.WEIS_root, FAST_dir)
    mdl.FAST_InputFile = FAST_fst
    
    case_dir = os.path.join(mdl.WEIS_root, 'examples/level2ccd') + os.sep
    mdl.FAST_steadyDirectory = os.path.join(
        case_dir, 'output02', os.path.splitext(FAST_fst)[0], 'steady')
    mdl.FAST_linearDirectory = os.path.join(
        case_dir, 'output02', os.path.splitext(FAST_fst)[0], 'linear')
    os.makedirs(mdl.FAST_steadyDirectory, exist_ok=True)
    os.makedirs(mdl.FAST_linearDirectory, exist_ok=True)
    
    mdl.CompMooring = 0 # 0=None, 1=MAP++, 2=FEAMooring, 3=MoorDyn, 4=OrcaFlex
    mdl.DOFs = ['GenDOF','TwFADOF1','PtfmPDOF']
    mdl.set_GBRatio() # Obtain and set gear box ratio
    mdl.HydroStates = True
    mdl.NLinTimes = 12
    mdl.TMax = 2000.0
    mdl.v_rated = 10.74
    cut_in = 3.0
    cut_out = 25.0
    #vrm2 = np.floor(mdl.v_rated) - 2.0
    #vrp2 = np.ceil(mdl.v_rated) + 2.0
    #mdl.WindSpeeds = np.append(
    #    np.append(
    #        np.linspace(cut_in, vrm2 - 1.0, int(np.ceil(vrm2 - cut_in))),
    #        np.linspace(vrm2, vrp2, 4*int(np.ceil(vrp2 - vrm2)) + 1)
    #    ),
    #    np.linspace(vrp2 + 1.0, cut_out, int(np.ceil(cut_out - vrp2)))
    #)
    #mdl.WindSpeeds = mdl.WindSpeeds.tolist()
    mdl.WindSpeeds \
            = np.arange(3, 8, 1.0).tolist() + np.arange(8, 9, 0.25).tolist() + np.arange(9, 12, 0.1).tolist() \
            + np.arange(12, 13, 0.25).tolist() + np.arange(13, 26, 1.0).tolist()
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


if __name__ == '__main__':
    
    #for pf in np.linspace(0.2, 1.2, 11):
    for pf in np.linspace(0.7, 1.2, 6):

        mdl = FASTmodel()
        mdl = set_IEA_UMaine(mdl)
        mdl.casename = 'pd'
        mdl.parallel = True
        mdl.cores = 56
        mdl.debug_level = 2
    
        # Prepare plant design
        plantdesign_list = []
        PtfmFactor = pf
        
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
        mdl.prepare_case_inputs(mdl.SOL_FLG_STEADY, plantdesign_list, str('PtfmMass%+09.2e' % PtfmFactor))
        mdl.runFAST_steady()
        
        # Linearization
        mdl.prepare_case_inputs(mdl.SOL_FLG_LINEAR, plantdesign_list, str('PtfmMass%+09.2e' % PtfmFactor))
        mdl.runFAST_linear()
        
        
        
