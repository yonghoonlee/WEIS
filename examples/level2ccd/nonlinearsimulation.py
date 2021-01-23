# Yong Hoon Lee, University of Illinois at Urbana-Champaign

import os, platform, random, shutil
import numpy as np
import weis
from weis.aeroelasticse.LinearFAST import LinearFAST
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from weis.aeroelasticse.Util.FileTools import save_yaml, load_yaml
from ROSCO_toolbox import utilities as ROSCO_utilities
from pCrunch import Processing

class FASTmodel(LinearFAST):
    
    SOL_FLG_LINEAR = 1
    SOL_FLG_NONLINEARSIM = 11
    
    def __init__(self, **kwargs):
        self.casename = ''
        self.outfiles = []
        self.WEIS_root = os.path.dirname(os.path.dirname(os.path.abspath(weis.__file__)))
        self.FAST_linearDirectory = None
        self.FAST_nonlinearsimDirectory = None
        self.path2dll = None
        self.case_inputs = {}
        self.DT = 0.025
        self.OutFileFmt = 2
        self.DT_Out = 'default'
        self.CompMooring = 0
        self.TrimCase = []
        self.TrimGain = 1e-4
        self.TrimTol = 1e-5
        self.VS_RtGnSp = 0.0
        self.VS_RtTq = 0.0
        self.VS_Rgn2K = 0.0
        self.VS_SlPc = 0.0
        self.BlPitch = 0.0
        self.RotSpeed = 0.0
        
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
        
    def runFAST_nonlinearsim(self):
        # Nonlinear computation
        if self.parallel:
            self.run_multi(self.cores)
        else:
            self.run_serial()
    
    def runFAST_linear(self):
        # Linearization
        if self.parallel:
            self.run_multi(self.cores)
        else:
            self.run_serial()

    def runFAST_nonlinear_lin_cond(self):
        # Nonlinear simulation
        if self.parallel:
            self.run_multi(self.cores)
        else:
            self.run_serial()
    
    def run_mpi(*args, **kwargs):
        print('run_mpi method will not be executed.')
        
    def get_fst_vt(self):
        fstR = InputReader_OpenFAST(FAST_ver=self.FAST_ver, dev_branch=self.dev_branch)
        fstR.FAST_InputFile = self.FAST_InputFile
        fstR.FAST_directory = self.FAST_directory
        fstR.execute()
        self.fst_vt = fstR.fst_vt

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
        if SolutionStage == self.SOL_FLG_LINEAR:
            self.FAST_runDirectory = self.FAST_linearDirectory
            namebase = self.casename+'_'+fid+'_linear'
        elif SolutionStage == self.SOL_FLG_NONLINEARSIM:
            self.FAST_runDirectory = self.FAST_nonlinearsimDirectory
            namebase = self.casename+'_'+fid+'_nonlinearsim'

        #if true, there will be a lot of hydronamic states, equal to num. states in ss_exct and ss_radiation models
        if any([d in ['PtfmSgDOF','PtfmSwDOF','PtfmHvDOF','PtfmRDOF','PtfmPDOF','PtfmyDOF'] for d in self.DOFs]):
            self.HydroStates = True # taking out to speed up for test
        else:
            self.HydroStates = False # taking out to speed up for test

        case_inputs = {}
        
        if SolutionStage == self.SOL_FLG_NONLINEARSIM:
            case_inputs[("Fst","TMax")]             = {'vals':[self.TMax], 'group':0}
        elif SolutionStage == self.SOL_FLG_LINEAR:
            case_inputs[("Fst","TMax")]             = {'vals':[4.0*self.TMax], 'group':0} # ensure enough time for convergence detection
        case_inputs[("Fst","DT")]                   = {'vals':[self.DT], 'group':0}
        case_inputs[("Fst","OutFileFmt")]           = {'vals':[self.OutFileFmt], 'group':0} # 1=Text, 2=Binary, 3=Both
        case_inputs[("Fst","DT_Out")]               = {'vals':[self.DT_Out], 'group':0}
        case_inputs[("ServoDyn","DLL_FileName")]    = {'vals':[self.path2dll], 'group':0}
        
        if type(self.WindSpeeds) == np.ndarray:
            self.WindSpeeds = self.WindSpeeds.tolist()
        elif type(self.WindSpeeds) in [int, float]:
            self.WindSpeeds = [self.WindSpeeds]
        if not (type(self.WindSpeeds[0]) == float):
            self.WindSpeeds = [float(ws) for ws in self.WindSpeeds]
        case_inputs[("InflowWind","WindType")]      = {'vals':[1], 'group':0}
        case_inputs[("InflowWind","HWindSpeed")]    = {'vals':self.WindSpeeds, 'group':1}
        
        if SolutionStage == self.SOL_FLG_NONLINEARSIM:

            case_dir = os.path.join(self.WEIS_root, 'examples/level2ccd') + os.sep
            fwind = os.path.join(case_dir, 'nonlinearsimulation.wnd')
            case_inputs[("InflowWind","WindType")]     = {'vals':[2], 'group':0} # Uniform wind
            case_inputs[("InflowWind","Filename_Uni")] = {'vals':[fwind], 'group':0} # Uniform wind file

            case_inputs[("Fst","Linearize")]        = {'vals':['False'], 'group':0}
            case_inputs[("Fst","CompMooring")]      = {'vals':[self.CompMooring], 'group':0}
            case_inputs[("AeroDyn15","AFAeroMod")]  = {'vals':[2], 'group':0}

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
            
        elif SolutionStage == self.SOL_FLG_LINEAR:
            case_inputs[("Fst","Linearize")]        = {'vals':['True'], 'group':0}
            case_inputs[("Fst","NLinTimes")]        = {'vals':[self.NLinTimes], 'group':0} # Azimuth angles
            case_inputs[("Fst","CalcSteady")]       = {'vals':['True'], 'group':0}
            case_inputs[("Fst","TrimCase")]         = {'vals':self.TrimCase, 'group':1}
            case_inputs[("Fst","TrimGain")]         = {'vals':[self.TrimGain], 'group':0}
            case_inputs[("Fst","TrimTol")]          = {'vals':[self.TrimTol], 'group':0}
            case_inputs[("Fst","CompMooring")]      = {'vals':[self.CompMooring], 'group':0} # 0=None, 1=MAP++, 2=FEAMooring, 3=MoorDyn, 4=OrcaFlex
            case_inputs[("Fst","CompHydro")]        = {'vals':[int(self.HydroStates)], 'group':0}
            case_inputs[("AeroDyn15","AFAeroMod")]  = {'vals':[1], 'group':0}
            case_inputs[("ServoDyn","PCMode")]      = {'vals':[0], 'group':0}
            case_inputs[("ServoDyn","VSContrl")]    = {'vals':[1], 'group':0}
            
            self.get_fst_vt() # Obtain OpenFAST input structure
            self.GBRatio = self.fst_vt['ElastoDyn']['GBRatio'] # GBRatio
            rosco_inputs = ROSCO_utilities.read_DISCON(self.fst_vt['ServoDyn']['DLL_InFile'])
            self.VS_RtGnSp = rosco_inputs['PC_RefSpd'] * 30 / np.pi * 0.5
            self.VS_RtTq = rosco_inputs['VS_RtTq']
            self.VS_Rgn2K = rosco_inputs['VS_Rgn2K']/ (30 / np.pi)**2
            self.VS_SlPc = 10.0
            self.BlPitch = rosco_inputs['PC_FinePit'] # set initial pitch to fine pitch (may be problematic at high wind speeds)
            self.RotSpeed = rosco_inputs['PC_RefSpd'] * 30 / np.pi # convert to rpm and use 95% of rated

            # Servo control parameters
            case_inputs[("ServoDyn","VS_RtGnSp")]   = {'vals':[self.VS_RtGnSp], 'group':0}
            case_inputs[("ServoDyn","VS_RtTq")]     = {'vals':[self.VS_RtTq], 'group':0}
            case_inputs[("ServoDyn","VS_Rgn2K")]    = {'vals':[self.VS_Rgn2K], 'group':0}
            case_inputs[("ServoDyn","VS_SlPc")]     = {'vals':[self.VS_SlPc], 'group':0}

            # Initial conditions
            case_inputs[('ElastoDyn','BlPitch1')]   = {'vals':[self.BlPitch], 'group': 0}
            case_inputs[('ElastoDyn','BlPitch2')]   = {'vals':[self.BlPitch], 'group': 0}
            case_inputs[('ElastoDyn','BlPitch3')]   = {'vals':[self.BlPitch], 'group': 0}
            case_inputs[("ElastoDyn","RotSpeed")]   = {'vals':[self.RotSpeed], 'group':0}
            
            # Hydrodyn inputs, these need to be state-space (2), but they should work if 0
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
        for var in ["BldPitch1", "BldPitch2", "BldPitch3", "NcIMURAxs", "NcIMURAys",
                    "IPDefl1",   "IPDefl2",   "IPDefl3",   "OoPDefl1",  "OoPDefl2",  "OoPDefl3",
                    "TipDxc1",   "TipDyc1",   "TipDzc1",   "TipDxb1",   "TipDyb1",
                    "TipDxc2",   "TipDyc2",   "TipDzc2",   "TipDxb2",   "TipDyb2",
                    "TipDxc3",   "TipDyc3",   "TipDzc3",   "TipDxb3",   "TipDyb3",
                    "RootMxc1",  "RootMyc1",  "RootMzc1",  "RootMxb1",  "RootMyb1",
                    "RootMxc2",  "RootMyc2",  "RootMzc2",  "RootMxb2",  "RootMyb2",
                    "RootMxc3",  "RootMyc3",  "RootMzc3",  "RootMxb3",  "RootMyb3",
                    "TwrBsMxt",  "TwrBsMyt",  "TwrBsMzt",
                    "GenPwr",    "GenTq",     "RotThrust", "RtAeroCp",  "RtAeroCt",  "RotSpeed",
                    "TTDspSS",   "TTDspFA",   "NacYaw",    "Wind1VelX", "Wind1VelY", "Wind1VelZ",
                    "LSSTipMxa", "LSSTipMya", "LSSTipMza", "LSSTipMxs", "LSSTipMys", "LSSTipMzs",
                    "LSShftFys", "LSShftFzs", "TwstDefl1", "TwstDefl2", "TwstDefl3",
                    "Spn2MLxb1", "Spn2MLxb2", "Spn2MLxb3", "Spn2MLyb1", "Spn2MLyb2", "Spn2MLyb3",
                    "TipRDxr",   "TipRDyr",   "TipRDzr"]:
            channels[var] = True

#        # Temporary treatment
#        case_inputs[("AeroDyn15","TwrShadow")] = {'vals':[0], 'group':0}
        
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
        
        # Save results into variables
        self.case_inputs = case_inputs
        self.channels = channels
        self.case_list = case_list
        self.case_name_list = case_name_list
        
# ==============================================================================
        
def set_IEA_UMaine(mdl, ws=None):
    
    # OpenFAST case
    mdl.FAST_ver = 'OpenFAST'
    mdl.dev_branch = True
    
    FAST_dir = 'examples/01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi'
    FAST_fst = 'IEA-15-240-RWT-UMaineSemi.fst'
    mdl.FAST_directory = os.path.join(mdl.WEIS_root, FAST_dir)
    mdl.FAST_InputFile = FAST_fst
    
    case_dir = os.path.join(mdl.WEIS_root, 'examples/level2ccd') + os.sep
    mdl.FAST_linearDirectory = os.path.join(
        case_dir, 'output', os.path.splitext(FAST_fst)[0], 'linear')
    mdl.FAST_nonlinearsimDirectory = os.path.join(
        case_dir, 'output', os.path.splitext(FAST_fst)[0], 'nonlinearsim')
    os.makedirs(mdl.FAST_linearDirectory, exist_ok=True)
    os.makedirs(mdl.FAST_nonlinearsimDirectory, exist_ok=True)
    
    mdl.CompMooring = 0 # 0=None, 1=MAP++, 2=FEAMooring, 3=MoorDyn, 4=OrcaFlex
    mdl.NLinTimes = 12
    mdl.TMax = 600.0
    mdl.v_rated = 10.74
    if ws == None:
        cut_in = 3.0
        cut_out = 25.0
        vrm2 = np.floor(mdl.v_rated) - 2.0
        vrp2 = np.ceil(mdl.v_rated) + 2.0
        mdl.WindSpeeds = np.append(
            np.append(
                np.linspace(cut_in, vrm2 - 1.0, int(np.ceil(vrm2 - cut_in))),
                np.linspace(vrm2, vrp2, 4*int(np.ceil(vrp2 - vrm2)) + 1)
            ),
            np.linspace(vrp2 + 1.0, cut_out, int(np.ceil(cut_out - vrp2)))
        )
        mdl.WindSpeeds = mdl.WindSpeeds.tolist()
    else:
        mdl.WindSpeeds = ws

    mdl.FAST_yamlfile_in = ''
    mdl.read_yaml = False
    mdl.FAST_yamlfile_out = ''
    mdl.write_yaml = False
    
    mdl.DT = 0.01
    mdl.DT_Out = 'default'
    mdl.OutFileFmt = 3 # 1=Text, 2=Binary, 3=Both
    mdl.TrimGain = 1e-4
    mdl.TrimTol = 1e-5
    
    # Trim case: torque for below-rated wind, pitch for over-rated wind
    TrimCase = 3*np.ones(len(mdl.WindSpeeds), dtype=int)
    TrimCase[np.array(mdl.WindSpeeds) < mdl.v_rated] = 2
    mdl.TrimCase = TrimCase.tolist()
    
    return mdl


if __name__ == '__main__':

    # PLANT PARAMETERS MODIFIED FOR DEBUG PURPOSE
    for pf in [1.0]:
    #for pf in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:

        mdl = FASTmodel()
        mdl = set_IEA_UMaine(mdl, ws=[9.0])
        mdl.casename = 'pd'
        mdl.parallel = False
        mdl.debug_level = 2

        # The following line can be removed once WEIS-packaged OpenFAST is updated to reflect
        # recent hotfix for the linearization segmentation fault error.
        #mdl.FAST_exe = '/Users/yonghoonlee/Dropbox/ATLANTIS_WEIS/openfast/install/bin/openfast'

        # DOFs
        mdl.DOFs = ['GenDOF','TwFADOF1','PtfmPDOF']
    
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

        caseid = str('PtfmMass%+09.2e' % PtfmFactor)

        ## Linearization
        #mdl.prepare_case_inputs(mdl.SOL_FLG_LINEAR, plantdesign_list, caseid)
        #mdl.runFAST_linear()
        
        # Nonlinear OpenFAST simulation using linearization conditions
        mdl.prepare_case_inputs(mdl.SOL_FLG_NONLINEARSIM, plantdesign_list, caseid)
        mdl.runFAST_nonlinearsim()
