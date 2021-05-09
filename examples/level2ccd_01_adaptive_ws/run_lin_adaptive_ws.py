import os, platform, random, shutil
import importlib, sys
import numpy as np
from linearization import adaptive_ws_linearization as AWSL
case_dir = os.path.dirname(os.path.abspath(__file__))

# Parameters

PtfmFactor = 1.0
InitialWindSpeeds = np.linspace(3.0, 25.0, 3)


# Setup IEA15 Turbine Model

linmodel = AWSL()

FAST_parameters = {
    'casename': 'P01',
    'fid': str('%+011.4e' % PtfmFactor),
    'parallel': True,
    'debug_level': 2,
    'FAST_ver': 'OpenFAST',
    'FAST_directory': os.path.join(
        os.path.dirname(os.path.dirname(importlib.util.find_spec('weis').origin)),
        'examples/01_aeroelasticse/OpenFAST_models',
        'IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi'
    ),
    'FAST_InputFile': 'IEA-15-240-RWT-UMaineSemi.fst',
    'FAST_steadyDirectory': os.path.join(case_dir, 'output', 'steady'),
    'FAST_linearDirectory': os.path.join(case_dir, 'output', 'linear'),
    'CompMooring': 0, # 0=None, 1=MAP++, 2=FEAMooring, 3=MoorDyn, 4=OrcaFlex
    'DOFs': ['GenDOF','TwFADOF1','PtfmPDOF'],
    'HydroStates': True,
    'NLinTimes': 12,
    'TMax': 2000.0,
    'v_rated': 10.74,
    'FAST_yamlfile_in': '',
    'FAST_yamlfile_out': '',
    'read_yaml': False,
    'write_yaml': False,
    'DT': 0.025,
    'DT_Out': 'default',
    'OutFileFmt': 2, # 1=Text, 2=Binary, 3=Both
    'TrimGain': 4e-5,
    'TrimTol': 1e-5,
    'VS_RtGnSp': 7.56*0.95, # Track 95% of rated generator speed to improve convergence near rated condition
    'VS_RtTq': 19.62e6,
    'VS_Rgn2K': 3.7e5,
    'VS_SlPc': 10.0,
}

os.makedirs(FAST_parameters['FAST_steadyDirectory'], exist_ok=True)
os.makedirs(FAST_parameters['FAST_linearDirectory'], exist_ok=True)


# Setup plant parameters

plantdesign_list = []

pd = linmodel.mdl.plant_design()
pd.modulename = 'ElastoDyn'
pd.deckname = 'PtfmMass'
pd.value = 1.7838E+07*PtfmFactor # mass
plantdesign_list.append(pd)

pd = linmodel.mdl.plant_design()
pd.modulename = 'ElastoDyn'
pd.deckname = 'PtfmRIner'
pd.value = 1.2507E+10*PtfmFactor # Ixx = mr^2
plantdesign_list.append(pd)

pd = linmodel.mdl.plant_design()
pd.modulename = 'ElastoDyn'
pd.deckname = 'PtfmPIner'
pd.value = 1.2507E+10*PtfmFactor # Iyy = mr^2
plantdesign_list.append(pd)

pd = linmodel.mdl.plant_design()
pd.modulename = 'ElastoDyn'
pd.deckname = 'PtfmYIner'
pd.value = 2.3667E+10*PtfmFactor # Izz = mr^2
plantdesign_list.append(pd)


# Apply parameters to model
linmodel.set_FAST_parameters(FAST_parameters)
linmodel.set_Plant_parameters(plantdesign_list)
linmodel.set_initial_ws(InitialWindSpeeds)


# Run SteadyFAST

linmodel.calc_steady()


# Run LinearFAST

linmodel.calc_linear()

