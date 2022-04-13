import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
from Configuration.ProcessModifiers.deepMETSonicTriton_cff import deepMETSonicTriton

# collect all SonicTriton-related process modifiers here
allSonicTriton = cms.ModifierChain(enableSonicTriton,deepMETSonicTriton)

