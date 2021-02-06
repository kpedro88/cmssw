import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton

process = cms.Process('TEST',Phase2C9,enableSonicTriton)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.TritonService.verbose = True
process.TritonService.fallback.verbose = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step3.root'),
    secondaryFileNames = cms.untracked.vstring()
)

# Output definition
process.OCoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('step4.root'),
    outputCommands = cms.untracked.vstring('keep *_objectCondensationCandidateProducer_*_*'),
    splitLevel = cms.untracked.int32(0)
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

process.load('RecoHGCal.GraphReco.objectCondensationCandidateProducer_cff')
process.objectCondensationCandidateProducer.Client.verbose = True

# Path and EndPath definitions
process.p = cms.Path(process.objectCondensationCandidateProducer)
process.o = cms.EndPath(process.OCoutput)

keepMsgs = ['TritonClient','TritonService','ObjectCondensationCandidateProducer','ObjectCondensationCandidateProducer:TritonClient']
for msg in keepMsgs:
    setattr(process.MessageLogger.cerr,msg,
        cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
        )
    )
