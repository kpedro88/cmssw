import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.inputFiles  = 'file:tmp.root'
#options.inputFiles = '/store/mc/RunIIFall17MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/90000/DCFE3F5F-AE42-E811-B6DB-008CFAF72A64.root'
#options.inputFiles = 'file:/storage/local/data1/home/jduarte1/forPatrick/FFA0194D-1BBC-EF4F-9B8F-8FBED2C62FC8.root'
options.maxEvents = -1
options.parseArguments()

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
process = cms.Process('PATtest',enableSonicTriton)

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

keepMsgs = ['TritonClient','TritonService']
keepMsgs.extend(['BoostedJetONNXJetTagsProducer'])
keepMsgs.extend(['ParticleNetSonicJetTagsProducer', 'ParticleNetSonicJetTagsProducer:TritonClient'])
for msg in keepMsgs:
    setattr(process.MessageLogger.cerr,msg,
        cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
        )
    )

## Options and Output Report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
                            skipBadFiles = cms.untracked.bool(True),
                            bypassVersionCheck = cms.untracked.bool(True),
                            overrideCatalog=cms.untracked.string(''),
                            fileNames=cms.untracked.vstring('file:tmp.root')
                            #fileNames=cms.untracked.vstring('file:test2.root')
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")
process.TritonService.verbose = True
# fallback server
process.TritonService.fallback.enable  = False
process.TritonService.fallback.verbose = False
process.TritonService.fallback.useGPU  = False
process.TritonService.servers.append(
    cms.PSet(
        name = cms.untracked.string("default"),
        address = cms.untracked.string("104.197.15.13"),
        port = cms.untracked.uint32(8021),
    )
)


## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic')
process.load("Configuration.StandardSequences.MagneticField_cff")


#process.test  = cms.EDProducer("PatatrackSonicProducer",
#                               InputLabel = cms.InputTag('rawDataCollector'),
#                               CablingMapLabel = cms.string(''),
#                               #preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK4/CHS/V00/preprocess.json',
#                               Client = cms.PSet(
#                                   timeout = cms.untracked.uint32(300),
#                                   mode = cms.string("Async"),
#                                   modelName = cms.string("identity_fp32"),
#                                   #modelName = cms.string("facile_all"),
#                                   modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/particlenet_AK4/config.pbtxt"),
#                                   #modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/facile_all_v5/config.pbtxt"),
#                                   modelVersion = cms.string("1"),
#                                   verbose = cms.untracked.bool(False),
#                                   allowedTries = cms.untracked.uint32(0),
#                                   useSharedMemory = cms.untracked.bool(False),
#                                   compression = cms.untracked.string(""),
#                               ),
#                               
##                               Client = cms.PSet(
##                                   timeout = cms.untracked.uint32(300),
##                                   mode = cms.string("Async"),
##                                   modelName = cms.string("particlenet_AK4"),
##                                   modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/particlenet_AK4/config.pbtxt"),
##                                   modelVersion = cms.string(""),
##                                   verbose = cms.untracked.bool(True),
##                                   allowedTries = cms.untracked.uint32(0),
##                                   useSharedMemory = cms.untracked.bool(True),
##                                   compression = cms.untracked.string(""),
##                                   preferredServer = cms.untracked.string(""),
##                                   outputs  = cms.untracked.vstring(""),
##                               ),
#                               debugMode = cms.untracked.bool(False)
#    )

from RecoBTag.ONNXRuntime.patatrack_cff import patatrackSONIC as pttSONIC
process.hltPTTSONIC      = pttSONIC.clone()
#process.test = pttSONIC.clone()

from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoA_cfi import siPixelDigisClustersFromSoA as _siPixelDigisClustersFromSoA
process.hltSiPixelClusters = _siPixelDigisClustersFromSoA.clone(
    src = "hltPTTSONIC"
)


from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAaaS_cfi import siPixelRecHitFromCUDAaaS as _siPixelRecHitFromCUDAaaS
process.hltSiPixelRecHits = _siPixelRecHitFromCUDAaaS.clone(
    pixelRecHitSrc = "hltPTTSONIC",
    src = "hltSiPixelClusters"
)

from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi import pixelTrackProducerFromSoA as _pixelTrackProducerFromSoA
process.hltPixelTracks = _pixelTrackProducerFromSoA.clone(
    beamSpot = "hltOnlineBeamSpot",
    pixelRecHitLegacySrc = "hltSiPixelRecHits",
    trackSrc = "hltPTTSONIC",
    #minNumberOfHits = cms.int32(0),
    minQuality = cms.string('dup'),
)

from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import pixelVertexFromSoA as _pixelVertexFromSoA
process.hltPixelVertices = _pixelVertexFromSoA.clone(
    src = "hltPTTSONIC",
    TrackCollection = "hltPixelTracks",
    beamSpot = "hltOnlineBeamSpot"
)

process.p = cms.Path(
    process.hltPTTSONIC*
    process.hltSiPixelClusters*
    process.hltSiPixelRecHits*
    process.hltPixelTracks*
    process.hltPixelVertices
)

#process.p = cms.Path(process.test)

patAlgosToolsTask = getPatAlgosToolsTask(process)

## Output Module Configuration (expects a path 'p')
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('patTuple.root'),
                               ## save only events passing the full path
                               #SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               ## save PAT output; you need a '*' to unpack the list of commands
                               ## 'patEventContent'
                               #outputCommands = cms.untracked.vstring('drop *', *patEventContentNoCleaning )
                               #outputCommands = cms.untracked.vstring('keep *')
                               #outputCommands = cms.untracked.vstring('drop *')
                               )

patAlgosToolsTask = getPatAlgosToolsTask(process)
process.outpath = cms.EndPath(process.out, patAlgosToolsTask)


from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent

process.out.fileName = 'test_particle_net_MINIAODSIM_noragged.root'

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True ),
    numberOfThreads = cms.untracked.uint32( 8 ),
    numberOfStreams = cms.untracked.uint32( 8 ),
    sizeOfStackForThreadsInKB = cms.untracked.uint32( 100*1024 )
)

