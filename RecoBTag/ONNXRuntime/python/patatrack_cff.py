import FWCore.ParameterSet.Config as cms

patatrackSONIC  = cms.EDProducer("PatatrackSonicProducer",
                               InputLabel = cms.InputTag('rawDataCollector'),
                               CablingMapLabel = cms.string(''),
                               #preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK4/CHS/V00/preprocess.json',                                                                                                                          
                               Client = cms.PSet(
                                   timeout = cms.untracked.uint32(300),
                                   mode = cms.string("Async"),
                                   modelName = cms.string("identity_fp32"),
                                   #modelName = cms.string("facile_all"),                                                                                                                                                                   
                                   modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/particlenet_AK4/config.pbtxt"),
                                   #modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/facile_all_v5/config.pbtxt"),                                                                                               
                                   modelVersion = cms.string("1"),
                                   verbose = cms.untracked.bool(False),
                                   allowedTries = cms.untracked.uint32(0),
                                   useSharedMemory = cms.untracked.bool(False),
                                   compression = cms.untracked.string(""),
                               ),

                               debugMode = cms.untracked.bool(False)
    )
