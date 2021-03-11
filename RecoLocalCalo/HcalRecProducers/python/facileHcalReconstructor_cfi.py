import FWCore.ParameterSet.Config as cms

sonic_hbheprereco = cms.EDProducer("FacileHcalReconstructor",
    Client = cms.PSet(
        batchSize = cms.untracked.uint32(10000),
        address = cms.untracked.string("ailab01.fnal.gov"),
        port = cms.untracked.uint32(8001),
        timeout = cms.untracked.uint32(300),
        modelName = cms.string("facile_plan_10k"),
        mode = cms.string("Async"),
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
        outputs = cms.untracked.vstring("Identity:0"),
    ),
    ChannelInfoName = cms.InputTag("hbhechannelinfo")
)
