import FWCore.ParameterSet.Config as cms

siPixelRecHitCUDAaaS = cms.EDProducer('SiPixelRecHitCUDAaaS',
                                      pixelRecHitSrc = cms.InputTag('siPixelRecHitsPreSplittingCUDA'),
                                      src = cms.InputTag('siPixelClustersPreSplitting'),
                                      mightGet = cms.optional.untracked.vstring
                                  )
