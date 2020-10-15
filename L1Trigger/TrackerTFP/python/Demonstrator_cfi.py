import FWCore.ParameterSet.Config as cms

TrackTriggerDemonstrator_params = cms.PSet (

  LabelIn  = cms.string( "TrackerTFPProducerKFin"          ), #
  LabelOut = cms.string( "TrackerTFPProducerKF"            ), #
  DirIPBB  = cms.string( "/heplnw039/tschuh/work/proj/kf/" ), # path to ipbb proj area
  RunTime  = cms.double( 3.0 )                                # runtime in ms

)