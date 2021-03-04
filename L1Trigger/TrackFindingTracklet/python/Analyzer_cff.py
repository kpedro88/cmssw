import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.Analyzer_cfi import TrackFindingTrackletAnalyzer_params
from L1Trigger.TrackFindingTracklet.ProducerKF_cfi import TrackFindingTrackletProducerKF_params

TrackFindingTrackletAnalyzerTracklet = cms.EDAnalyzer( 'trackFindingTracklet::AnalyzerTracklet', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducerKF_params )
TrackFindingTrackletAnalyzerKFin = cms.EDAnalyzer( 'trackFindingTracklet::AnalyzerKFin', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducerKF_params )
TrackFindingTrackletAnalyzerKF = cms.EDAnalyzer( 'trackerTFP::AnalyzerKF', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducerKF_params )