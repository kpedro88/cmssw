import FWCore.ParameterSet.Config as cms
from RecoVertex.Configuration.RecoVertex_cff import unsortedOfflinePrimaryVertices, trackWithVertexRefSelector, trackRefsForJets, sortedPrimaryVertices, offlinePrimaryVertices, offlinePrimaryVerticesWithBS,vertexrecoTask

from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA2D_vectParameters

unsortedOfflinePrimaryVertices4D = unsortedOfflinePrimaryVertices.clone(TkClusParameters = DA2D_vectParameters,
                                                                        TrackTimesLabel = cms.InputTag("tofPID:t0safe"),
                                                                        TrackTimeResosLabel = cms.InputTag("tofPID:sigmat0safe"),
                                                                        )
trackWithVertexRefSelectorBeforeSorting4D = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4D",
                                                                                    ptMax=9e99,
                                                                                    ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4D = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4D")
offlinePrimaryVertices4D=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4D",
                                                            particles="trackRefsForJetsBeforeSorting4D",
                                                            trackTimeTag=cms.InputTag("tofPID:t0safe"),
                                                            trackTimeResoTag=cms.InputTag("tofPID:sigmat0safe"),
                                                            assignment=dict(useTiming=True))
offlinePrimaryVertices4DWithBS=offlinePrimaryVertices4D.clone(vertices="unsortedOfflinePrimaryVertices4D:WithBS")

unsortedOfflinePrimaryVertices4DnoPID = unsortedOfflinePrimaryVertices4D.clone(TrackTimesLabel = "trackExtenderWithMTD:generalTrackt0",
                                                                         TrackTimeResosLabel = "trackExtenderWithMTD:generalTracksigmat0",
                                                                         )
trackWithVertexRefSelectorBeforeSorting4DnoPID = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4DnoPID",
                                                                                  ptMax=9e99,
                                                                                  ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4DnoPID = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4DnoPID")
offlinePrimaryVertices4DnoPID=offlinePrimaryVertices4D.clone(vertices="unsortedOfflinePrimaryVertices4DnoPID",
                                                          particles="trackRefsForJetsBeforeSorting4DnoPID",
                                                          trackTimeTag="trackExtenderWithMTD:generalTrackt0",
                                                          trackTimeResoTag="trackExtenderWithMTD:generalTracksigmat0")
offlinePrimaryVertices4DnoPIDWithBS=offlinePrimaryVertices4DnoPID.clone(vertices="unsortedOfflinePrimaryVertices4DnoPID:WithBS")

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import quickTrackAssociatorByHits
from CommonTools.RecoAlgos.tofPID_cfi import tofPID
