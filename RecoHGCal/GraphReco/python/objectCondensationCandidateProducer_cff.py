import FWCore.ParameterSet.Config as cms

from RecoHGCal.GraphReco.objectCondensationCandidateProducer_cfi import objectCondensationCandidateProducer as _objectCondensationCandidateProducer

objectCondensationCandidateProducer = _objectCondensationCandidateProducer.clone(
    Client = dict(
        timeout = 300,
        modelName = "hgcal_oc_reco",
        modelVersion = "",
        modelConfigPath = "RecoHGCal/GraphReco/data/models/hgcal_oc_reco/config.pbtxt",
        outputs = ["output"],
    )
)
