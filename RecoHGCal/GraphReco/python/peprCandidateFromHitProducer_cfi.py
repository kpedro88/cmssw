# coding: utf-8

"""
Initialization file for the WindowInference module.
"""


__all__ = ["peprCandidateFromHitProducer"]


import math

import FWCore.ParameterSet.Config as cms


peprCandidateFromHitProducer = cms.EDProducer("peprCandidateFromHitProducer",
    # the collections of rechits to use
    recHitCollections=cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits"),
    ),
    tracks = cms.InputTag("generalTracks"),
    simClusters = cms.InputTag("mix", "MergedCaloTruth"),
    inputTensorName=cms.string("input"),
    outputTensorName=cms.string("output"),
    # whether or not the model in the graph expects a batch dimension
    batchedModel=cms.bool(True),
    # dimension of the padding of the second dimension, i.e., the rec hits themselves
    padSize=cms.uint32(10000),
    # graph to the trained model
    #graphPath=cms.string("graph.pb"),
    pipeName=cms.string("arrayspipe"),
    minEta=cms.double(1.6),
    maxEta=cms.double(3.0),
    # # window size in phi and eta
    # etaFrameWidth=cms.double(0.1),
    # phiFrameWidth=cms.double(0.1),
    # # overlap in phi and eta
    # nEtaSegments=cms.uint32(3),
    # nPhiSegments=cms.uint32(1),
    # window size in phi and eta
    etaFrameWidth=cms.double(100),
    phiFrameWidth=cms.double(100),
    # overlap in phi and eta
    nEtaSegments=cms.uint32(1),
    nPhiSegments=cms.uint32(1),   
    # names of the input and output tensors
)
