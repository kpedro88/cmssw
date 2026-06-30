#!/usr/bin/env python3
"""Run the TorchAOTProducer on multiple threads and report time/memory.

Usage:
    cmsRun torchAOTProducer_cfg.py [nthreads=N] [nevents=N] [ninfer=N] \
                                   [batch=N] [features=N] [model=path]

The model loaded into the GlobalCache is shared, read-only, across all stream
threads. Run with several values of `nthreads` and compare the
'TimeMemorySummary' RSS to confirm the weights are NOT duplicated per thread.
"""
import os
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

opts = VarParsing.VarParsing("analysis")
opts.register("nthreads", 1, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, "number of threads (== streams)")
opts.register("nevents", 2000, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, "number of events")
opts.register("ninfer", 50, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, "inferences per event (work per event)")
opts.register("batch", 256, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, "input batch size")
opts.register("features", 10, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, "input feature dimension (match the model!)")
opts.register("model", "PhysicsTools/PyTorch/test/data/aot_producer_model.pt2",
              VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, "FileInPath to the .pt2 model")
# parseArguments() chokes on no input file for the 'analysis' base; guard it.
opts.inputFiles = []
opts.parseArguments()

process = cms.Process("TorchAOT")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opts.nevents))

# N threads == N concurrent streams; this is the knob the memory test scans.
process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(opts.nthreads),
    numberOfStreams=cms.untracked.uint32(opts.nthreads),
    wantSummary=cms.untracked.bool(True),
)

# Keep logging quiet but allow the per-job summary.
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

# REQUIRED whenever PyTorch is used in CMSSW: pin torch's internal threading to 1
# so that all parallelism is at the framework (stream) level. This is what makes
# the memory-scaling measurement clean.
process.PyTorchService = cms.Service("PyTorchService")

# The tool that prints RSS / VSIZE and timing at the end of the job.
process.Timing = cms.Service("Timing", summaryOnly=cms.untracked.bool(True))
process.SimpleMemoryCheck = cms.Service(
    "SimpleMemoryCheck",
    ignoreTotal=cms.untracked.int32(1),
    moduleMemorySummary=cms.untracked.bool(True),
)

process.torchAOT = cms.EDProducer(
    "TorchAOTProducer",
    model_path=cms.FileInPath(opts.model),
    batchSize=cms.int32(opts.batch),
    nFeatures=cms.int32(opts.features),
    nInferencePerEvent=cms.uint32(opts.ninfer),
)

process.p = cms.Path(process.torchAOT)
