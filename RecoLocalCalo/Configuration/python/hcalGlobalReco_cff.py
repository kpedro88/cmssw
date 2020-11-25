import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
hcalGlobalRecoTask = cms.Task(hbhereco)
hcalGlobalRecoSequence = cms.Sequence(hcalGlobalRecoTask)

#--- for Run 3 and later
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB

from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco
run3_HB.toReplaceWith(hbhereco, _phase1_hbheprereco)

#--- for Run 3 on GPU
from Configuration.ProcessModifiers.gpu_cff import gpu

from RecoLocalCalo.HcalRecProducers.hcalCPURecHitsProducer_cfi import hcalCPURecHitsProducer as _hcalCPURecHitsProducer
gpu.toReplaceWith(hbhereco, _hcalCPURecHitsProducer.clone(
    recHitsM0LabelIn = "hbheRecHitProducerGPU",
    recHitsM0LabelOut = "",
    recHitsLegacyLabelOut = ""
))

#--- ML-based reco using SONIC+Triton
hbhechannelinfo = _phase1_hbheprereco.clone(
    makeRecHits = False,
    saveInfos = True,
    processQIE8 = False,
)
from RecoLocalCalo.HcalRecProducers.facileHcalReconstructor_cfi import sonic_hbheprereco as _sonic_hbheprereco
from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
(enableSonicTriton & run3_HB).toReplaceWith(hbhereco, _sonic_hbheprereco)
(enableSonicTriton & run3_HB).toModify(hcalGlobalRecoTask, lambda x: x.add(hbhechannelinfo))
