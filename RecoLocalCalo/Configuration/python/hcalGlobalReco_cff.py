import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

#--- for Run 1 and Run 2
from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import hbhereco as _phase0_hbhereco
hbhereco = SwitchProducerCUDA(
    cpu = _phase0_hbhereco.clone()
)
hcalGlobalRecoTask = cms.Task(hbhereco)
hcalGlobalRecoSequence = cms.Sequence(hcalGlobalRecoTask)

hcalOnlyGlobalRecoTask = cms.Task()
hcalOnlyGlobalRecoSequence = cms.Sequence(hcalOnlyGlobalRecoTask)

#--- for Run 3 and later
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB

from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco
run3_HB.toReplaceWith(hbhereco.cpu, _phase1_hbheprereco)
run3_HB.toReplaceWith(hcalOnlyGlobalRecoTask, cms.Task(hbhereco))

#--- for Run 3 on GPU
from Configuration.ProcessModifiers.gpu_cff import gpu

from RecoLocalCalo.HcalRecProducers.hcalCPURecHitsProducer_cfi import hcalCPURecHitsProducer as _hbherecoFromCUDA
(run3_HB & gpu).toModify(hbhereco,
    cuda = _hbherecoFromCUDA.clone(
        produceSoA = False
    )
)
run3_HB.toReplaceWith( hbhereco, _phase1_hbheprereco ) # >=Run3

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
