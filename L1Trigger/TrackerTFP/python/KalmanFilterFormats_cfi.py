import FWCore.ParameterSet.Config as cms

TrackTriggerKalmanFilterFormats_params = cms.PSet (

    BaseShiftx0           = cms.int32(  -6 ),
    BaseShiftx1           = cms.int32( -12 ),
    BaseShiftx2           = cms.int32(  -7 ),
    BaseShiftx3           = cms.int32(  -8 ),
    BaseShiftv0           = cms.int32( -14 ),
    BaseShiftv1           = cms.int32( -14 ),
    BaseShiftr0           = cms.int32( -16 ),
    BaseShiftr1           = cms.int32( -13 ),
    BaseShiftS00          = cms.int32(  -3 ),
    BaseShiftS01          = cms.int32(  -8 ),
    BaseShiftS12          = cms.int32(   1 ),
    BaseShiftS13          = cms.int32(   0 ),
    BaseShiftK00          = cms.int32( -16 ),
    BaseShiftK10          = cms.int32( -22 ),
    BaseShiftK21          = cms.int32( -22 ),
    BaseShiftK31          = cms.int32( -23 ),
    BaseShiftR00          = cms.int32(  -6 ),
    BaseShiftR11          = cms.int32(   3 ),
    BaseShiftR00Rough     = cms.int32(  10 ),
    BaseShiftR11Rough     = cms.int32(  10 ),
    BaseShiftInvR00Approx = cms.int32(  -7 ),
    BaseShiftInvR11Approx = cms.int32(  -5 ),
    BaseShiftInvR00Cor    = cms.int32( -15 ),
    BaseShiftInvR11Cor    = cms.int32( -15 ),
    BaseShiftInvR00       = cms.int32( -20 ),
    BaseShiftInvR11       = cms.int32( -31 ),
    BaseShiftC00          = cms.int32(   5 ),
    BaseShiftC01          = cms.int32(  -4 ),
    BaseShiftC11          = cms.int32(  -7 ),
    BaseShiftC22          = cms.int32(   3 ),
    BaseShiftC23          = cms.int32(  -1 ),
    BaseShiftC33          = cms.int32(   1 ),

    BaseShiftr02    = cms.int32(  -5 - 10 ),
    BaseShiftChi20  = cms.int32(  -5 - 10 ),
    BaseShiftr12    = cms.int32(   5 - 10 ),
    BaseShiftChi21  = cms.int32(  -5 - 10 ),
    BaseShiftChi2   = cms.int32(  -5 )

)