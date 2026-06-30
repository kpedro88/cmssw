# TorchAOTProducer: AOTInductor model as a shared GlobalCache resource

A demonstration that a `torch.export` / AOTInductor PyTorch model can be loaded
**once** into an `edm::GlobalCache` and used **concurrently, read-only, from many
CMSSW stream threads** — with memory cost O(model), not O(model × threads).

## Components

- **`plugins/TorchAOTProducer.cc`** — `edm::stream::EDProducer<edm::GlobalCache<AOTCache>>`.
  `initializeGlobalCache()` loads one `cms::torch::ModelAOT`
  (`torch::inductor::AOTIModelPackageLoader`); every stream runs inference on that
  same shared const model in `produce()`.
- **`test/make_aot_producer_model.py`** — exports an AOTInductor `.pt2`. Default is
  a tiny MLP; `--big` makes a ~0.5 GB-weight model. Run inside `cmsenv` so the
  in-release Python torch compiles the package with the CMS gcc (ABI-compatible
  with CMS libtorch).
- **`test/torchAOTProducer_cfg.py`** — cmsRun config. Knobs: `nthreads`, `nevents`,
  `ninfer`, `batch`, `features`, `model` (FileInPath). Loads
  `PyTorchService` (pins torch to 1 internal thread) and the `Timing` /
  `SimpleMemoryCheck` services for the memory summary.
- **`test/scan_threads.sh`** — runs the config for 1/2/4/8 threads and tabulates
  peak RSS.
- **`test/run_torchAOT_memtest.sh`** — self-contained `scram b runtests` entry:
  generates a small model, runs shared vs control on 1 and 4 threads, asserts the
  shared model's `.so` RSS (where the weights live) stays flat across threads.

## Quick start

```bash
cd $CMSSW_BASE/src && cmsenv
scram b -j8

cd PhysicsTools/PyTorch/test
# a model with ~88 MB of weights makes the effect obvious:
python3 - <<'PY'
import torch,torch.nn as nn
class M(nn.Module):
  def __init__(s):
    super().__init__();L=[];d=512
    for _ in range(6):L+=[nn.Linear(d,2048),nn.ReLU()];d=2048
    L+=[nn.Linear(d,10)];s.n=nn.Sequential(*L)
  def forward(s,x):return s.n(x)
m=M().eval();ep=torch.export.export(m,(torch.randn(8,512),),
  dynamic_shapes={"x":{0:torch.export.Dim("b",min=1,max=4096)}})
torch._inductor.aoti_compile_and_package(ep,package_path="data/aot_mid.pt2")
PY

M=PhysicsTools/PyTorch/test/data/aot_mid.pt2
# shared GlobalCache: RSS barely grows with threads (weights loaded once)
cmsRun torchAOTProducer_cfg.py nthreads=1 nevents=200 features=512 model=$M
cmsRun torchAOTProducer_cfg.py nthreads=8 nevents=200 features=512 model=$M
```

Inspect the `MemoryReport> Peak rss size ...` line and the `of which .so's ...
(RSS)` line in the output. See `../../../cmssw_aot_memory_scaling.md` (top of the
CMSSW area) for the full results and discussion.

## Note on `scram b runtests`

`pytorch-torchAOT-memtest` is registered in `test/BuildFile.xml`. Running the full
`scram b runtests` rebuilds all of the package's (CUDA) test binaries first, which
is slow; the memory test itself can also be run directly:

```bash
LOCALTOP=$CMSSW_BASE bash test/run_torchAOT_memtest.sh
```
