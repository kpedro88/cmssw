#!/usr/bin/env python3
"""Create an AOTInductor .pt2 for the GlobalCache memory-scaling test producer.

Run inside cmsenv (uses the in-release Python torch + the CMS gcc, so the
generated shared library is ABI-compatible with CMS libtorch automatically).

Usage: make_aot_producer_model.py [output.pt2] [--big]
  default: a small MLP
  --big:   a wider/deeper MLP (~tens of MB of weights) so that per-thread weight
           duplication, if it happened, would be obvious in RSS.
"""
import os, sys, torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=10, hidden=256, out_dim=1, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def main():
    out = "aot_producer_model.pt2"
    big = "--big" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        out = args[0]
    torch.manual_seed(0)
    if big:
        # ~ (1024*4096 + 4096*4096*6 + ...) floats -> ~430 MB, weights dominate RSS
        model = MLP(in_dim=1024, hidden=4096, out_dim=10, depth=8).eval()
        in_dim = 1024
    else:
        model = MLP().eval()
        in_dim = 10
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparams:,}  (~{nparams*4/1e6:.1f} MB fp32)")

    example = (torch.randn(8, in_dim),)
    batch = torch.export.Dim("batch", min=1, max=4096)
    ep = torch.export.export(model, example, dynamic_shapes={"x": {0: batch}})
    path = torch._inductor.aoti_compile_and_package(ep, package_path=os.path.abspath(out))
    print("wrote", path)

    # reference output for a fixed input (ones) so the C++ side can sanity-check
    with torch.no_grad():
        ref = model(torch.ones(1, in_dim))
    print("ref_out[0,0] for input=ones:", float(ref.flatten()[0]))

if __name__ == "__main__":
    main()
