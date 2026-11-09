"""Throughput and memory of FlowHeadNet on random tiles (one training step / one inference)."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowhead.model import FlowHeadNet

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
for gaps, frozen, B, crop, amp in [((1, 2, 4, 7), "frozen", 8, 256, True), ((1, 2, 4, 7), "frozen", 16, 256, True),
                                    ((1, 2, 4, 7), "frozen", 8, 384, True)]:
    m = FlowHeadNet(gaps=gaps, backbone=frozen).to(dev).train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], 1e-5)
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(B, 8, crop, crop, device=dev); mk = torch.ones_like(x, dtype=torch.bool)
    g = torch.full((B,), 0.1, device=dev); dt = torch.full((B,), 0.1, device=dev)
    for i in range(8):
        if i == 3:
            torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            v, s, _ = m(x, mk, g, dt)
        loss = v.float().square().mean() + s.float().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); dtt = (time.time() - t) / 5
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for i in range(5):
            if i == 2:
                torch.cuda.synchronize(); t = time.time()
            m(x, mk, g, dt)
    torch.cuda.synchronize(); dti = (time.time() - t) / 3
    print(f"gaps {gaps} frozen {frozen} B {B} crop {crop} amp {amp}: train {dtt*1000:.0f} ms/step "
          f"({B/dtt:.1f} win/s), infer {B/dti:.1f} win/s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    del m, opt
    torch.cuda.empty_cache()
