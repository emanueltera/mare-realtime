"""Rectify the cue-probe renders with each run's val θ (matched by slug), as sets under cue_sets/<variant>."""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/workspace/mare-realtime/experiments/e2_learned_velocity")
from e2 import geom, rectify
variant = sys.argv[1]
A = Path("/workspace/data/e2/audit_leak")
val = pd.read_csv("/workspace/data/e2/sets/val/runs.csv")
val["slug"] = val.run_id.str.replace(r"^e2-val-\d+-", "", regex=True)
th = dict(zip(val.slug, val.theta_deg))
def theta_for(key):
    rid = key.split(":", 1)[1]
    slug = rid.split("-", 3)[3]   # e2-cue-NNN-<slug>
    return float(th[slug])
geom.theta_for = theta_for
render = A / "cue_render" / variant / "e2-cue"
out = A / "cue_sets" / variant
class Args: pass
a = Args(); a.render = str(render); a.out = str(out); a.set = f"cue-{variant}"; a.n = 384; a.keep = "all"; a.workers = 16; a.poll = 1.0; a.done_flag = None
rectify.drive(a, stream=False)
