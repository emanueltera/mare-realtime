import numpy as np, pandas as pd, json
from pathlib import Path
A=Path("/workspace/data/e2/audit_leak")
val=pd.read_csv("/workspace/data/e2/sets/val/runs.csv"); val["slug"]=val.run_id.str.replace(r"^e2-val-\d+-","",regex=True)
rows=[]
for v in ["ctrl","gate0","gaterev","foam0","cloud0"]:
    r=pd.read_csv(A/"cue_sets"/v/"runs.csv")
    for rid in r.run_id:
        slug=rid.split("-",3)[3]
        vr=val[val.slug==slug].iloc[0]
        a=np.load(A/"cue_sets"/v/"tiles"/f"{rid}.npy").astype(np.float32)
        b=np.load(Path("/workspace/data/e2/sets/val/tiles")/f"{vr.run_id}.npy").astype(np.float32)
        c=np.load(A/"cue_sets"/"ctrl"/"tiles"/f"{rid}.npy").astype(np.float32)
        m=np.isfinite(a)&np.isfinite(b)
        rows.append({"variant":v,"run":slug,"theta_match":abs(r.set_index("run_id").theta_deg[rid]-vr.theta_deg)<1e-9,
          "rel_diff_vs_val":float(np.abs(a[m]-b[m]).mean()/np.abs(b[m]).mean()),
          "rel_diff_vs_ctrl":float(np.nanmean(np.abs(a-c))/np.nanmean(np.abs(c))),
          "frac_px_changed_vs_ctrl":float((np.abs(a-c)>1e-3*np.abs(c)+1e-6)[np.isfinite(c)].mean())})
D=pd.DataFrame(rows); D.to_csv(A/"cue_tile_diff.csv",index=False)
print(D.groupby("variant").agg(theta_ok=("theta_match","all"),rel_vs_val=("rel_diff_vs_val","median"),rel_vs_ctrl=("rel_diff_vs_ctrl","median"),frac_changed=("frac_px_changed_vs_ctrl","median")).to_string())
