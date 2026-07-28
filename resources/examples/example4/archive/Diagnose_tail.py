# Diagnose the weight tail: what do the high-weight events look like?
# Run this on the scalar output npz.
import numpy as np
import sys
f = sys.argv[1] if len(sys.argv)>1 else "output/MiniBooNE_ScalarPrimakoff_multichannel_observables.npz"
d = np.load(f)
print("keys:", list(d.keys()))
w = d["weight"]; E = d["E_vis"]*1e3; c = d["cos_theta"]
ok = np.isfinite(w) & (w>0)
w,E,c = w[ok],E[ok],c[ok]
print(f"N={len(w)}  sum={w.sum():.3e}  median={np.median(w):.3e}  max={w.max():.3e}")
print(f"max/median={w.max()/np.median(w):.2e}\n")
# sort by weight, show the top 10
idx = np.argsort(w)[::-1]
print("TOP 10 weight events (w, E_gamma[MeV], cos_theta):")
for i in idx[:10]:
    print(f"  w={w[i]:.3e}  E_gamma={E[i]:7.1f}  cos={c[i]:+.4f}")
print("\nBOTTOM 5 (typical) events:")
for i in idx[-5:]:
    print(f"  w={w[i]:.3e}  E_gamma={E[i]:7.1f}  cos={c[i]:+.4f}")
# what fraction of total weight is in the top 1%, 5 events?
sw = np.sort(w)[::-1]
print(f"\ntop 1 event = {sw[0]/w.sum()*100:.1f}% of sum")
print(f"top 5 events = {sw[:5].sum()/w.sum()*100:.1f}% of sum")
print(f"top 1% events = {sw[:max(1,len(w)//100)].sum()/w.sum()*100:.1f}% of sum")
# Do high weights cluster at high or low E_gamma? at forward cos?
hi = w > np.percentile(w, 99)
print(f"\nhigh-weight (top 1%) events: E_gamma range [{E[hi].min():.0f},{E[hi].max():.0f}] MeV, cos range [{c[hi].min():.3f},{c[hi].max():.3f}]")
print(f"typical events:             E_gamma range [{E[~hi].min():.0f},{E[~hi].max():.0f}] MeV, cos range [{c[~hi].min():.3f},{c[~hi].max():.3f}]")
