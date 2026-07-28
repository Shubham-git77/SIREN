"""
FHC (neutrino mode) vs RHC (antineutrino mode) comparison of the NuMI dark-portal
in-window anchored yields across the three SBN detectors and three portals.

RHC numbers are the 5-file g4numi RHC result; FHC is parsed from the FHC pipeline
log (standard fhc, decay-in-flight). Grouped bars: detector x portal, RHC vs FHC.
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Fallback RHC (5-file g4numi RHC) if no RHC log is given.
RHC_DEFAULT = {
    "ICARUS":    {"scalar": 1999, "pseudo": 2892, "vector": 1325},
    "SBND":      {"scalar": 188,  "pseudo": 499,  "vector": 73},
    "MiniBooNE": {"scalar": 1297, "pseudo": 2613, "vector": 289},
}


def parse_log(logpath):
    """Pull in-window anchored per-portal numbers per detector from a run log."""
    fhc = {d: {} for d in ("ICARUS", "SBND", "MiniBooNE")}
    txt = open(logpath).read()
    # ICARUS line: 'PORTAL IN-WINDOW ANCHORED (... ): scalar=.. pseudo=.. vector=..'
    pats = {
        "ICARUS":    r"PORTAL IN-WINDOW ANCHORED[^:]*:\s*scalar=([\d.]+)\s+pseudo=([\d.]+)\s+vector=([\d.]+)",
        "SBND":      r"SBND NuMI IN-WINDOW ANCHORED[^:]*:\s*scalar=([\d.]+)\s+pseudo=([\d.]+)\s+vector=([\d.]+)",
        "MiniBooNE": r"MiniBooNE NuMI IN-WINDOW ANCHORED[^:]*:\s*scalar=([\d.]+)\s+pseudo=([\d.]+)\s+vector=([\d.]+)",
    }
    for det, pat in pats.items():
        m = re.search(pat, txt)
        if m:
            fhc[det] = {"scalar": float(m.group(1)), "pseudo": float(m.group(2)),
                        "vector": float(m.group(3))}
    return fhc


def main():
    # usage: plot_fhc_vs_rhc.py <FHC_log> [RHC_log]
    fhc_log = sys.argv[1]
    FHC = parse_log(fhc_log)
    if len(sys.argv) > 2:
        RHC = parse_log(sys.argv[2])
        rhc_tag = "single-file rhc_2008"
    else:
        RHC = RHC_DEFAULT
        rhc_tag = "5-file rhc"
    dets = ["ICARUS", "SBND", "MiniBooNE"]
    portals = ["scalar", "pseudo", "vector"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    x = np.arange(len(portals)); wbar = 0.38
    for ax, det in zip(axes, dets):
        rhc = [RHC[det][p] for p in portals]
        fhc = [FHC.get(det, {}).get(p, 0) for p in portals]
        b1 = ax.bar(x - wbar/2, rhc, wbar, label="RHC ($\\bar\\nu$-mode)", color="tab:blue")
        b2 = ax.bar(x + wbar/2, fhc, wbar, label="FHC ($\\nu$-mode)", color="tab:orange")
        for b in (b1, b2):
            ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
        ax.set_xticks(x); ax.set_xticklabels(portals)
        ax.set_title("%s $\\times$ NuMI" % det)
        ax.set_ylabel("in-window anchored events (3e21 POT)")
        # ratio annotation
        for i, p in enumerate(portals):
            r = FHC.get(det, {}).get(p, 0) / RHC[det][p] if RHC[det][p] else 0
            ax.text(i, max(rhc[i], fhc[i]) * 1.08, "x%.1f" % r, ha="center",
                    fontsize=8, color="dimgray")
        ax.legend(fontsize=8)
        ax.set_ylim(top=max(max(rhc), max(fhc)) * 1.25)
    fig.suptitle("NuMI dark-portal signal: FHC (neutrino mode) vs RHC (antineutrino mode)  "
                 "--  in-window anchored, verified NuMI->BNB transform  (RHC = %s, FHC = single-file fhc_1007)\n"
                 "(vector = imprecise: ~5-event MiniBooNE anchor denominator)" % rhc_tag, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(HERE, "output", "FHC_vs_RHC_comparison.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
    # text table
    print("\n%-11s %-8s %10s %10s %8s" % ("detector", "portal", "RHC", "FHC", "FHC/RHC"))
    for det in dets:
        for p in portals:
            r, f = RHC[det][p], FHC.get(det, {}).get(p, 0)
            print("%-11s %-8s %10.0f %10.0f %8.2f" % (det, p, r, f, f/r if r else 0))


if __name__ == "__main__":
    main()
