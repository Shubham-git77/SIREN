"""Single source of truth for SBN beam exposures (POT).

These used to be hardcoded independently in each config, which produced a real
inconsistency: ScalarPortal_ICARUS_multichannel.py carried ICARUS_POT = 6e20
labelled "ICARUS NuMI exposure", while analytic_NuMI_ICARUS.py carried
ICARUS_NUMI_POT = 3e21. Same detector, two numbers differing by 5x, and the
label on the first was wrong -- that config runs the BNB chain (its geometry is
ICARUS_MODULE_CENTERS_BNB at z = 600 m in BNB coordinates), so 6e20 was a BNB
exposure wearing a NuMI name. Which script you happened to run decided your
answer.

The physical point that fixes it: SBND and ICARUS sit on the SAME BNB beamline
in the SAME run, so the POT delivered to them is identical by construction. It
is one number, not two. ICARUS additionally sits off-axis on the NuMI beamline,
which is a genuinely separate exposure and therefore a separate name.

Every value here is overridable from the environment so a scan can vary the
exposure without editing code -- nothing downstream should hardcode a POT.
"""
import os


def _env(name, default):
    """Read a POT from the environment, falling back to the nominal value."""
    return float(os.environ.get(name, default))


# --- BNB -------------------------------------------------------------------
# SBN programme nominal neutrino-mode exposure. One beam, one run, so this is
# what BOTH near (SBND) and far (ICARUS) detectors receive.
BNB_POT_SBN_RUN = _env("BNB_POT", 6.6e20)

SBND_POT = BNB_POT_SBN_RUN
ICARUS_BNB_POT = BNB_POT_SBN_RUN          # identical by construction, same beam

# --- NuMI ------------------------------------------------------------------
# ICARUS also sits ~6 degrees off the NuMI axis. This is a different beam and a
# different exposure, hence a different name -- never interchange it with the
# BNB number above.
ICARUS_NUMI_POT = _env("ICARUS_NUMI_POT", 3.0e21)

# --- MiniBooNE -------------------------------------------------------------
# Neutrino-mode dataset of Aguilar-Arevalo et al., PRD 103 052002 (2021)
# (arXiv:2006.16883) -- the exposure the fitted excess corresponds to, and the
# one Dutta-Kim fit against. The older 6.46e20 is the 2007 first result and
# must not be used with the 2020/2021 data.
MINIBOONE_POT = _env("MINIBOONE_POT", 18.75e20)

if __name__ == "__main__":
    print("BNB (SBN run)   : %.3e POT  -> SBND and ICARUS alike" % BNB_POT_SBN_RUN)
    print("ICARUS on NuMI  : %.3e POT  (separate beam)" % ICARUS_NUMI_POT)
    print("MiniBooNE       : %.3e POT  (2020/2021 nu-mode dataset)" % MINIBOONE_POT)
    print("ICARUS NuMI / BNB ratio = %.2f" % (ICARUS_NUMI_POT / ICARUS_BNB_POT))
