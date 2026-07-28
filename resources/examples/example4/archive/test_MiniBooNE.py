"""
Minimal test: load the MiniBooNE detector via the new SIREN GDML pipeline
and inspect the resulting DetectorModel. No simulation — just confirm the
geometry composes, the GDML parses, and the sectors/materials look right.
"""

import sys

print("=" * 64)
print("  MiniBooNE detector load test")
print("=" * 64)

# 1. Import the loader -------------------------------------------------------
try:
    from siren._util import load_detector
except Exception as e:
    print("FAIL: could not import load_detector from siren._util")
    print("      %r" % e)
    sys.exit(1)
print("  [ok] imported load_detector")

# 2. Load MiniBooNE ----------------------------------------------------------
try:
    model = load_detector("SBN", detector="MiniBooNE")
except Exception as e:
    print("FAIL: load_detector('SBN', detector='MiniBooNE') raised:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("  [ok] load_detector returned a %s" % type(model).__name__)

# 3. Detector origin / rotation ---------------------------------------------
try:
    origin = model.DetectorOrigin
    print("  DetectorOrigin (BNB frame): %s" % (origin,))
except Exception as e:
    print("  [warn] could not read DetectorOrigin: %r" % e)

try:
    rot = model.DetectorRotation
    print("  DetectorRotation (quat)   : X=%.4f Y=%.4f Z=%.4f W=%.4f"
          % (rot.X, rot.Y, rot.Z, rot.W))
except Exception as e:
    print("  [warn] could not read DetectorRotation: %r" % e)

# 4. Sectors -----------------------------------------------------------------
try:
    sectors = model.Sectors
    print("\n  Sectors (%d total):" % len(sectors))
    print("  %-28s %-8s %s" % ("name", "level", "material_id"))
    print("  " + "-" * 50)
    for s in sectors:
        print("  %-28s %-8s %s" % (s.name, s.level, s.material_id))
except Exception as e:
    print("  [warn] could not list sectors: %r" % e)
    import traceback
    traceback.print_exc()

# 5. Materials ---------------------------------------------------------------
try:
    mats = model.GetMaterials()
    # try to enumerate material names if the API exposes it
    print("\n  Materials present (probing common MiniBooNE/site keys):")
    for name in ["MINERAL_OIL", "OIL", "MineralOil", "STEEL", "CARBON_STEEL",
                 "AIR", "CONCRETE", "DIRT", "TILL", "DOLOMITE", "ROCK"]:
        try:
            mid = mats.GetMaterialId(name)
            print("    %-16s -> id %s" % (name, mid))
        except Exception:
            pass
except Exception as e:
    print("  [warn] could not query materials: %r" % e)

# 6. Quick geometry sanity: column depth from beam toward tank center --------
try:
    from siren.math import Vector3D
    o = model.DetectorOrigin
    print("\n  Tank center in BNB frame ~ %s" % (o,))
    print("  (expected near x=0, y~+1.9 m, z~541.3 m per sbn_geometry)")
except Exception as e:
    print("  [warn] geometry sanity probe failed: %r" % e)

print("\n" + "=" * 64)
print("  Done. If sectors include the oil/veto/steel/vault volumes and")
print("  materials resolve, the MiniBooNE GDML pipeline is working.")
print("=" * 64)
o = model.DetectorOrigin.get()   # or model.DetectorOrigin.GetPosition() depending on API
print(o)
