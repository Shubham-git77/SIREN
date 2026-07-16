"""
One-shot patcher: add a separate `g_D_upscatter` coupling to
load_vector_portal_onshell in SIREN's DarkNewsTables/processes.py.

This lets the V1 vertices (production V1->chichi, chi'->chiV1, V1->e+e-)
use g_D = g_1 (BR-fixed) while the upscattering chi N -> chi' N uses
g_D_upscatter = g'_2, as required by the paper's double-mediator benchmark
(Dutta et al., Table II: epsilon_1=7e-5, epsilon_2=1e-4, g'_2^2/4pi=0.5).

Idempotent: running twice is harmless (it detects an existing patch).
Edits ONLY load_vector_portal_onshell; the plain and offshell factories
are left untouched.

Usage:
    python patch_onshell_coupling.py
    python patch_onshell_coupling.py /path/to/processes.py   # explicit path
"""

import os
import re
import sys
import shutil


def find_processes_py():
    # explicit path wins
    if len(sys.argv) > 1:
        return sys.argv[1]
    # else locate via the installed siren package
    try:
        from siren import _util as u
        return os.path.join(u.resource_package_dir(),
                            "processes", "DarkNewsTables", "processes.py")
    except Exception as e:
        sys.exit("Could not locate processes.py automatically (%r).\n"
                 "Pass the path explicitly:\n"
                 "  python patch_onshell_coupling.py "
                 "~/SIREN_interface/resources/processes/DarkNewsTables/processes.py"
                 % e)


def main():
    path = find_processes_py()
    if not os.path.exists(path):
        sys.exit("File not found: %s" % path)

    with open(path, "r") as f:
        src = f.read()

    if "g_D_upscatter" in src:
        print("Already patched (found 'g_D_upscatter'). Nothing to do.")
        print("File: %s" % path)
        return

    lines = src.splitlines(keepends=True)

    # --- 1. Locate the onshell function span ---------------------------------
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"\s*def\s+load_vector_portal_onshell\s*\(", ln):
            start = i
            break
    if start is None:
        sys.exit("Could not find 'def load_vector_portal_onshell(' in %s" % path)

    # function ends at the next top-level 'def ' (col 0) after start
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"def\s+\w+\s*\(", lines[j]):   # column-0 def
            end = j
            break
    span = range(start, end)
    print("Found load_vector_portal_onshell at lines %d-%d" % (start + 1, end))

    edits = 0

    # --- 2. Add `g_D_upscatter=None,` after the first `pdgid_chi=...,` --------
    for i in span:
        if re.search(r"\bpdgid_chi\s*=", lines[i]):
            indent = re.match(r"(\s*)", lines[i]).group(1)
            lines.insert(i + 1, "%sg_D_upscatter=None,\n" % indent)
            edits += 1
            print("  [1/3] inserted g_D_upscatter parameter after line %d"
                  % (i + 1))
            # shift end since we inserted a line
            end += 1
            span = range(start, end)
            break
    else:
        sys.exit("Could not find a 'pdgid_chi=' parameter inside onshell.")

    # --- 3. Default it right after the VectorPortal module load --------------
    for i in span:
        if "_load_local_module(\"VectorPortal\")" in lines[i] \
           or "_load_local_module('VectorPortal')" in lines[i]:
            indent = re.match(r"(\s*)", lines[i]).group(1)
            block = ("%sif g_D_upscatter is None:\n"
                     "%s    g_D_upscatter = g_D\n" % (indent, indent))
            lines.insert(i + 1, block)
            edits += 1
            print("  [2/3] inserted default (g_D_upscatter = g_D) after line %d"
                  % (i + 1))
            end += 1
            span = range(start, end)
            break
    else:
        print("  [2/3] WARNING: could not find _load_local_module(\"VectorPortal\"); "
              "add the default manually:\n"
              "        if g_D_upscatter is None:\n"
              "            g_D_upscatter = g_D")

    # --- 4. In the VectorPortalUpscatteringXS(...) call, g_D=g_D -> g_D_upscatter
    # find the upscatter call, then the g_D=g_D line within its argument block
    ups_start = None
    for i in span:
        if "VectorPortalUpscatteringXS(" in lines[i]:
            ups_start = i
            break
    if ups_start is None:
        sys.exit("Could not find VectorPortalUpscatteringXS( call inside onshell.")
    # search the next ~20 lines for g_D=g_D
    for i in range(ups_start, min(ups_start + 25, end)):
        m = re.match(r"(\s*)g_D\s*=\s*g_D\s*,\s*$", lines[i])
        if m:
            lines[i] = "%sg_D=g_D_upscatter,\n" % m.group(1)
            edits += 1
            print("  [3/3] changed g_D=g_D -> g_D=g_D_upscatter in upscatter "
                  "call at line %d" % (i + 1))
            break
    else:
        sys.exit("Could not find 'g_D=g_D,' inside the VectorPortalUpscatteringXS "
                 "call. Change it manually to 'g_D=g_D_upscatter,'.")

    if edits < 3:
        print("WARNING: only %d/3 edits applied; review the file." % edits)

    # --- backup and write -----------------------------------------------------
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print("Backup written: %s" % bak)
    else:
        print("Backup already exists: %s (not overwritten)" % bak)

    with open(path, "w") as f:
        f.writelines(lines)
    print("Patched: %s" % path)
    print("\nVerify with:")
    print("  python3 -c \"import inspect,os; from siren import _util as u; "
          "m=u.load_module('p',os.path.join(u.resource_package_dir(),"
          "'processes','DarkNewsTables','processes.py')); "
          "print('PATCH OK' if 'g_D_upscatter' in "
          "inspect.signature(m.load_vector_portal_onshell).parameters "
          "else 'PATCH MISSING')\"")


if __name__ == "__main__":
    main()
