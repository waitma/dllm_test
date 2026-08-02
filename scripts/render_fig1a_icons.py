#!/usr/bin/env python3
"""Render Nature-palette PyMOL cartoon icons for fig1a.

PDBs: 1AO7 (TCR–pMHC), 1NL0 (Ab–Ag), 3K1K (Nb–Ag), 1BRS (PPI).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pymol import cmd

OUT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/figures/icons")
PDB = Path("/tmp/struct_icons")
NATURE = ["3C6E8F", "C17F65", "5B8F7A", "8B7BA0", "B0A16B", "6A8FA3"]

SPECS = [
    dict(name="tcr_pmhc", pdb="1ao7.pdb", sel="chain A+B+C+D+E", turn=(5, -30, 0)),
    dict(name="antibody", pdb="1nl0.pdb", sel="chain H+L+G", turn=(15, 35, 0)),
    dict(name="nanobody", pdb="3k1k.pdb", sel="chain A+B", turn=(20, -40, 0)),
    dict(name="ppi", pdb="1brs.pdb", sel="chain A+D", turn=(10, 40, 0)),
]


def render(spec: dict) -> None:
    cmd.reinitialize()
    cmd.load(str(PDB / spec["pdb"]), "mol")
    cmd.remove(f"mol and not ({spec['sel']})")
    cmd.hide("everything")
    cmd.show("cartoon", "polymer")
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_flat_sheets", 1)
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_shadow", 0)
    cmd.set("antialias", 2)
    cmd.set("ambient", 0.48)
    cmd.set("specular", 0.12)
    cmd.set("shininess", 8)
    cmd.set("light_count", 2)
    cmd.bg_color("white")
    for i, ch in enumerate(cmd.get_chains("mol")):
        col = NATURE[i % len(NATURE)]
        cname = f"nat{i}"
        cmd.set_color(cname, [int(col[j : j + 2], 16) / 255 for j in (0, 2, 4)])
        cmd.color(cname, f"chain {ch}")
    cmd.orient("visible")
    tx, ty, tz = spec["turn"]
    cmd.turn("x", tx)
    cmd.turn("y", ty)
    cmd.turn("z", tz)
    cmd.zoom("visible", 1.8)
    hi = OUT / f"{spec['name']}_hi.png"
    cmd.png(str(hi), width=500, height=500, dpi=150, ray=1)
    im = Image.open(hi).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    arr = np.asarray(im)
    ys, xs = np.where((arr < 248).any(axis=2))
    pad = 18
    im = im.crop(
        (
            max(0, xs.min() - pad),
            max(0, ys.min() - pad),
            min(im.width, xs.max() + 1 + pad),
            min(im.height, ys.max() + 1 + pad),
        )
    )
    w, h = im.size
    side = max(w, h) + 8
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((240, 240), Image.Resampling.LANCZOS)
    canvas.save(OUT / f"{spec['name']}.png", "PNG", optimize=True)
    hi.unlink(missing_ok=True)
    print("wrote", spec["name"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SPECS:
        render(s)
