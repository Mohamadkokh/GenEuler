from geneuler.mesh import Mesh

# 1) Box (works already)
m1 = Mesh.from_bbox({"x": (-60, 60), "y": (0, 150), "z": (-120, 120)}, h=6.0, msh_path="out/box.msh")

xdmf, h5 = m1.save_xdmf("out/box")

# Floor & attachments
floor_idx = m1.floor_nodes(y=0.0)
attachments = [((0,130,-110), 6.0), ((20,140,0), 8.0)]
attach_idx = m1.attachment_nodes(attachments)

# 2) STL/OBJ: shell-only (auto, because it's not watertight)
m2 = Mesh.from_surface(
    "/Users/mohamadkokh/Desktop/GenEuler/Code/roundedBox1-40x40x40_boxOnly.stl",
    h=4.0,
    reconstruct_volume=True,      # you can keep True; preflight will switch to shell-only
    msh_path="out/chassis_shell.msh",
)

xdmf2, h52 = m2.save_xdmf("out/chassis_shell")

# 3) STEP/IGES: full volume tets (if solids present)
m3 = Mesh.from_surface(
    "/Users/mohamadkokh/Desktop/GenEuler/Code/roundedBox1-40x40x40_boxOnly.stl",   # <- this should be a CAD file, not the STL
    h=3.0,
    reconstruct_volume=True,
    msh_path="out/bracket.msh",
)

xdmf3, h53 = m3.save_xdmf("out/bracket")

# Save XDMF for FEM (only meaningful if tets exist)
if m3.tets.size:
    m3.save_xdmf("out/bracket")
