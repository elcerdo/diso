import os
import torch
import torch.nn as nn
import trimesh
from diso import DiffMC
from diso import DiffDMC


# define a sphere SDF
class SphereSDF:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.aabb = torch.stack([center - radius, center + radius], dim=-1)

    def __call__(self, points):
        return torch.norm(points - self.center, dim=-1) - self.radius


class CapsuleSDF:
    def __init__(self, center, radius, height):
        self.center = center
        self.radius = radius
        self.half_height = height / 2
        delta = torch.tensor([self.radius, self.radius, self.half_height + self.radius])
        self.aabb = torch.stack([center - delta, center + delta], dim=-1)

    def __call__(self, points_):
        points = torch.abs(points_ - self.center)
        # xys = points[:, :, :, :2]
        # zzs = points[:, :, :, 2]
        # selection = zzs > self.height
        # print("$$$", selection.shape, selection.dtype, xys.shape, zzs.shape)
        dists = torch.norm(points - torch.tensor([0, 0, self.half_height]), dim=-1)
        # dists = (
        #     torch.select(
        #         selection,
        #         torch.norm(xys, dim=-1),
        #         torch.norm(points - torch.tensor([0, 0, self.height]), dim=-1),
        #     )
        #     - self.radius
        # )
        print("!!!", points.shape, dists.shape)
        return dists - self.radius


os.makedirs("out", exist_ok=True)

# define a sphere
s_x = 0.0
s_y = 0.0
s_z = 0.0
radius = 0.5
height = 1.5
# sphere = SphereSDF(torch.tensor([s_x, s_y, s_z]), radius)
shape = CapsuleSDF(torch.tensor([s_x, s_y, s_z]), radius, height)

# create the iso-surface extractor
diffmc = DiffMC(dtype=torch.float32).cuda()
diffdmc = DiffDMC(dtype=torch.float32).cuda()

# create a grid
dimX, dimY, dimZ = 64, 64, 64
grids = torch.stack(
    torch.meshgrid(
        torch.linspace(0, 1, dimX),
        torch.linspace(0, 1, dimY),
        torch.linspace(0, 1, dimZ),
        indexing="ij",
    ),
    dim=-1,
)
grids[..., 0] = grids[..., 0] * (shape.aabb[0, 1] - shape.aabb[0, 0]) + shape.aabb[0, 0]
grids[..., 1] = grids[..., 1] * (shape.aabb[1, 1] - shape.aabb[1, 0]) + shape.aabb[1, 0]
grids[..., 2] = grids[..., 2] * (shape.aabb[2, 1] - shape.aabb[2, 0]) + shape.aabb[2, 0]

# query the SDF input
sdf = shape(grids)
sdf = sdf.requires_grad_(True).cuda()
sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)

# randomly deform the grid
deform = torch.nn.Parameter(
    torch.rand(
        (sdf.shape[0], sdf.shape[1], sdf.shape[2], 3),
        dtype=torch.float32,
        device="cuda",
    ),
    requires_grad=True,
)

# DiffMC with random grid deformation
verts, faces = diffmc(sdf, 0.5 * torch.tanh(deform))
verts = verts.cpu() * (shape.aabb[:, 1] - shape.aabb[:, 0]) + shape.aabb[:, 0]
mesh = trimesh.Trimesh(
    vertices=verts.detach().cpu().numpy(),
    faces=faces.cpu().numpy(),
    process=False,
)
mesh.export("out/diffmc_sphere_w_deform.obj")

# DiffMC without grid deformation
verts, faces = diffmc(sdf, None)
verts = verts.cpu() * (shape.aabb[:, 1] - shape.aabb[:, 0]) + shape.aabb[:, 0]
mesh = trimesh.Trimesh(
    vertices=verts.detach().cpu().numpy(),
    faces=faces.cpu().numpy(),
    process=False,
)
mesh.export("out/diffmc_sphere_wo_deform.obj")

# DiffDMC with random grid deformation
verts, faces = diffdmc(sdf, 0.5 * torch.tanh(deform))
verts = verts.cpu() * (shape.aabb[:, 1] - shape.aabb[:, 0]) + shape.aabb[:, 0]
mesh = trimesh.Trimesh(
    vertices=verts.detach().cpu().numpy(),
    faces=faces.cpu().numpy(),
    process=False,
)
mesh.export("out/diffdmc_sphere_w_deform.obj")

# DiffDMC without grid deformation
verts, faces = diffdmc(sdf, None)
verts = verts.cpu() * (shape.aabb[:, 1] - shape.aabb[:, 0]) + shape.aabb[:, 0]
mesh = trimesh.Trimesh(
    vertices=verts.detach().cpu().numpy(),
    faces=faces.cpu().numpy(),
    process=False,
)
mesh.export("out/diffdmc_sphere_wo_deform.obj")

print("examples saved to out/")
