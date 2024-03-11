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

    def __call__(self, points__):
        points = torch.abs(points__ - self.center)
        points_ = points[:, :, :, :2]
        elevations_ = points[:, :, :, 2]
        dists = torch.norm(points - torch.tensor([0, 0, self.half_height]), dim=-1)
        dists_ = torch.norm(points_, dim=-1)
        dists__ = torch.where(elevations_ > self.half_height, dists, dists_)
        print("!!!", points__.shape, dists__.shape)
        return dists__ - self.radius


class UnionSDF:
    def __init__(self, shape_aa, shape_bb):
        self.shape_aa = shape_aa
        self.shape_bb = shape_bb
        print("!!!!", shape_aa.aabb, shape_bb.aabb)
        self.aabb = torch.zeros_like(shape_aa.aabb)
        assert self.aabb.shape == (3, 2)
        self.aabb[:, 0] = torch.minimum(shape_aa.aabb[:, 0], shape_bb.aabb[:, 0])
        self.aabb[:, 1] = torch.maximum(shape_aa.aabb[:, 1], shape_bb.aabb[:, 1])

    def __call__(self, points):
        dists_aa = self.shape_aa(points)
        dists_bb = self.shape_bb(points)
        return torch.minimum(dists_aa, dists_bb)


os.makedirs("out", exist_ok=True)

# define a sphere
s_x = 0.0
s_y = 0.0
s_z = 0.0
radius = 0.5
height = 1.5
shape_sphere = SphereSDF(torch.tensor([s_x, s_y, s_z]), radius * 1.5)
shape_capsule = CapsuleSDF(torch.tensor([s_x, s_y, s_z]), radius, height)
shape = UnionSDF(shape_sphere, shape_capsule)

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
