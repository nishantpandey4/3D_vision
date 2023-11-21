import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch
import numpy as np
import mcubes
import imageio
import pytorch3d
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from utils import get_device, get_mesh_renderer, get_points_renderer


def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser

def render_360_mesh(mesh, image_size=256, output_path='images/q_1-1.gif', device=None, dist=3):
    renderer = get_mesh_renderer(image_size=image_size)


    num_views=12
    angles = np.linspace(-180, 180, num_views, endpoint=False)
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=15,
        azim=angles[i],
    )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
        lights = pytorch3d.renderer.PointLights(location=[[1, 1, 3]], device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()

        image = Image.fromarray((rend * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, duration=100,loop=0)
def fit_mesh(mesh_src, mesh_tgt, args, device):
    image_size=256
    device=None
    
    if device is None:
        device = get_device()
    color=torch.tensor([0.3, 0.4, 1]).to(device)
    renderer = get_mesh_renderer(image_size=image_size)

    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')
    
    vertices = mesh_src.verts_packed()
    faces = mesh_src.faces_packed()
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * color  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    render_360_mesh(mesh.detach(), output_path='Images/mesh_pred.gif', device=device, dist=1.5)

    vertices = mesh_tgt.verts_packed()
    faces = mesh_tgt.faces_packed()
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * color  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    render_360_mesh(mesh.detach(), output_path='Images/mesh_gt.gif', device=device, dist=1.5)

def render_360_pc(point_cloud, image_size=256, output_path='Images/q_5-1_pc1.gif', device=None):
    renderer = get_points_renderer(image_size=image_size)
 
    num_views = 12
    angles = np.linspace(-180, 180, num_views, endpoint=False)
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
        dist=1,
        elev=0,
        azim=angles[i],
    )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
        
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()

        image = Image.fromarray((rend * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, duration=100,loop=0)
def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    device=None
    if device is None:
        device = get_device()
    background_color=(1, 1, 1)
    image_size=256
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = pointclouds_src.detach().cpu().numpy()  # Taking the first item in the batch
    # print(point_cloud)
    verts = torch.tensor(point_cloud[0]).to(device).unsqueeze(0)
    device = verts.device
    rgb = (torch.ones_like(verts) * torch.tensor([1.0, 0.0, 0.0], device=device))
    point_cloud = pytorch3d.structures.Pointclouds(points=verts,features=rgb)
    
    render_360_pc(point_cloud, output_path='Images/pc_pred.gif', device=device)

    point_cloud1 = pointclouds_tgt.detach().cpu().numpy()  # Taking the first item in the batch
    # print(point_cloud)
    verts = torch.tensor(point_cloud1[0]).to(device).unsqueeze(0)
    device = verts.device
    rgb = (torch.ones_like(verts) * torch.tensor([1.0, 0.0, 0.0], device=device))
    point_cloud1 = pytorch3d.structures.Pointclouds(points=verts,features=rgb)
    
    render_360_pc(point_cloud1, output_path='Images/pc_gt.gif', device=device)

def fit_voxel(voxels_src, voxels_tgt, args):
    voxel_size=32
    device=None
    min_value = -1.8
    max_value = 1.8
    image_size=256
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    if device is None:
        device = get_device()
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    voxels_src = voxels_src[0].detach().cpu().numpy()  # Taking the first item in the batch

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels_src), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    # mesh_vis = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
    # mesh_vis.show()
    # lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    # renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=0)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    # rend = renderer(mesh, cameras=cameras, lights=lights)
    # image=rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    render_360_mesh(mesh, output_path='Images/vox_pred.gif', device=device)

    voxels_tgt = voxels_tgt[0].detach().cpu().numpy()  # Taking the first item in the batch

    vertices1, faces1 = mcubes.marching_cubes(mcubes.smooth(voxels_tgt), isovalue=0)
    vertices1 = torch.tensor(vertices1).float()
    faces1 = torch.tensor(faces1.astype(int))
    vertices1 = (vertices1/ voxel_size) * (max_value - min_value) + min_value

    textures1 = (vertices1 - vertices1.min()) / (vertices1.max() - vertices1.min())
    textures1 = pytorch3d.renderer.TexturesVertex(vertices1.unsqueeze(0))
    mesh1 = pytorch3d.structures.Meshes([vertices1], [faces1], textures=textures1).to(device)
    # mesh_vis = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
    # mesh_vis.show()
    # lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4]], device=device,)
    # renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=0)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    # rend1 = renderer(mesh1, cameras=cameras, lights=lights)
    # image1=rend1[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    # plt.imsave("Images/tgt_image.jpg", image1)
    # Concatenate images along the width
    # combined_image = np.concatenate((image, image1), axis=1)
    # plt.imsave("Images/Voxel_visual.jpg", combined_image)
    render_360_mesh(mesh1, output_path='Images/vox_gt.gif', device=device)




def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args, device=args.device)        



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
    