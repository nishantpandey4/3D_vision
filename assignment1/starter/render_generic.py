"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle
import imageio

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_plant(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders three different point clouds of a plant. 
    The first point cloud is the point cloud of the plant in the first image.
    The second point cloud is the point cloud of the plant in the second image.
    The third point cloud is the point cloud of the plant made from the union of the
    first and second image.
    """
    if device is None:
        device = get_device()
        # print(device)
    data=load_rgbd_data()
    #Extracts the first image
    image=torch.tensor(data.get("rgb1"),device=device)
    mask=torch.tensor(data.get("mask1"))
    depth=torch.tensor(data.get("depth1"))
    camera=data.get("cameras1")
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    
    points,rgb=unproject_depth_image(image, mask, depth, camera)
    verts = (points).unsqueeze(0)
    rgb = (rgb).unsqueeze(0)
    
    theta_degrees=2
    angles = np.linspace(1, 360, theta_degrees,endpoint=False)
    my_images1=[]
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    """
    Renders the first point cloud
    """
    for angle in angles:
        # print(angle)
        # point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
        R, T = pytorch3d.renderer.look_at_view_transform(6, 0, angle,up=((0, -1, 0),))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255)
        my_images1.append(rend)
        
    #Extracts the second image
    image=torch.tensor(data.get("rgb2"),device=device)
    mask=torch.tensor(data.get("mask2"))
    depth=torch.tensor(data.get("depth2"))
    camera=data.get("cameras2")
    points1,rgb1=unproject_depth_image(image, mask, depth, camera)
    verts1 = (points1).unsqueeze(0)
    rgb1 = (rgb1).unsqueeze(0)
    theta_degrees=2
    angles = np.linspace(1, 360, theta_degrees, endpoint=False)
    my_images2=[]
    point_cloud = pytorch3d.structures.Pointclouds(points=verts1, features=rgb1)
    """
    Renders the second point cloud
    """
    for angle in angles:
        # print(angle)
        # point_cloud = pytorch3d.structures.Pointclouds(points=verts1, features=rgb1)
        R, T = pytorch3d.renderer.look_at_view_transform(6, 0, angle,up=((0, -1, 0),))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255)
        my_images2.append(rend)
    
    """
    Combines the first and second point cloud
    """
    # Concatenate along dim=1 to combine the point clouds in the same batch.
    points2 = torch.cat((points.unsqueeze(0), points1.unsqueeze(0)), dim=1).squeeze(0)
    
    # Concatenate along dim=1 to combine the rgb features in the same batch.
    rgb2 = torch.cat((rgb, rgb1), dim=1).squeeze(0)
    
    verts2 = points2.unsqueeze(0)
    rgb2=rgb2.unsqueeze(0)
    theta_degrees=2
    angles = np.linspace(1, 360, theta_degrees,endpoint=False)
    my_images3=[]
    point_cloud = pytorch3d.structures.Pointclouds(points=verts2, features=rgb2)
    """
    Renders the third point cloud
    """
    for angle in angles:
        # print(angle)
        # point_cloud = pytorch3d.structures.Pointclouds(points=verts2, features=rgb2)
        R, T = pytorch3d.renderer.look_at_view_transform(6, 0, angle,up=((0, -1, 0),))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255)
        my_images3.append(rend)
    
    return np.array(my_images1, dtype=np.uint8),np.array(my_images2, dtype=np.uint8),np.array(my_images3, dtype=np.uint8)


def render_torus(image_size=256, num_samples=2000, device=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2*np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta,indexing='xy')
    R=torch.tensor(3)
    r=torch.tensor(2)
    #Parametric equation of a torus.
    x = (R+r*torch.cos(Theta) )* torch.cos(Phi)
    y = (R+r*torch.cos(Theta) )* torch.sin(Phi)
    z = r*torch.sin(Theta) 

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)
    theta_degrees=120
    angles = np.linspace(1, 360, theta_degrees, endpoint=False)
    my_images=[]
    
    for angle in angles:
        
        R, T = pytorch3d.renderer.look_at_view_transform(10, 15, angle)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,T=T, device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(sphere_point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3] 
        rend = (rend * 255)
        my_images.append(rend)
    return  np.array(my_images, dtype=np.uint8)


def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    """
    Renders a torus mesh using implicit equation.
    """
    if device is None:
        device = get_device()
    min_value = -5.6
    max_value = 5.6
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3,indexing='xy')
    R=torch.tensor(3)
    r=torch.tensor(2)
    #Implicit equation of a torus.
    voxels = (X**2 + Y**2 + Z**2 +R**2-r**2)**2-torch.tensor(4)*(R**2)*(X**2 + Y**2)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    theta_degrees=50
    angles = np.linspace(1, 360, theta_degrees, endpoint=False)
    my_images=[]
    
    for angle in angles:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=0, azim=angle)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend=rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        rend = (rend * 255)
        my_images.append(rend)
    return np.array(my_images, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=2000)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image1,image2,image3 = render_plant(image_size=args.image_size)
        imageio.mimsave('GIF/plant1.gif',image1, duration=10,loop=0)
    
        imageio.mimsave('GIF/plant2.gif',image2, duration=10,loop=0)
    
        imageio.mimsave('GIF/plant3.gif',image3, duration=10,loop=0)
    elif args.render == "parametric":
        image = render_torus(image_size=args.image_size, num_samples=args.num_samples)
        imageio.mimsave('GIF/torus.gif',image, duration=100,loop=0)
    elif args.render == "implicit":
        image = render_torus_mesh(image_size=args.image_size)
        imageio.mimsave('GIF/implicit_torus.gif',image, duration=100,loop=0)
    else:
        raise Exception("Did not understand {}".format(args.render))


