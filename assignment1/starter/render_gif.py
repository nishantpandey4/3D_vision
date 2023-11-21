"""
This script renders a gif of a rotating cow.
Usage:
    python -m starter.render_gif 
"""
import argparse
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    cow_path="data/cow.obj", image_size=512, color=[0.5, 0.3, 1], device=None,
):
    """
    Renders 360 view of a mesh.
    """
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    theta_degrees=50
    angles = np.linspace(1, 360, theta_degrees, endpoint=False)
    my_images=[]
    for angle in angles:
    # Create a new mesh for each viewpoint
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)

        # Prepare the camera transformation for the current angle
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=5.0, elev=0, azim=angle, degrees=True, eye=None, at=((0, 0, 0),), up=((0, 1, 0),), device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        # Render the image for the current viewpoint
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend=rend.cpu().numpy()[0, ..., :3]
        rend = (rend * 255)
        my_images.append(rend) 
    return np.array(my_images, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    imageio.mimsave('GIF/my_gif.gif',image, duration=100,loop=0)
