"""
The following renders a 360 view of a cube.
Usage:
    python -m starter.render_cube 
"""
import argparse
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from starter.utils import get_device, get_mesh_renderer

def render_cube(image_size=512, color=[0, 1, 0.6], device=None,):
    """
    Renders a cube.

    Args:
        image_size (int, optional): _description_. Defaults to 512.
        color (list, optional): _description_. Defaults to [0, 1, 0.6].
        device (_type_, optional): _description_. Defaults to None.
    """
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)
    vertices = torch.tensor([
    [0, 0, 0],  # Vertex 0
    [0.5, 0, 0],  # Vertex 1
    [0.5, 0.5, 0],  # Vertex 2
    [0, 0.5, 0],  # Vertex 3
    [0, 0, 0.5],  # Vertex 4
    [0.5, 0, 0.5],  # Vertex 5
    [0.5, 0.5, 0.5],  # Vertex 6
    [0, 0.5, 0.5],  # Vertex 7
], dtype=torch.float32)

    faces = torch.tensor([
    [0, 3, 2], [0, 2, 1],  
    [4, 7, 6], [4, 6, 5],  
    [0, 4, 5], [0, 5, 1],  
    [1, 2, 6], [1, 6, 5],  
    [2, 3, 7], [2, 7, 6],  
    [0, 3, 7], [0, 7, 4],  
], dtype=torch.int64)
    vertices=vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
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
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=2.0, elev=0, azim=angle, degrees=True, eye=None, at=((0, 0, 0),), up=((0, 1, 0),), device=device)
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
    
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = render_cube(image_size=args.image_size)
    imageio.mimsave('GIF/cube.gif',image, duration=10,loop=0)
