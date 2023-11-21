"""
Dolly zoom effect.
Usage:
    python -m starter.dolly_zoom 
"""

import argparse
import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer

def dolly_zoom(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    fovs = torch.linspace(5, 120, num_frames)
    fov1=torch.deg2rad(torch.tensor(5,dtype=torch.float32,device=device))
    #initial distance
    d1=50
    renders = []
    for fov in tqdm(fovs):
        fov2=torch.deg2rad(fov)
        #calculate new distance
        distance = d1 * (torch.tan(fov1 / 2) / torch.tan(fov2 / 2))  # TODO: change this.
        #calculate new T
        T = torch.tensor([[ 0,0, distance]],dtype=torch.float32,device=device)  # TODO: Change this.    
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov,T=T,device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)
        #update fov1 and d1
        fov1=fov2
        d1=distance
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration,loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output_file", type=str, default="GIF/dolly.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    dolly_zoom(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
