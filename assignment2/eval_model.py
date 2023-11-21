import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
import numpy as np
import imageio
from PIL import Image, ImageDraw
from utils import *
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--vis_freq', default=1000, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int) #5000
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=2.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)
    
    return images, mesh
def image_plot(feed_dict,step):
        image=feed_dict['images']
        image=image.squeeze(0).cpu().numpy()
        # print(image.shape)
        image = cv2.resize(image, (256, 256))
        plt.imsave(f"{step}img.jpg",image)
def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'Graphs/eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics
def render_360_mesh(mesh, image_size=256, output_path='Images/mesh/q_1-1.gif', device=None, dist=3):
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
def render_mesh(mesh, output_path):
    device=mesh.device
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0), device = device)
    textures = textures * torch.tensor([0.7, 0.7, 1], device = device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    render_360_mesh(mesh.detach(), output_path=output_path, device=device, dist=1.5)

def render_360_pc(point_cloud, image_size=256, output_path='Images/point_cloud/q_5-1_pc1.gif', device=None):
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
    
def render_pc(points, output_path):
    device=points.device
    points = points.detach()[0]
    color = (points - points.min()) / (points.max() - points.min())
    pc = pytorch3d.structures.Pointclouds(points=[points], features=[color]).to(device)
    render_360_pc(pc, output_path=output_path, device=device)
def render_vox(vox, output_path):
    device=vox.device
    mesh = pytorch3d.ops.cubify(vox, thresh=0.5, device=device)
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0))
    textures = textures * torch.tensor([0.7, 0.7, 1], device=device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    render_360_mesh(mesh, output_path=output_path, device=device)

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth') 
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        # TODO:
        img_step = step % 50 
        num = step // 50 
        if img_step == 0:
            if args.type == "vox":
                # image_plot(feed_dict,num)
                render_vox(predictions[0], output_path=f'Images/vox/q_2-1-pred-{num}.gif')
                vox_gt = feed_dict['voxels'].to(args.device)
                render_vox(vox_gt[0], output_path=f'Images/vox/q_2-1-gt-{num}.gif')
            if args.type == "point":
                render_pc(predictions, output_path=f'Images/point_cloud/q_2-2-pred-{num}.gif')
                gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
                render_pc(gt_points, output_path=f'Images/point_cloud/q_2-2--gt-{num}.gif')
            if args.type == "mesh":
                render_mesh(predictions, output_path=f'Images/mesh/q_2-3-pred-{num}.gif')
                render_mesh(mesh_gt.to(args.device), output_path=f'Images/mesh/q_2-3-gt-{num}.gif')
      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
