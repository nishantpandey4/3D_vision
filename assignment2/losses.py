import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing
# define losses here

def voxel_loss(voxel_src, voxel_tgt):
    # voxel_src: b x h x w x d (logits before sigmoid)
    # voxel_tgt: b x h x w x d (binary target)

    # Compute binary cross entropy with logits
    sigmoid = torch.nn.Sigmoid()
    func = torch.nn.BCELoss() 
    loss = func(sigmoid(voxel_src),voxel_tgt)
    return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	knn_s_t=knn_points(point_cloud_src, point_cloud_tgt, K=1)
	dist_st = knn_s_t.dists[..., 0].sum(1)
	knn_t_s=knn_points(point_cloud_tgt, point_cloud_src, K=1)
	dist_ts = knn_t_s.dists[..., 0].sum(1)
	loss_chamfer = torch.mean(dist_st+dist_ts)
	
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method = 'uniform')
	# implement laplacian smoothening loss
	return loss_laplacian