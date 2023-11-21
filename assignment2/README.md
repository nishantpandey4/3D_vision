# CMSC848F Assignment 2: Single View to 3D
# Author: Nishant Awdeshkumar Pandey 


## 1. Exploring loss functions (15 points)
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid (5 points)

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 
The output can be found in Images folder.
### 1.2. Fitting a point cloud (5 points)

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 
The output can be found in Images folder.

### 1.3. Fitting a mesh (5 points)

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 
The output can be found in Images folder.

## 2. Reconstructing 3D from single view (85 points)

### 2.1. Image to voxel grid (20 points)

# References for the decoders are in the model.py file 

Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint`

Every 50th image is visualized.
The gt and prediction outputs are in the 'Images/vox' folder
The graphs are stored in the Graphs folder.
### 2.2. Image to point cloud (20 points)

Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point' --load_checkpoint`

Every 50th image is visualized.
The gt and prediction outputs are in the 'Images/point_cloud' folder
The graphs are stored in the Graphs folder.

### 2.3. Image to mesh (20 points)

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline, feel free to tune the hyperparameters as per your need. 

After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py --type 'mesh' --load_checkpoint`

Every 50th image is visualized.
The gt and prediction outputs are in the 'Images/mesh' folder
The graphs are stored in the Graphs folder.

### 2.4. Quantitative comparisions(10 points)

For evaluating you can run:
`python eval_model.py --type voxel|mesh|point --load_checkpoint`

Webpage consists the explanation for the same.

### 2.5. Analyse effects of hyperparms variations (5 points)

I have varied n_points and while keeping the other parameters same to get the output. 

### 2.6. Interpret your model (10 points)

Run the command below, to find to outputs in the 'Images/Part6' folder. 
`python interpret_model.py --type 'point' --load_checkpoint`



