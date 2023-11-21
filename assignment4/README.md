## Assignment 4 - CMSC848f
# Author - Nishant Pandey  

## Q1. Classification Model (40 points)

Complete model initialization and prediction in `train.py` and `eval_cls.py`. Run `python train.py --task cls` to train the model, and `python eva_cls.py` for evaluation. Check out the arguments and feel free to modify them as you want.

Results : `output/cls`

## Q2. Segmentation Model (40 points) 

Complete model initialization and prediction in `train.py` and `eval_seg.py`. Run `python train.py --task seg` to train the model. Running `python eval_seg.py` will save two gif's, one for ground truth and the other for model prediction. 

Results : `output/seg`

Set epochs to 20 to save time the test accuracy drops but is fine.


## Q3. Robustness Analysis (20 points) 

1. You can rotate the input point clouds by certain degrees and report how much the accuracy falls. To get the output for 
classification use `eval_cls.py --exp_num 1`.  To get the output for segmentation use `eval_seg.py --exp_num 1`.
Change the value of `rot` in function `rotate_point_cloud` in `utils.py`. To obtain desired degree of rotation. 

2. You can input a different number of points points per object (modify `--num_points` when evaluating models in `eval_cls.py` and `eval_seg.py`)