Sunday 5/22
- To-do's:
    - Build supervised model to grasp where a human would
        - train it 
    - Create simulation environment for pickup

Monday 5/23
- To-do's:
    - Edit existing simulation
        - MVP: load in objects from contactDB & randomly pick points on object to pick 

Tuesday 5/24
- To-do's:
    - Finish data upload pipeline in SSL pybullet sim
        - Automate process to convert: ply -> obj -> urdf (with obj mesh)
    - Look into Huy's recs on unsupervised learning for robotics papers

Wedns 5/25
- To-do's:
    - Finish unsupervised training environment in pybullet 

Thurs 5/26
- To-do's:
    - Create dataloading pipeline 
        - Allow for type (use vs handoff) to alter dataloader
    - Go through pybullet docs and get a better understanding


Friday 5/27
- To-do's:
    - Solve dependancy errors and issues 
    - Research unsupervised learning 

Tues 5/31
- To-do's:
    - Create method to import the voxels given an object name
    - Read over ContactDB paper & get better understanding of their code
    - Follow up with Claire 

Wedns 6/1
- To-do's:
    - Create method to visualize the voxel representation in pybullet 
    - Upload Code to Github
    - Create doc with paper recs, progress, and project plans for Claire 

Thursday 6/2
- To-do's: 
    - Create method to stack diversenet predictions 
    - Create 3D CNN that predicts most likely single grasp 
    - Create unsupervised network 

Monday 6/13
- To-do's: 
    - Switch to pytorch lightning 
    - Implement loading from checkpoint
    - hyperparam finetuning
    - evaluation on test set 
    - visualization for val
    - accuracy plotting 
    - early stopping/ saving best checkpoint

Tuesday 6/14
- To-do's: 
    - Get tensorboard to work with visualizing predictions 

Wednsday 6/15
- to-do's:
    - Method to change predictions to grid of images 
    - Add n_images and save params to animate method
    - Create two viewing methods: 
        1) see all the 10 predictions in all different views compared to combined ground truth 
        2) see combined predictions compared to combined ground truth from all angles
    - Plot accuracy 
    - Plot images in grid 
    - Early stopping (Done)
    - Hyperparam finetuning
    - Visualize the combined groundtruths next to the val prediction at each step 

Thursday 6/16
- To-do's: 
    - Wedns tasks 
    - Make a 3dCNN model with lightning
    - refactor dataloader for 3dCNN
    - refactor train and model for 3dCNN


Backburner: 
----------------------
Tensorboard 
weights and biases
git commit all changes and when running experiment log git hash 
Ray (extra) -- parallelizer


-Try to overfit to only one example 
-Visualize outputs
-Fix logging 
-Fix checkpointing
- Maybe use adam optimizer 

- Talk to professor about reimplementing paper 