# semantic-segmentation

Each sample, the brain tumor of human, is represented by a tensor of size = 4 x H x W x D, where the four 3D tensor represents the a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes of tumor scan. You are expected to predict the tumor segmentation at voxel level, i.e. the output dimension for each sample would be H x W x D. 
Each sample does not have the fixed dimension - H, W, and D, so our are model handles this circumstance accordingly. 

## train.py
This file creates and trains our model according to specified number of epochs and learning rate.
This file resizes the input images to [128,160,128] and pun into the model.
The model outputs [N,4,128,160,128] and resized the image back again to match the orginal dimension, [N,4,H,W,D].
Then it calculates the loss using softmax and dice loss, and backpropagates it to train the model.

## predict.py
This file imports the trained model defined by
```bash
checkpoint = torch.load('trained_model.npy')
```
and outputs prediction using the test dataset. 


## utils.py
This file converts the predictions into the required format for kaggle submission.
