**Please use ChebyShevTrain.py for training model.**
__This is a Python code that performs registration of medical images using a technique ChebyShev Conv HyperNet. The code imports several libraries such as os, numpy, neurite, voxelmorph, tensorflow, nibabel, and others to perform various tasks such as reading and processing image files, defining and training deep learning models, and computing metrics for evaluating the performance of the model.__

The code first defines a function to read NIfTI image files and load the image data. It then splits the image data into training, validation, and test sets. The next function defines a generator that selects random pairs of images for each training iteration to align. The generator also computes the gradient loss on the predicted deformation.

The a Chebyshev hypernetwork model for registration and compiles it using the Adam optimizer with a learning rate of 1e-4. The model is trained using the hypermorph generator for 1500 epochs and 100 steps per epoch.

Finally,computes and prints the dice score and target registration error for each image in the test set using the computed slices from the model. These metrics are commonly used to evaluate the accuracy of image registration algorithms.

Loss function and base network can also be refered to voxelmorph.
To use the VoxelMorph library, either clone this repository and install the requirements listed in setup.py or install directly with pip.

pip install voxelmorph
VoxelMorph have several voxelmorph tutorials
the main VoxelMorph tutorial explains VoxelMorph and Learning-based Registration.
a tutorial on training vxm on OASIS data, which we processed and released for free for HyperMorph.
an additional small tutorial on warping annotations together with images
another tutorial on template (atlas) construction with VoxelMorph.
