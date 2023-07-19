# Epic-Aff Dataset

This is the dataset introduced on the ICCV 2023 paper **Multi-label affordance mapping from egocentric vision**, by Lorenzo Mur-Labadia, Ruben Martinez-Cantin and Josechu Guerrero Campo. Please, do not hesitate to ask any question on the following mail *lmur@unizar.es* :✉️:

## Dataset creation: automatic annotations

The EPIC-Aff dataset is a new dataset build on the Epic Kitchens 100 and Epic Kitchens VISOR, containing **automatic annotations with multi-label segmentation masks for the interaction hotspots**,  generated by the intersection of both datasets. We provide **38,335** images in two different versions of the dataset (easy-EPIC Aff with 20 classes and complex-EPIC Aff with 50 classes). The annotations represent the hotspots in the space with an affordable action, extracted from the past interactions performed on that region and the actual scene context (present objects). Please, refer to the paper for more information 

The total size of the dataset is 15 GB, which we have divided in the different data type. We also provide a example sequence on the PO3_EPIC_100_Example. The full dataset can be downloaded [here](https://zenodo.org/record/8162678)

-**Images** :📸: : we already provide the images extracted from the videos of EPIC-100 Kitchens in 480x854 of resolution. This avoids download the approximate 700 GB of that dense dataset. [link](https://zenodo.org/record/8162678/files/EPIC_Aff_images.zip?download=1)

-**Annotations in 3D** :📝: : in a pickle format, we provide a dictionary with the Colmap data (camera pose, camera intrinsics and keypoints), the distribution of the interacting objects, the annotation of the interaction and the distribution of the neutral objects. We encourage to the research community to use this data to develop new tasks like goal path planning. [link](https://zenodo.org/record/8162678/files/EPIC_Aff_3D_output.zip?download=1)

-**Affordance annotations in the 2D** :📝:: we already run the project_from_3D_to_2D.py for all the sequences in order to provide a pickle dictionary with the location of the interaction points for the afforded-actions. We provide two versions of the dataset:
    - Easy EPIC-Aff (20 classes): [link](https://zenodo.org/record/8162678/files/EPIC_Aff_20_classes_2d_output_labels.zip?download=1)
    - Complex EPIC-Aff (50 classes): [link](https://zenodo.org/record/8162678/files/EPIC_Aff_50_classes_2d_output_labels.zip?download=1)

-**VISORs masks** :🎭:: the semantic mask wit the active objets, which we consider dynamic. In order to obtain the dynamic masks for COLMAP, we select the dynamic and static objects. [link](https://zenodo.org/record/8162678/files/EPIC_Aff_masks_from_VISOR.zip?download=1)

We detail the procedure for extracting multi-label affordance regions.

### 1. Detect the spatial localization of the interaction

On one hand, we use the narration annotations of the Epic Kitchens 100 to obtain the semantics of the interaction (e.g "cut onion"). Then, we use the masks provided by EPIC VISOR to discover the location of that interaction, placed in the center of the intersection between the respective hand/glove and the interacting object. This provides an understanding about where the interaction occurs at that time step.

<p align="center" width="100%">
    <img width="32%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000003682.jpg"> 
    <img width="32%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000019463.jpg"> 
    <img width="32%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000049183.jpg"> 
</p>
<p align="center" width="100%">
    <img width="32%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000049183.jpg"> 
    <img width="32%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_02_frame_0000000946.jpg"> 
    <img width="32%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_02_frame_0000005376.jpg"> 
</p>

### 2. Leverage all to the 3D

In a second stage, using Structure from Motion algorithms (COLMAP), we get the camera pose and the global localization of the interaction in the 3D space. This creates a historical distribution of all the taken actions in that environment, cross-linking alng different episodes. In the following images, we show in blue the different camera poses, in grey the Colmap keypoints and the different locations where the interactions occur. For each specific physical kitchen, we accumulated all the EPIC videos where the agent interacted. Note that for some sequences, the EPIC-50 and EPIC-100 was different, while in other it was the same environment.

<p align="center" width="100%">
    <img width="47%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/Screenshot%20from%202022-12-14%2016-28-24.png"> 
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/Screenshot%20from%202022-12-13%2010-31-56.png"> 
</p>

This created a 3D representation with all the past interactions performed in that environment.

### 3. Reproject the 3D to the 2D to obtain the affordances.

Using the camera intrinsic matrix and the camera pose provided in the "3D_output" directories, we reproject all the past interactions by running *"project_from_3D_to_2D.py"*. Since the affordances are all the possible actions for the agent depending on the context, we filter the past interactions by the current distribution of the objects in each time-step. For that, we use the VISOR annotations for the active objets and we assume a constant distribution of passive objcts (cupboard, oven, hob, fridge) since its distribution did not change with time. For example, although the VISOR annotation does not detect any "active cupboard", if we have opened a cupboard in the past in that location, it means that there is a cupboard innactive. Therefore, we should detect that past interaction as a affordance, since it is a possible action associated to that 3D region.

We show some images of different affordances. Each point represents the location of a past interaction whose interacting objects are present.

<p align="center" width="100%">
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_05_frame_0000111070.png"> 
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_12_frame_0000008119.png"> 
</p>

Finally, we apply a Gaussian heatmap for each afforded actions in order to create a potential interaction region. We show respectively: takeable, insertable, cuttable and driable. Note that in inference, we assume a possitive affordance label when gaussian heat map is greater than 0.25.

<p align="center" width="100%">
    <img width="24%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_02_frame_0000016034%20take.png"> 
    <img width="24%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_02_frame_0000033785%20insert.png"> 
    <img width="24%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_02_frame_0000065888%20cut.png"> 
    <img width="24%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_04_frame_0000006974%20dry.png"> 
</p>

*Note*: the files in the *2D_output_labels* directories only contain the pixel points with the affordances and its semantic labels. When you run data.py, in the dataloader we incorporate a function to obtain the Gaussian heatmaps in an efficient way. This avoids to load the *N* masks.

## Dataset pipeline
We also share the code for the dataset pipeline extraction, and we encourage the research community to apply in other scenarios. 
