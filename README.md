# Epic_kitchens_affordances

The EPIC-Affordance dataset is a new dataset build on the Epic Kitchens 100 and Epic Kitchens VISOR. It contains **automatic annotations** generated by the intersection of both datasets. On one hand, we use the narration annotations of the Epic Kitchens 100 to obtain the semantics of the interaction (e.g "cut onion"). Then, we use the masks provided by EPIC VISOR to discover the location of that interaction, placed in the center of the intersection between the respective hand/glove and the interacting object. This provides an understanding about where the interaction occurs at that time step.

[P01_01_frame_0000003682](https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000003682.jpg)

<img width="964" alt="java 8 and prio java 8  interaction_img" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000003682.jpg">

In a second stage, using Structure from Motion algorithms (COLMAP), we obtain the camera pose the global localization of the interaction in the 3D space. Running this for all the frames in the kitchen where an interaction occur, we obtain a historical distribution of all the taken actions in that kitchen.

The dataset is !
