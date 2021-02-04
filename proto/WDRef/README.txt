- id_WDRef.mat
    The index of identities in the WDRef database. Same index means the same person.

- id_lfw.mat
    The index of identities in the LFW database. Same index means the same person.

- lbp_WDRef.mat
    LBP features of WDRef database. Each row contains the LBP feature of one image.
 The feature dimension is 5900 and the number of image is 71846

- lbp_lfw.mat
    LBP feature of LFW database. Each row contains the LBP feature of one image.

- le_WDRef.mat
    LE features of WDRef database. Each row contains the LE feature of one image.
The feature dimension is 20736 and the number of image is 71846

- le_lfw.mat
    LE feature of LFW database. Each row contains the LE feature of one image.

- pairlist_lfw.mat
    3000 intra personal pairs and 3000 extra personal pairs of LFW database. For 
example, pairlist_lfw.IntraPersonalPair(1, :) = [56, 59]. It means lbp_lfw(56, :) 
and lbp_lfw(59, :) are features of the same person. These pairs are the same as 
benchmark of LFW. It could be used in the test.

- imagelist_lfw.mat
    The original image file name of each feature.

Please cite as:
Dong Chen, Xudong Cao, Liwei Wang, Fang Wen, Jian Sun. Bayesian Face Revisited: 
A Joint Formulation. European Conference on Computer Vision 2012.