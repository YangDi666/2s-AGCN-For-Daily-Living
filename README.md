# 2s-AGCN
Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition in CVPR19

# Note

PyTorch version >=Pytorch0.4. \


# Data Preparation

 - `mkdir data`
 - Download the raw data from [Smarthome](https://project.inria.fr/toyotasmarthome/). Then put them under the data directory:
 
        -data\  
          -smarthome_raw\  
            -smarthome_skeletons\
             - ... .json
               ... .json
               ...
            
- For other datasets: [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) / [Skeleton-Kinetics](https://github.com/yysijie/st-gcn)

- Preprocess the data with
  
    `cd data_gen`

    `python smarthome_gendata.py`
  

- Generate the bone data with: 
    
    `python gen_bone_data.py`
     
# Training & Testing

Change the config file depending on what you want.


    `python main.py --config ./config/smarthome-cross-subject/train_joint.yaml`

    `python main.py --config ./config/smarthome-cross-subject/train_bone.yaml`
To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer. 

    `python main.py --config ./config/smarthome-cross-subject/test_joint.yaml`

    `python main.py --config ./config/smarthome-cross-subject/test_bone.yaml`
    
There are 3 [pre-trained models](https://drive.google.com/drive/folders/18S_GjkZXthEv0Hv7JBzUN4Yo0env9bZN?usp=sharing) for 3 versions of skeletons. For testing them, change the model path in the config files.

Then combine the generated scores with: 

    `python ensemble.py --datasets smarthome/xsub`

For evaluation:

    `python evaluation.py runs/smarthome_cs_agcn_test_joint_right.txt 31`
     
# Reference

    @inproceedings{2sagcn2019cvpr,  
          title     = {Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},  
          author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
          booktitle = {CVPR},  
          year      = {2019},  
    }
    
    @article{shi_skeleton-based_2019,
        title = {Skeleton-{Based} {Action} {Recognition} with {Multi}-{Stream} {Adaptive} {Graph} {Convolutional} {Networks}},
        journal = {arXiv:1912.06971 [cs]},
        author = {Shi, Lei and Zhang, Yifan and Cheng, Jian and LU, Hanqing},
        month = dec,
        year = {2019},
	}

    @InProceedings{Das_2019_ICCV,
        author = {Das, Srijan and Dai, Rui and Koperski, Michal and Minciullo, Luca and Garattoni, Lorenzo and Bremond, Francois and Francesca, Gianpiero},
        title = {Toyota Smarthome: Real-World Activities of Daily Living},
        booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
        month = {October},
        year = {2019}
        }
