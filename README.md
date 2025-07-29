# SKD-RSL
Code for the paper: "Synergistic Knowledge Distillation via Reciprocal and Self Learning".

## Install
```bash
pip install -r requirements.txt
```
 
## Running
1.Download the dataset and place it in the data folder.
2.Start training with the instructions in the run_sh file.

## Explanation

The meaning explanations of the document are as follows:
1. evaluate_model.py: Evaluate the performance of a single model;
2. plot_t-sne.py: The plots are saved in './visualizations/plt_tsne_save/';
3. train_one_epoch_ddp.py: Train the code for one epoch in a distributed manner;
4. train_one_epoch.py: All methods train one epoch of code;
5. train_online.py: Code for online knowledge distillation comparison method;
6. train_ours_111.py: The code used for training SKD-RSL；
7. train_ours_ddp.py: Distributed training code, mainly used for training imagenet；
8. train_student.py: Offline Knowledge Distillation Comparison Method Code；
9. train_teacher.py: Used for training the baseline code；
10. utils_ddp.py: The utils used in the distributed training process；
11. utils_train_common_ddp.py: The running parameters and general functions during the distributed training process；
12. utils_train_common.py: Some running parameters and general functions during the training process；
13. utils.py: The utils used during the training process.
    
## Citation
Coming soon...
