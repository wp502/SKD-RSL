# SKD-RSL
Code for the paper: "Synergistic Knowledge Distillation via Reciprocal and Self Learning".

## Install
```bash
pip install -r requirements.txt
```
 
## Running
1.Download the dataset and place it in the data folder.

2.Start training with the instructions in the run_sh file.

The meaning explanations of the document are as follows:

1. evaluate_model.py: 评估单个模型性能
2. plot_t-sne.py: 【画图】【论文中用到】存放在'./visualizations/plt_tsne_save/'
3. train_one_epoch_ddp.py: 分布式训练一个epoch的代码
4. train_one_epoch.py: 所有方法训练一个epoch的代码
5. train_online.py: 在线知识蒸馏对比方法代码
6. train_ours_111.py: 训练FKD-FFBL的代码
7. train_ours_ddp.py: 分布式训练代码，主要用于训练imagenet
8. train_student.py: 离线知识蒸馏对比方法代码
9. train_teacher.py: 训练baseline代码
10. utils_ddp.py: 分布式训练过程中使用的utils
11. utils_train_common_ddp.py: 分布式训练过程中的运行参数、通用函数
12. utils_train_common.py: 训练过程中的一些运行参数、通用函数
13. utils.py: 训练过程中使用的utils
14. 方案二调参.xlsx: 所有实验结果
    
## Citation
Coming soon...
