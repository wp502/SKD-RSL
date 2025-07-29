#!/bin/bash

# 断点续训 在原有命令的基础上 加3个运行参数，分别是 --is_resume --checkpoint_t --checkpoint_s
# 分别表示 “是否恢复（bool型，不带参数）”，“教师模型的断点（模型路径参数）”，“学生模型的断点（模型路径参数）”