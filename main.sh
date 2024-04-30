#!/bin/bash
# @Author: Chaowei Wu
# @Email:  chaoweiwu@aikucun.com
# @Date:   2023-02-14 15:03:06
# @Last Modified by:   Chaowei Wu
# @Last Modified time: 2023-07-29 11:16:10
# @Description: 使用 8001 端口


d_current=$1
cd /data/fred/retrieval_google/retrieval_google
bash dev.sh $d_current > logs/${d_current}.log 
