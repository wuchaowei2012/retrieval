#!/bin/bash
# @Author: Chaowei Wu
# @Email:  chaoweiwu@aikucun.com
# @Date:   2023-02-14 15:03:06
# @Last Modified by:   Chaowei Wu
# @Last Modified time: 2023-07-29 11:16:10
# @Description: 使用 8001 端口

set +x
echo "begin time:`date`"

# d_current=`date  "+%Y%m%d%H%M"`
d_current=2024042720

# d_current=$1
echo ${d_current}

######## step1  使用脚本从obs 拉取数据到本机
ROOT_PATH="/data/fred/retrieval_google/retrieval_google"
cd ${ROOT_PATH}

CANDiDATE_DIR="${ROOT_PATH}/data/candidate/${d_current}"

# /data/fred/anaconda3/envs/bertpy310/bin/
python pyobs_tools_fast.py 'akc-bigdata' akc_alg_hour.db/alg_all_product240427/"${d_current:0:8}"/${d_current} ${CANDiDATE_DIR} "True"  part-r-0



exit 0
######## step2  清除数据
DATA_DIR=${ROOT_PATH}/data
OUTPUT_DIR="${DATA_DIR}/output"
INPUT_DIR="${DATA_DIR}/input"

VOC_DIR=${INPUT_DIR}/vocs/                   # cat 类型变量的 vocabulary
BIN_DIR=${INPUT_DIR}/bins                    # bin 类型变量的 边界信息

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint2"   # TF checkpoint
EXPORT_DIR="${OUTPUT_DIR}/export"            # 导出模型

##删除export路径下模型历史文件，不然无法上传obs
rm -rf ${EXPORT_DIR}
mkdir ${EXPORT_DIR}

function logStop() {
   sync
   sleep 1
}

logStop

######## step5 训练模型

# train
# conda activate tf1.15
echo "-----------train model using detail data-------------"

# evaluation /data/fred/anaconda3/envs/aikucun/bin/
python main_func.py \
  --model_dir=$CHECKPOINT_DIR \
  --vocabulary_path=$VOC_DIR \
  --bins_path=$BIN_DIR \
  --test_file=$TRAIN_DIR  \
  --export_dir=$EXPORT_DIR  \
  --dcurrent=$d_current \
  --train=False

python main_func.py \
  --model_dir=$CHECKPOINT_DIR \
  --vocabulary_path=$VOC_DIR \
  --bins_path=$BIN_DIR \
  --train_file=$TRAIN_DIR \
  --export_dir=$EXPORT_DIR  \
  --dcurrent=$d_current \
  --train=True

