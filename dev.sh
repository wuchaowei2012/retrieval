
#!/bin/bash
# d_current=`date  "+%Y%m%d%H%M"`
# d_current=2024042720
d_current=$1

ROOT_PATH="/data/fred/retrieval_google/retrieval_google"
cd ${ROOT_PATH}

TRAINSETPATH="${ROOT_PATH}/data/trainset"

rm -rf $TRAINSETPATH
mkdir -p $TRAINSETPATH

for i in {1..3}
do   
   hours_ago=`date -d "${i} hours ago" "+%Y%m%d%H"`
   echo "hours_ago:${hours_ago}"

   last_two_digits="${hours_ago: -2}"
    if ((10#$last_two_digits >= 0 && 10#$last_two_digits < 8)); then
        echo "最后两位数字大于 0 且小于 8"
    else
        TRAINSET_DIR="${TRAINSETPATH}/${hours_ago}"
        echo ${hours_ago}
        python pyobs_tools_fast.py 'akc-bigdata' akc_alg_hour.db/positive_dataset_240427/"${hours_ago:0:8}"/${hours_ago} ${TRAINSET_DIR} "True"  part-r-
        # echo "最后两位数字不符合条件"
    fi
done


echo "begin download candidate files"
CANDiDATE_DIR="${ROOT_PATH}/data/candidate/${d_current}"


python pyobs_tools_fast.py 'akc-bigdata' akc_alg_hour.db/alg_all_product240427/"${d_current:0:8}"/${d_current} ${CANDiDATE_DIR} "True"  part-r-0

# 训练模型
python retrieval-model-dev2.py $TRAINSETPATH $CANDiDATE_DIR