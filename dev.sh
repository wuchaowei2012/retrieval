
#!/bin/bash

ROOT_PATH="/data/fred/retrieval_google/retrieval_google"
cd ${ROOT_PATH}


for i in {1..24}
do   
   hours_ago=`date -d "${i} hours ago" "+%Y%m%d%H"`
   
   last_two_digits="${hours_ago: -2}"
    if ((10#$last_two_digits >= 0 && 10#$last_two_digits < 8)); then
        echo "最后两位数字大于 0 且小于 8"
    else
        TRAINSET_DIR="${ROOT_PATH}/data/trainset/${hours_ago}"
        echo ${hours_ago}
        python pyobs_tools_fast.py 'akc-bigdata' akc_alg_hour.db/positive_dataset_240427/"${hours_ago:0:8}"/${hours_ago} ${TRAINSET_DIR} "True"  part-r-
        # echo "最后两位数字不符合条件"
    fi

done

