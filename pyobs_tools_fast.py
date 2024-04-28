# -*- coding: utf-8 -*-
# @Author: Chaowei Wu
# @Email:  chaoweiwu@aikucun.com
# @Date:   2023-02-14 14:55:40
# @Last Modified by:   Chaowei Wu
# @Last Modified time: 2023-07-11 17:51:14
# @Description:

import pyobs
import sys
from multiprocessing.dummy import Pool as ThreadPool

def downloadShard(rst, name_pattern):
    last_part = rst.split("/")[-1]
    name_pattern_len = len(name_pattern)
    
    if last_part[0:name_pattern_len] != name_pattern:
        return

    pyobs.download_file(
        fs, rst, '{}/{}'.format(save_path, rst.split("/")[-1]))

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("abnormal parameter length")
        sys.exit(-1)

    save_path = sys.argv[3]
    fs = sys.argv[1]
    path = sys.argv[2]

    save_path = sys.argv[3]
    is_folder = sys.argv[4]
    name_pattern = sys.argv[5]

    # 对于 用户行为数据 'akc_alg.db/tmpfg1014' akc-bigdata
    # 对于映射 'akc_alg.db/tmpfg1019' akc-bigdata

    rsts = pyobs.getobs_filepath(fs, path)
    # print("file system:", fs, "path:", path, "rsts:", rsts)
    params = [(rst, name_pattern) for rst in rsts]
    if is_folder == "True":
        threads = ThreadPool(4)
        threads.starmap(downloadShard, params)
        threads.close()
        threads.join()

    else:
        assert len(rsts) == 1, "should be 1"
        pyobs.download_file(fs, rsts[0], save_path)
