
import os
import numpy as np
import pickle as pkl
import time
from pprint import pprint
import time
import scann

import glob

import tensorflow as tf
import tensorflow_recommenders as tfrs

# this repo
from src.two_tower_jt import two_tower as tt
from src.two_tower_jt import train_utils as train_utils
from util import feature_set_utils as feature_utils

import warnings
warnings.filterwarnings('ignore')
import sys


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO


VOCAB_DICT=dict()
batch_size = 1024   # batch_size = 1024 

# build the model
USE_CROSS_LAYER = True
USE_DROPOUT = True
SEED = 1234
MAX_PLAYLIST_LENGTH = 5
EMBEDDING_DIM = 16   
# PROJECTION_DIM = int(EMBEDDING_DIM / 4) 
PROJECTION_DIM = int(EMBEDDING_DIM / 2) 
DROPOUT_RATE = 0.05
MAX_TOKENS = 20000
# LAYER_SIZES=[256,128]
LAYER_SIZES=[128,64]

LR = .05



# train config
NUM_EPOCHS = 5
VALID_FREQUENCY = 5
HIST_FREQ = 0
EMBED_FREQ = 1

LOCAL_TRAIN_DIR="/data/fred/retrieval_google/retrieval_google/data/model"
LOCAL_CHECKPOINT_DIR = f"{LOCAL_TRAIN_DIR}/chkpts" # my_model.ckpt
LOCAL_EMB_FILE = f'{LOCAL_TRAIN_DIR}/embs/metadata.tsv'



def get_candidate_data_set(candidate_path):
    candidate_files = []
    candidate_path=os.path.join(candidate_path, "*")

    for blob in glob.glob(candidate_path):
        candidate_files.append(blob)
            
    print("candidate_files:", candidate_files)


    candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)

    parsed_candidate_dataset = candidate_dataset.interleave(
        train_utils.full_parse,
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).map(
        feature_utils.parse_candidate_tfrecord_fn, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).with_options(
        options
    )
    # for x in parsed_candidate_dataset.batch(2).take(2):
    #     pprint(x)

    parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem
    return parsed_candidate_dataset


def get_dataset(options, batch_size, NUM_EPOCHS, trainset_path, shuffle_buffer_size = 10000, is_peek=False):
    train_files = []
    for blob in glob.glob(os.path.join(trainset_path, "*/*")):
        if blob.split("/")[-1].startswith("part"):
            train_files.append(blob)
    print("raw file count: ", len(train_files))

    train_dataset_raw = tf.data.Dataset.from_tensor_slices(train_files).prefetch(tf.data.AUTOTUNE)

    # 数据集大小
    dataset_size = train_dataset_raw.reduce(0, lambda x, _: x + 1).numpy()

    # print("dataset_size:", dataset_size)
    # 定义训练集和测试集大小
    # train_size = int(0.05 * dataset_size)
    # test_size = dataset_size - train_size

    test_size=1
    train_size = dataset_size - test_size

    # 拆分数据集为训练集和测试集
      # shuffle_buffer_size 可以根据数据集大小进行调整
    train_dataset_0 = train_dataset_raw.take(train_size)
    test_dataset_0 = train_dataset_raw.skip(train_size)

    train_dataset = train_dataset_0.interleave(
        train_utils.full_parse,
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    ).shuffle(shuffle_buffer_size).map(feature_utils.parse_towers_tfrecord,num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat().with_options(options)

    if is_peek:
        for x in train_dataset.batch(1).take(1):
            pprint(x)

    valid_dataset = test_dataset_0.prefetch(
        tf.data.AUTOTUNE).interleave(train_utils.full_parse, num_parallel_calls=tf.data.AUTOTUNE,cycle_length=tf.data.AUTOTUNE,deterministic=False).map(
        feature_utils.parse_towers_tfrecord, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat().with_options(options)
    
    return train_dataset,valid_dataset


def getMaxCheckFilePath(checkpoint_dir):
    # 获取文件夹中以数字结尾的文件列表
    files = glob.glob(os.path.join(checkpoint_dir, "*[0-9].*"))

    files=sorted(files)
    if len(files) > 0:
        return files[-1]
    return None


def loadLatestCheckpoint(model):
    # latest_checkpoint = tf.train.latest_checkpoint(LOCAL_CHECKPOINT_DIR)
    latest_checkpoint = getMaxCheckFilePath(LOCAL_CHECKPOINT_DIR)

    # Need to call it to set the shapes.
    data = {
        'user_id': np.array(["1421560"]),
        'gmv_1y_ranking_type': np.array(["gmv_1y_ranking_type"]),
        'promotion_sentivity_1y': np.array(["promotion_sentivity_1y"]),
        'addr_city_level': np.array(["addr_city_level"]),
        'address_type': np.array(["address_type"]),
        'region_name': np.array(["region_name"]),
        'area_province_name': np.array(["area_province_name"]),
        'area_city_name': np.array(["area_city_name"]),
        'area_county_name': np.array(["area_county_name"]),
        'share_ranking_360d': np.array(["share_ranking_360d"]),
        'login_num_30d': np.array([1421560]),
        'last7d_login_num': np.array([1421560]),
        'share_num_360d': np.array([1421560]),
        'orders_30d': np.array([1421560]),
        'orders_7d': np.array([1421560]),
        'addr_number': np.array([2]),
        'gmv_30d': np.array([7.]),
        'gmv_7d': np.array([1.]),
        
        'activity_spu_code':np.array(["110"]),
        'brand_id':np.array(["110"]),
        'back_first_ctgy_id':np.array(["110"]),
        'back_second_ctgy_id':np.array(["110110"]),
        'back_third_ctgy_id':np.array(["110110110"]),
        'activity_mode_code':np.array(["2"]),
        'activity_id':np.array(["110110110"]),
        'is_exchange': np.array([1]),
        'is_high_commission': np.array([1]),
        'is_hot': np.array([1]),
        'is_ka_brand': np.array([1]),
        'is_new': np.array([1]),
        'is_oversea': np.array([1]),
        'is_chaoji_pinpai': np.array([1]),
        'is_wholesale_pop': np.array([1]),
        'is_tuangou': np.array([1]),
        'is_virtual': np.array([1]),
        'is_jifen_duihuan': np.array([1]),
        'is_n_x_discount': np.array([1]),
        'is_n_x_cny': np.array([1]),
        'is_youxuan_haowu': np.array([1]),
        'max_c_sale_price': np.array([20.]),
    }
    
    v= model(data)
    print("@@@ \t dummpy output:", v)
    model.compile(optimizer=opt)

    if latest_checkpoint:
        print("@@@ \t latest_checkpoint success:", latest_checkpoint)
        model.load_weights(latest_checkpoint)
    return model


# inspect layers
def check_model_layer(model):
    print("\nPlaylist (query) Tower:")
    for i, l in enumerate(model.query_tower.layers):
        print("query",i, l.name)
        
    print("\nTrack (candidate) Tower:")
    for i, l in enumerate(model.candidate_tower.layers):
        print(i, l.name)


# 定义模型检查点回调
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super(CustomModelCheckpoint, self).__init__()
        self.checkpoint_dir = checkpoint_dir

    def getMaxEpoch(self):
        # 获取文件夹中以数字结尾的文件列表
        files = glob.glob(os.path.join(self.checkpoint_dir, "*[0-9].*"))

        # 如果文件夹为空或没有以数字结尾的文件，则返回 None
        if not files:
            print("@@@ \t No files with numbers found in the folder.")
            max_number = 0
        else:
            # 提取文件名中的数字部分，并找到最大的数字
            numbers = [int( os.path.splitext(os.path.basename(file))[0].split("step")[-1] ) for file in files]
            max_number = max(numbers)
        return max_number
    
    def on_epoch_end(self, epoch, logs=None):
        max_epoch= self.getMaxEpoch()
        print("@@@ \t max_epoch:", max_epoch, self.checkpoint_dir)
        
        if logs is not None :    
            # 获取当前所有的checkpoint文件
            existing_checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "ckpt-step*.h5"))
            if len(existing_checkpoints) >= 7:
                # 如果已经有7个以上的checkpoint文件，则删除最旧的一个
                oldest_checkpoint = min(existing_checkpoints, key=os.path.getctime)
                os.remove(oldest_checkpoint)
                print(f"Deleted oldest checkpoint: {oldest_checkpoint}")

            # 构造文件路径
            filepath = os.path.join(self.checkpoint_dir, f"ckpt-step{max_epoch+1 :03d}.h5")
            # 保存模型权重
            self.model.save_weights(filepath, overwrite=True)
            print(f"Saved checkpoint at epoch {epoch }")


def save_two_towers(model):
    experiment = f'local-train-v2'

    invoke_time = time.strftime("%Y%m%d-%H%M%S")
    RUN_NAME = f'run-{invoke_time}'

    print(f"/data/fred/retrieval_google/retrieval_google/local_train_dir/{experiment}/{RUN_NAME}")
    
    # save query tower
    tf.saved_model.save(
        model.query_tower, export_dir=f"/data/fred/retrieval_google/retrieval_google/local_train_dir/{experiment}/{RUN_NAME}/model-dir/query_model"
    )

    # save candidate tower
    tf.saved_model.save(
        model.candidate_tower, export_dir=f"/data/fred/retrieval_google/retrieval_google/local_train_dir/{experiment}/{RUN_NAME}/model-dir/candidate_model"
    )


# start_time = time.time()
# eval_dict_v1 = model.evaluate(valid_dataset, return_dict=True)
# end_time = time.time()
# elapsed_mins = int((end_time - start_time) / 60)
# print(f"elapsed_mins: {elapsed_mins}")
# print("eval_dict_v1:", eval_dict_v1)

# todo 部署的时候需要用sscan 么?

def save_scann_model(parsed_candidate_dataset, model):
    start_time = time.time()

    scann = tfrs.layers.factorized_top_k.ScaNN(
        k=50,
        query_model=model.query_tower,
        num_reordering_candidates=500,
        num_leaves_to_search=30)

    scann.index_from_dataset(
    candidates=parsed_candidate_dataset.batch(128).cache().map(
        lambda x: (
            x['activity_spu_code'], 
            model.candidate_tower(x)
            )
        )
    )


    # Need to call it to set the shapes.
    rst=scann({
    'user_id': np.array(["1421560"]),
    'gmv_1y_ranking_type': np.array(["gmv_1y_ranking_type"]),
    'promotion_sentivity_1y': np.array(["promotion_sentivity_1y"]),
    'addr_city_level': np.array(["addr_city_level"]),
    'address_type': np.array(["address_type"]),
    'region_name': np.array(["region_name"]),
    'area_province_name': np.array(["area_province_name"]),
    'area_city_name': np.array(["area_city_name"]),
    'area_county_name': np.array(["area_county_name"]),
    'share_ranking_360d': np.array(["share_ranking_360d"]),
    'login_num_30d': np.array([1421560]),
    'last7d_login_num': np.array([1421560]),
    'share_num_360d': np.array([1421560]),
    'orders_30d': np.array([1421560]),
    'orders_7d': np.array([1421560]),
    'addr_number': np.array([2]),
    'gmv_30d': np.array([7.]),
    'gmv_7d': np.array([1.]),
})

    path = os.path.join("./scannmodel/", f"{int(time.time())}")
    tf.saved_model.save(
        scann, path, options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
    )

    loaded = tf.saved_model.load(path)

    end_time = time.time()
    elapsed_scann_mins = int((end_time - start_time) / 60)
    print(f"elapsed_scann_mins: {elapsed_scann_mins}")


def eval_use_scann(valid_dataset, model):
    start_time = time.time()

    model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
    candidates=scann
)
    model.compile()

    scann_result = model.evaluate(valid_dataset, return_dict=True, verbose=1)
    print("scann_result:", scann_result)

    end_time = time.time()

    elapsed_scann_eval_mins = int((end_time - start_time) / 60)
    print(f"elapsed_scann_eval_mins: {elapsed_scann_eval_mins}")

    start_time = time.time()

# eval_use_scann(valid_dataset, model)

if __name__ == "__main__":
    print("LOCAL_TRAIN_DIR: ",LOCAL_TRAIN_DIR)
    print("LOCAL_CHECKPOINT_DIR: ",LOCAL_CHECKPOINT_DIR)
    print("LOCAL_EMB_FILE: ",LOCAL_EMB_FILE)
    print("args count:", len(sys.argv))

    trainset_path = sys.argv[1]
    candidate_path= sys.argv[2]

    parsed_candidate_dataset = get_candidate_data_set(candidate_path)
    train_dataset, valid_dataset = get_dataset(options, batch_size, NUM_EPOCHS, trainset_path)

    opt = tf.keras.optimizers.Adagrad(LR)

    # print(f"PROJECTION_DIM: {PROJECTION_DIM}")

    model = tt.TheTwoTowers(
        layer_sizes=LAYER_SIZES, 
        vocab_dict=VOCAB_DICT, 
        parsed_candidate_dataset=parsed_candidate_dataset,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        seed=SEED,
        use_cross_layer=USE_CROSS_LAYER,
        use_dropout=USE_DROPOUT,
        dropout_rate=DROPOUT_RATE,
        max_tokens=MAX_TOKENS,
    )
    model=loadLatestCheckpoint(model)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(LOCAL_TRAIN_DIR, "logs"), 
        histogram_freq=HIST_FREQ, 
        write_graph=True,
        # embeddings_freq=EMBED_FREQ,
        # embeddings_metadata=LOCAL_EMB_FILE
            # profile_batch=(20,50) #run profiler on steps 20-40 - enable this line if you want to run profiler from the utils/ notebook
        )

    start_time = time.time()
    # 创建自定义模型检查点回调实例
    # LOCAL_CHECKPOINT_DIR todo 配置到参数里
    custom_model_checkpoint_callback = CustomModelCheckpoint(LOCAL_CHECKPOINT_DIR)

    layer_history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        validation_freq=VALID_FREQUENCY,
        epochs=NUM_EPOCHS,
        validation_steps=25,
        callbacks=[
            tensorboard_callback, 
            custom_model_checkpoint_callback,
        ], 
        verbose=1
    )


    end_time = time.time()
    val_keys = [v for v in layer_history.history.keys()]
    runtime_mins = int((end_time - start_time) / 60)


    # gather the metrics for the last epoch to be saved in metrics
    metrics_dict = {"train-time-minutes": runtime_mins}

    _ = [metrics_dict.update({key: layer_history.history[key][-1]}) for key in val_keys]
    print("metrics_dict:", metrics_dict)


    save_scann_model(parsed_candidate_dataset, model)
