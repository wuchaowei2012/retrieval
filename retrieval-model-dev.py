
import os
import numpy as np
import pickle as pkl
import time
from pprint import pprint
import time
import scann

import glob

# tensorflow
import tensorflow as tf
import tensorflow_recommenders as tfrs

# this repo
from src.two_tower_jt import two_tower as tt
from src.two_tower_jt import train_utils as train_utils
from util import feature_set_utils as feature_utils

import warnings
warnings.filterwarnings('ignore')


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

batch_size = 1024   # batch_size = 1024 
# with open('vocab_dict.pkl', 'rb') as filehandler:
#     VOCAB_DICT = pkl.load(filehandler)

# print("VOCAB_DICT:", VOCAB_DICT)
VOCAB_DICT=dict()

# build the model
USE_CROSS_LAYER = True
USE_DROPOUT = True
SEED = 1234
MAX_PLAYLIST_LENGTH = 5
EMBEDDING_DIM = 16   
PROJECTION_DIM = int(EMBEDDING_DIM / 4) # 50  
SEED = 1234
DROPOUT_RATE = 0.05
MAX_TOKENS = 20000
# LAYER_SIZES=[256,128]
LAYER_SIZES=[128,64]

LR = .1

# setup vertex experiment
EXPERIMENT_NAME = f'local-train-v2'

invoke_time = time.strftime("%Y%m%d-%H%M%S")
RUN_NAME = f'run-{invoke_time}'

print(f"RUN_NAME: {RUN_NAME}")

# train config
NUM_EPOCHS = 5
VALID_FREQUENCY = 5
HIST_FREQ = 0
EMBED_FREQ = 1

LOCAL_TRAIN_DIR = f"local_train_dir/{EXPERIMENT_NAME}/{RUN_NAME}/"
LOCAL_CHECKPOINT_DIR = f"{LOCAL_TRAIN_DIR}/chkpts" # my_model.ckpt
LOCAL_EMB_FILE = f'{LOCAL_TRAIN_DIR}/embs/metadata.tsv'


print("LOCAL_TRAIN_DIR: ",LOCAL_TRAIN_DIR)
print("LOCAL_CHECKPOINT_DIR: ",LOCAL_CHECKPOINT_DIR)
print("LOCAL_EMB_FILE: ",LOCAL_EMB_FILE)


candidate_files = []
for blob in glob.glob("/data/fred/retrieval_google/retrieval_google/data/candidate/2024042714/*"):
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


train_files = []
for blob in glob.glob("/data/fred/retrieval_google/retrieval_google/data/trainset/*/*"):
    if blob.split("/")[-1].startswith("part"):
        train_files.append(blob)
print("train_files: ", train_files)

# train_files    
train_dataset_raw = tf.data.Dataset.from_tensor_slices(train_files).prefetch(
    tf.data.AUTOTUNE,
)

# 假设 candidate_dataset 是你的数据集

# 数据集大小
dataset_size = train_dataset_raw.reduce(0, lambda x, _: x + 1).numpy()

print("dataset_size:", dataset_size)
# 定义训练集和测试集大小
train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size

# 拆分数据集为训练集和测试集
shuffle_buffer_size = 10000  # 可以根据数据集大小进行调整
train_dataset_0 = train_dataset_raw.take(train_size)
test_dataset_0 = train_dataset_raw.skip(train_size)

train_dataset = train_dataset_0.interleave(
    train_utils.full_parse,
    cycle_length=tf.data.AUTOTUNE, 
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False,
).shuffle(shuffle_buffer_size
).map(
    feature_utils.parse_towers_tfrecord,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(
    batch_size 
).prefetch(
    tf.data.AUTOTUNE,
).with_options(
    options
)

# for x in train_dataset.batch(1).take(1):
#     pprint(x)


valid_dataset = test_dataset_0.prefetch(
    tf.data.AUTOTUNE,
).interleave(
    train_utils.full_parse,
    num_parallel_calls=tf.data.AUTOTUNE,
    cycle_length=tf.data.AUTOTUNE, 
    deterministic=False,
).map(
    feature_utils.parse_towers_tfrecord, 
    num_parallel_calls=tf.data.AUTOTUNE
).batch(
    batch_size
).prefetch(
    tf.data.AUTOTUNE,
).with_options(
    options
)



parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem


opt = tf.keras.optimizers.Adagrad(LR)

print(f"PROJECTION_DIM: {PROJECTION_DIM}")

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

model.compile(optimizer=opt)
# inspect layers
## Quick look at the layers
print("Playlist (query) Tower:")

for i, l in enumerate(model.query_tower.layers):
    print("query",i, l.name)
    
    
print("Track (candidate) Tower:")
for i, l in enumerate(model.candidate_tower.layers):
    print(i, l.name)



tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOCAL_TRAIN_DIR, 
    histogram_freq=HIST_FREQ, 
    write_graph=True,
    embeddings_freq=EMBED_FREQ,
    embeddings_metadata=LOCAL_EMB_FILE
        # profile_batch=(20,50) #run profiler on steps 20-40 - enable this line if you want to run profiler from the utils/ notebook
    )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=LOCAL_CHECKPOINT_DIR + "/cp-{epoch:03d}-loss={loss:.2f}.ckpt", 
    save_weights_only=True,
    save_best_only=True,
    monitor='total_loss',
    mode='min',
    save_freq='epoch',
    verbose=1,
)

#start the timer and training
start_time = time.time()

layer_history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    validation_freq=VALID_FREQUENCY,
    epochs=NUM_EPOCHS,
    # steps_per_epoch=50,
    validation_steps=100,
    callbacks=[
        tensorboard_callback, 
        model_checkpoint_callback
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



# save query tower
tf.saved_model.save(
    model.query_tower, export_dir=f"/data/fred/retrieval_google/retrieval_google/local_train_dir/{EXPERIMENT_NAME}/{RUN_NAME}/model-dir/query_model"
)

# save candidate tower
tf.saved_model.save(
    model.candidate_tower, export_dir=f"/data/fred/retrieval_google/retrieval_google/local_train_dir/{EXPERIMENT_NAME}/{RUN_NAME}/model-dir/candidate_model"
)


# TODO - modularize into src
valid_files = []

for blob in glob.glob("/data/fred/retrieval_google/retrieval_google/temp/ndr-v1-myproject32549-bucket/data/v1/valid/*"):
    if '.tfrecords' in blob:
        valid_files.append(blob)


valid = tf.data.TFRecordDataset(valid_files).take(16384)
valid_parsed = valid.map(feature_utils.parse_tfrecord)
cached_valid = valid_parsed.batch(4096).cache()

print("cardinality:", cached_valid.cardinality().numpy())
start_time = time.time()

eval_dict_v1 = model.evaluate(valid_dataset, return_dict=True)
end_time = time.time()
elapsed_mins = int((end_time - start_time) / 60)
print(f"elapsed_mins: {elapsed_mins}")
print("eval_dict_v1:", eval_dict_v1)

# todo 部署的时候需要用sscan 么?

start_time = time.time()

scann = tfrs.layers.factorized_top_k.ScaNN(
    k=50,
    query_model=model.query_tower,
    num_reordering_candidates=500,
    num_leaves_to_search=30
)

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



path = os.path.join(f"./scannmodel/{int(time.time())}", "scannmodel")
tf.saved_model.save(
    scann,
    path,
    options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
)

loaded = tf.saved_model.load(path)









# end_time = time.time()

# elapsed_scann_mins = int((end_time - start_time) / 60)
# print(f"elapsed_scann_mins: {elapsed_scann_mins}")


# start_time = time.time()

# model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
#     candidates=scann
# )
# model.compile()

# scann_result = model.evaluate(
#     valid_dataset, 
#     return_dict=True, 
#     verbose=1
# )

# end_time = time.time()

# elapsed_scann_eval_mins = int((end_time - start_time) / 60)
# print(f"elapsed_scann_eval_mins: {elapsed_scann_eval_mins}")

# # Save the candidate embeddings to GCS for use in Matching Engine later
# start_time = time.time()

# candidate_embeddings = parsed_candidate_dataset.batch(10000).map(
#     lambda x: [
#         x['activity_spu_code'],
#         train_utils.tf_if_null_return_zero(
#             model.candidate_tower(x)
#         )
#     ]
# )

# elapsed_mins = int((time.time() - start_time) / 60)
# print(f"elapsed_mins: {elapsed_mins}")
# # candidate_embeddings
# len(list(candidate_embeddings))
# CANDIDATE_EMB_JSON = 'candidate_embeddings.json'

# start_time = time.time()

# for batch in candidate_embeddings:
#     songs, embeddings = batch
#     with open(CANDIDATE_EMB_JSON, 'a') as f:
#         for song, emb in zip(songs.numpy(), embeddings.numpy()):
#             f.write('{"id":"' + str(song) + '","embedding":[' + ",".join(str(x) for x in list(emb)) + ']}')
#             f.write("\n")
            
# end_time = time.time()

# elapsed_mins = int((end_time - start_time) / 60)
# print(f"elapsed_mins: {elapsed_mins}")
# print("embeddings:", embeddings)