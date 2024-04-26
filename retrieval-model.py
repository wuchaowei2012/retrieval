# 用于初步训练双塔模型
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# os.environ['TF_GPU_THREAD_MODE']='gpu_private'
# os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
# os.environ["CLOUD_ML_PROJECT_ID"] = PROJECT_ID

import json
import numpy as np
import pickle as pkl
import logging
import time
from pprint import pprint

import scann



# tensorflow
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorboard.plugins import projector

# google cloud
# from google.cloud import aiplatform as vertex_ai
# from google.cloud import storage

# storage_client = storage.Client(project=PROJECT_ID)

# this repo
from src.two_tower_jt import two_tower as tt
from src.two_tower_jt import train_utils as train_utils
from util import feature_set_utils as feature_utils

import warnings
warnings.filterwarnings('ignore')




# create database object

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

batch_size = 2  # batch_size = 1024 
train_files = []

import glob
for blob in glob.glob("/data/fred/retrieval_google/retrieval_google/temp/ndr-v1-myproject32549-bucket/data/v1/train/*"):
    if '.tfrecords' in blob:
        train_files.append(blob)

print("train_files: ", train_files)

# train_files    
train_dataset = tf.data.Dataset.from_tensor_slices(train_files).prefetch(
    tf.data.AUTOTUNE,
)

train_dataset = train_dataset.interleave(
    train_utils.full_parse,
    cycle_length=tf.data.AUTOTUNE, 
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False,
).map(
    feature_utils.parse_towers_tfrecord,
    # feature_utils.parse_tfrecord,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(
    batch_size 
).prefetch(
    tf.data.AUTOTUNE,
).with_options(
    options
)

for x in train_dataset.batch(1).take(1):
    pprint(x)

# train_dataset

valid_files = []

for blob in glob.glob("/data/fred/retrieval_google/retrieval_google/temp/ndr-v1-myproject32549-bucket/data/v1/valid/*"):
    if '.tfrecords' in blob:
        valid_files.append(blob)

val_ds = tf.data.Dataset.from_tensor_slices(valid_files)



valid_dataset = val_ds.prefetch(
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

# valid_dataset = valid_dataset.cache() #1gb machine mem + 400 MB in candidate ds (src/two-tower.py)

for x in valid_dataset.batch(1).take(1):
    pprint(x)
valid_dataset


candidate_files = []


for blob in glob.glob("/data/fred/retrieval_google/retrieval_google/temp/ndr-v1-myproject32549-bucket/data/v1/candidates/*"):
    if '.tfrecords' in blob:
        candidate_files.append(blob)
        

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

parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem
# parsed_candidate_dataset

filehandler = open('vocab_dict.pkl', 'rb')
VOCAB_DICT = pkl.load(filehandler)
filehandler.close()

VOCAB_DICT


# build the model
USE_CROSS_LAYER = True
USE_DROPOUT = True
SEED = 1234
MAX_PLAYLIST_LENGTH = 5
EMBEDDING_DIM = 128   
PROJECTION_DIM = int(EMBEDDING_DIM / 4) # 50  
SEED = 1234
DROPOUT_RATE = 0.33
MAX_TOKENS = 20000
LAYER_SIZES=[256,128]

LR = .1
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
    # max_playlist_length=MAX_PLAYLIST_LENGTH,
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

# setup vertex experiment
EXPERIMENT_NAME = f'local-train-v2'

invoke_time = time.strftime("%Y%m%d-%H%M%S")
RUN_NAME = f'run-{invoke_time}'




print(f"RUN_NAME: {RUN_NAME}")
# print(f"LOG_DIR: {LOG_DIR}")

# train config
NUM_EPOCHS = 1
VALID_FREQUENCY = 5
HIST_FREQ = 0
EMBED_FREQ = 1

LOCAL_TRAIN_DIR = f"local_train_dir/{EXPERIMENT_NAME}/{RUN_NAME}/"
LOCAL_CHECKPOINT_DIR = f"{LOCAL_TRAIN_DIR}/chkpts" # my_model.ckpt
LOCAL_EMB_FILE = f'{LOCAL_TRAIN_DIR}/embs/metadata.tsv'


print("LOCAL_TRAIN_DIR: ",LOCAL_TRAIN_DIR)
print("LOCAL_CHECKPOINT_DIR: ",LOCAL_CHECKPOINT_DIR)
print("LOCAL_EMB_FILE: ",LOCAL_EMB_FILE)



tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOCAL_TRAIN_DIR, 
    histogram_freq=HIST_FREQ, 
    write_graph=True,
    embeddings_freq=EMBED_FREQ,
    embeddings_metadata=LOCAL_EMB_FILE
    
    
        # profile_batch=(20,50) #run profiler on steps 20-40 - enable this line if you want to run profiler from the utils/ notebook
    )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=LOCAL_CHECKPOINT_DIR + "/cp-{epoch:03d}-loss={loss:.2f}.ckpt", # cp-{epoch:04d}.ckpt" cp-{epoch:04d}.ckpt"
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
    steps_per_epoch=50,
    validation_steps=100,
    callbacks=[
        tensorboard_callback, 
        # train_utils.UploadTBLogsBatchEnd(
        #     log_dir=LOG_DIR, 
        #     experiment_name=EXPERIMENT_NAME, 
        #     tb_resource_name=TB_RESOURCE_NAME
        # ),
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

# eval_dict_v1 = model.evaluate(valid_dataset, return_dict=True)
# end_time = time.time()
# elapsed_mins = int((end_time - start_time) / 60)
# print(f"elapsed_mins: {elapsed_mins}")
# print("eval_dict_v1:", eval_dict_v1)

# todo 部署的时候需要用sscan 么?

start_time = time.time()

scann = tfrs.layers.factorized_top_k.ScaNN(
    num_reordering_candidates=500,
    num_leaves_to_search=30
)

scann.index_from_dataset(
    candidates=parsed_candidate_dataset.batch(128).cache().map(
        lambda x: (
            x['track_uri_can'], 
            model.candidate_tower(x)
        )
    )
)

end_time = time.time()

elapsed_scann_mins = int((end_time - start_time) / 60)
print(f"elapsed_scann_mins: {elapsed_scann_mins}")


start_time = time.time()

model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
    candidates=scann
)
model.compile()

scann_result = model.evaluate(
    valid_dataset, 
    return_dict=True, 
    verbose=1
)

end_time = time.time()

elapsed_scann_eval_mins = int((end_time - start_time) / 60)
print(f"elapsed_scann_eval_mins: {elapsed_scann_eval_mins}")

# Save the candidate embeddings to GCS for use in Matching Engine later
start_time = time.time()

candidate_embeddings = parsed_candidate_dataset.batch(10000).map(
    lambda x: [
        x['track_uri_can'],
        train_utils.tf_if_null_return_zero(
            model.candidate_tower(x)
        )
    ]
)

elapsed_mins = int((time.time() - start_time) / 60)
print(f"elapsed_mins: {elapsed_mins}")
# candidate_embeddings
len(list(candidate_embeddings))
CANDIDATE_EMB_JSON = 'candidate_embeddings.json'

start_time = time.time()

for batch in candidate_embeddings:
    songs, embeddings = batch
    with open(CANDIDATE_EMB_JSON, 'a') as f:
        for song, emb in zip(songs.numpy(), embeddings.numpy()):
            f.write('{"id":"' + str(song) + '","embedding":[' + ",".join(str(x) for x in list(emb)) + ']}')
            f.write("\n")
            
end_time = time.time()

elapsed_mins = int((end_time - start_time) / 60)
print(f"elapsed_mins: {elapsed_mins}")
print("embeddings:", embeddings)