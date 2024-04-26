import json
import numpy as np
import pickle as pkl
from pprint import pprint
import time

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU
import gc
# from numba import cuda

import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_io as tfio

from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.blob import Blob

import google.cloud.aiplatform as vertex_ai

from src.two_tower_jt import two_tower as tt
from src.two_tower_jt import train_utils
from src.two_tower_jt import feature_sets

import warnings
warnings.filterwarnings('ignore')


CANDIDATE_MODEL_DIR = f'/data/fred/retrieval_google/retrieval_google/local_train_dir/local-train-v2/run-20240425-184306/model-dir/candidate_model'
print(f"CANDIDATE_MODEL_DIR: {CANDIDATE_MODEL_DIR}")


loaded_candidate_model = tf.saved_model.load(CANDIDATE_MODEL_DIR)


print("signatures",loaded_candidate_model.signatures)
# signatures _SignatureMap({'serving_default': <ConcreteFunction signature_wrapper(*, track_uri_can, track_danceability_can, artist_pop_can, artist_genres_can, track_valence_can, track_name_can, track_acousticness_can, duration_ms_can, track_pop_can, artist_uri_can, track_energy_can, album_name_can, artist_followers_can, track_time_signature_can, track_loudness_can, track_instrumentalness_can, track_mode_can, track_speechiness_can, track_liveness_can, track_key_can, album_uri_can, artist_name_can, track_tempo_can) at 0x7F46E8384A60>})

print("signatures.keys: ",list(loaded_candidate_model.signatures.keys()))

candidate_predictor = loaded_candidate_model.signatures["serving_default"]
print("candidate_predictor.structured_outputs: ",candidate_predictor.structured_outputs)

print("candidate_predictor.output_shapes:", candidate_predictor.output_shapes)
candidate_features = feature_sets.get_candidate_features()
print("candidate_features:", candidate_features)


import glob
candidate_files = []

for blob in glob.glob("/data/fred/retrieval_google/retrieval_google/temp/ndr-v1-myproject32549-bucket/data/v1/candidates/*"):
    candidate_files.append(blob)

# tf-data-option
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
print("candidate_files: ", candidate_files)
candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)

parsed_candidate_dataset = candidate_dataset.interleave(
    # lambda x: tf.data.TFRecordDataset(x),
    train_utils.full_parse,
    cycle_length=tf.data.AUTOTUNE, 
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False
).map(
    feature_sets.parse_candidate_tfrecord_fn, 
    num_parallel_calls=tf.data.AUTOTUNE
).with_options(
    options
)

# parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem
for features in parsed_candidate_dataset.take(1):
    pprint(features)
    print("_______________")
    
    
start_time = time.time()

embs_iter = parsed_candidate_dataset.batch(10000).map(
    lambda data: (
        data["track_uri_can"],
        loaded_candidate_model(data)
    )
)

embs = []
for emb in embs_iter:
    embs.append(emb)

end_time = time.time()
elapsed_time = int((end_time - start_time) / 60)
print(f"elapsed_time: {elapsed_time}")

print(f"Length of embs: {len(embs)}")

x,y = embs[0]
print("one piece of data:", x,y)

# cleanup the embbedding
start_time = time.time()

cleaned_embs = [] #clean up the output
track_uris = []
for ids , embedding in embs:
    cleaned_embs.extend(embedding.numpy())
    track_uris.extend(ids.numpy())

end_time = time.time()
elapsed_time = int((end_time - start_time) / 60)
print(f"elapsed_time: {elapsed_time}")

print(f"Length of cleaned_embs: {len(cleaned_embs)}")
print(cleaned_embs[0])



track_uris_decoded = [z.decode("utf-8") for z in track_uris]

print(f"Length of track_uris: {len(track_uris)}")

print(f"Length of track_uris_decoded: {len(track_uris_decoded)}")


start_time = time.time()

bad_records = []

for i, emb in enumerate(cleaned_embs):
    bool_emb = np.isnan(emb)
    for val in bool_emb:
        if val:
            bad_records.append(i)

end_time = time.time()
elapsed_time = int((end_time - start_time) / 60)
print(f"elapsed_time: {elapsed_time}")

bad_record_filter = np.unique(bad_records)

print(f"bad_records: {len(bad_records)}")
print(f"bad_record_filter: {len(bad_record_filter)}")

# 将坏点数据过滤

start_time = time.time()

track_uris_valid = []
emb_valid = []

for i, pair in enumerate(zip(track_uris_decoded, cleaned_embs)):
    if i in bad_record_filter:
        pass
    else:
        t_uri, embed = pair
        track_uris_valid.append(t_uri)
        emb_valid.append(embed)
        
end_time = time.time()
elapsed_time = int((end_time - start_time) / 60)
print(f"elapsed_time: {elapsed_time}")

# write embedding vectors to json file
VERSION = 'local'
# TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

embeddings_index_filename = f'candidate_embs_{VERSION}.json'

with open(f'{embeddings_index_filename}', 'w') as f:
    for prod, emb in zip(track_uris_valid, emb_valid):
        f.write('{"id":"' + str(prod) + '",')
        f.write('"embedding":[' + ",".join(str(x) for x in list(emb)) + "]}")
        f.write("\n")