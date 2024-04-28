import json
import numpy as np
import pickle as pkl
from pprint import pprint
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import tensorflow_recommenders as tfrs

from util.feature_set_utils import  get_candidate_features, parse_candidate_tfrecord_fn

from src.two_tower_jt import two_tower as tt
from src.two_tower_jt import train_utils
from src.two_tower_jt import feature_sets
import warnings
warnings.filterwarnings('ignore')


CANDIDATE_MODEL_DIR = f'/data/fred/retrieval_google/retrieval_google/local_train_dir/local-train-v2/run-20240428-034031/model-dir/candidate_model'
print(f"CANDIDATE_MODEL_DIR: {CANDIDATE_MODEL_DIR}")


loaded_candidate_model = tf.saved_model.load(CANDIDATE_MODEL_DIR)


print("signatures",loaded_candidate_model.signatures)
# signatures.keys:  ['serving_default']

print("signatures.keys: ",list(loaded_candidate_model.signatures.keys()))

candidate_predictor = loaded_candidate_model.signatures["serving_default"]
print("candidate_predictor.structured_outputs: ",candidate_predictor.structured_outputs)

print("candidate_predictor.output_shapes:", candidate_predictor.output_shapes)
candidate_features = get_candidate_features()
print("candidate_features:", candidate_features)


import glob
candidate_files = []


for blob in glob.glob("/data/fred/retrieval_google/retrieval_google/data/candidate/2024042714/*"):
    candidate_files.append(blob)

# tf-data-option
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
print("candidate_files: ", candidate_files)
candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)

parsed_candidate_dataset = candidate_dataset.interleave(
    train_utils.full_parse,
    cycle_length=tf.data.AUTOTUNE, 
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False
).map(
    parse_candidate_tfrecord_fn, 
    num_parallel_calls=tf.data.AUTOTUNE
).with_options(
    options
)

# parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem
for features in parsed_candidate_dataset.take(1):
    pprint(features)
    print("_______________")
    
    
start_time = time.time()

embs_iter = parsed_candidate_dataset.batch(1000).map(
    lambda data: (
        data["activity_spu_code"],
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
# print(cleaned_embs[0])



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