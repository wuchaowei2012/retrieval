import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

MAX_PLAYLIST_LENGTH = 5 # cfg.MAX_PLAYLIST_LENGTH



# tf data parsing functions
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
