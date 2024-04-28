import numpy as np
import logging
import tensorflow as tf

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================================================================
# TODO - handle relative imports for local and cloud execution
# ================================================================

# relative imports running locally
from . import local_utils as cfg

MAX_PLAYLIST_LENGTH = cfg.MAX_PLAYLIST_LENGTH # 5 | cfg.MAX_PLAYLIST_LENGTH
# ================================================================
        

# ===================================================
# get_candidate_features
# ===================================================
def get_candidate_features():
    '''
    candiate tower features
    '''
    candidate_features = {
        "activity_spu_code":tf.io.FixedLenFeature(dtype=tf.string, shape=()),            
        "brand_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "back_first_ctgy_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "back_second_ctgy_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "back_third_ctgy_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),           
        "activity_mode_code":tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
        "activity_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        
        "is_exchange":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_high_commission":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_hot":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_ka_brand":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_new":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_oversea":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_chaoji_pinpai":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_wholesale_pop":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_tuangou":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_virtual":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_jifen_duihuan":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_n_x_discount":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_n_x_cny":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_youxuan_haowu":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        
        "tenant_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "comprehensive_score":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
        "hour_score":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
        "max_c_sale_price":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    }
    
    return candidate_features

# ===================================================
# get_all_features
# ===================================================
def get_all_features(MAX_PLAYLIST_LENGTH: int, ranker: bool = False):
    '''
    features for both towers and ranker
    '''
    feats = {
        # ===================================================
        # candidate track features
        # ===================================================
        "activity_spu_code":tf.io.FixedLenFeature(dtype=tf.string, shape=()),            
        "brand_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "back_first_ctgy_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "back_second_ctgy_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "back_third_ctgy_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),           
        "activity_mode_code":tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
        "activity_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),      
        "max_c_sale_price":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
        
        "is_exchange":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_high_commission":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_hot":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),

        "is_ka_brand":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_new":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_oversea":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_chaoji_pinpai":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_wholesale_pop":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_tuangou":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_virtual":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_jifen_duihuan":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_n_x_discount":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_n_x_cny":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "is_youxuan_haowu":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "tenant_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "comprehensive_score":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
        "hour_score":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),  

        # ===================================================
        # user features
        # ===================================================        
        "user_id":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "gmv_1y_ranking_type":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "promotion_sentivity_1y":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "addr_city_level":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "address_type":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "region_name":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        
        "area_province_name":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "area_city_name":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "area_county_name":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "share_ranking_360d":tf.io.FixedLenFeature(dtype=tf.string, shape=()),    
        
        "is_seller":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),            
        "is_distributor":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),      
        
        
        "addr_number":tf.io.FixedLenFeature(dtype=tf.int64, shape=()), # todo 检查这个特征有没有遗漏            
       
 
        
        "login_num_30d":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),    
        "last7d_login_num":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),    
        "share_num_360d":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),    
        
        "orders_30d":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        "orders_7d":tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
        
        "gmv_30d":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),  
        "gmv_7d":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),   

        # ===================================================
        # summary playlist features
        # ===================================================
        
        # "pl_name_src" : tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
        # 'pl_collaborative_src' : tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
        # 'pl_duration_ms_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), 
        # 'num_pl_songs_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()),  # num_pl_songs_new | n_songs_pl_new
        # 'num_pl_artists_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), 
        # 'num_pl_albums_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), 
        

        # ===================================================
        # ragged playlist-track features
        # ===================================================
        # bytes / string
        # "track_uri_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "track_name_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "artist_uri_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "artist_name_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
        # "album_uri_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "album_name_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "artist_genres_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 

        # Float List
        # "duration_ms_songs_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "track_pop_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "artist_pop_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "artists_followers_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_danceability_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_energy_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_key_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "track_loudness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_mode_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_speechiness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_acousticness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_instrumentalness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_liveness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)), 
        # "track_valence_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        # "track_tempo_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)), 
        
        # bytes / string
        # "track_time_signature_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
    }
    
    # ===================================================
    # playlist-track rank labels
    # ===================================================
    """
    if ranker == True:
        
        add_rank_entry = {"candidate_rank": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),}
        
        feats.update(add_rank_entry)
    """
    return feats

# ===================================================
# get_audio_ranker_feats
# ===================================================
def get_audio_ranker_feats(MAX_PLAYLIST_LENGTH, list_wise = False):
    '''
    features for both towers
    '''
    feats = {
        # ===================================================
        # summary playlist features
        # ===================================================
        "pl_name_src" : tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
        # ===================================================
        # ragged playlist features
        # ===================================================
        "track_name_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
        # audio feats
        "track_danceability_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        "track_energy_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        "track_key_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
        "track_loudness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        "track_mode_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
        "track_speechiness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        "track_acousticness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        "track_instrumentalness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        "track_liveness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)), 
        "track_valence_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
        "track_tempo_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)), 
        "track_time_signature_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 
        
        # ===================================================
        # candidate track features
        # ===================================================
        "track_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "artist_genres_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        # audio feats
        "track_danceability_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_energy_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_key_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_loudness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_mode_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_speechiness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_acousticness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_instrumentalness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_liveness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_valence_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_tempo_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_time_signature_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        
        # label - rank single (1) candidate track per playlist example
        "candidate_rank": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    }
    if list_wise == True:
        
        # update label - to rank multiple candidate tracks per playlist example
        feats["candidate_rank"] = tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)), 
    
    return feats

# tf data parsing functions
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

# data loading and parsing
def full_parse(data):
    # used for interleave - takes tensors and returns a tf.dataset
    data = tf.data.TFRecordDataset(data)
    return data

def parse_tfrecord(example):
    """
    TODO - will move once parse functions below are completely adopted
    
    Reads a serialized example from GCS and converts to tfrecord
    """
    feats = get_all_features(MAX_PLAYLIST_LENGTH)
    
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example,
        feats
        # features=feats
    )
    return example

def parse_towers_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    """
    feats = get_all_features(MAX_PLAYLIST_LENGTH, ranker=False)
    
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example,
        feats
    )
    return example

def parse_rank_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    """
    feats = get_all_features(MAX_PLAYLIST_LENGTH, ranker=True)
    
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example,
        feats
    )
    return example

def parse_candidate_tfrecord_fn(example):
    """
    Reads candidate serialized examples from gcs and converts to tfrecord
    """
    candidate_features = get_candidate_features()
    
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example, 
        features=candidate_features
    )
    
    return example

def parse_audio_rank_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    
    > returns rank label for single candidate track per example
    """
    feats = get_audio_ranker_feats(MAX_PLAYLIST_LENGTH)
    
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example,
        feats
        # features=feats
    )
    return example

def parse_lw_audio_rank_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    
    > passes `list_wise` parameter to return rank label for multiple 
        candidate tracks per example
    """
    feats = get_audio_ranker_feats(MAX_PLAYLIST_LENGTH, list_wise=True)
    
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example,
        feats
        # features=feats
    )
    return example

# get_candidate_features, get_all_features, full_parse, parse_tfrecord, parse_candidate_tfrecord_fn

# # ===================================================
# # get feature mapping
# # ===================================================
# def get_feature_mapping(key):
#     """
#     returns chosen parse function
    
#     example:
#         desired_mapping = get_feature_mapping(MY_CHOICE)
#     """
    
#     map_dict = {
#         "towers": parse_towers_tfrecord,
#         "rank": parse_rank_tfrecord,
#         "audio-rank": parse_audio_rank_tfrecord,
#         "lw-audio-rank": parse_lw_audio_rank_tfrecord,
#         # "query": feature_utils.parse_XXXX,
#     }
#     return map_dict[key]