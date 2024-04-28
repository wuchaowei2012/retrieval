import tensorflow as tf
import tensorflow_recommenders as tfrs

import numpy as np
import pickle as pkl
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from pprint import pprint


# new fix to train image + ENTRY CMD 
from . import train_utils
from . import train_config as cfg

# TODO
MAX_PLAYLIST_LENGTH = cfg.TRACK_HISTORY # 5
PROJECT_ID = cfg.PROJECT_ID

# ======================
# Vocab Adapts
# ======================
# > TODO: vocab adapts, min/max, etc. - from train set only?
# > TODO: think about Feature Store integration

# ========================================
# playlist tower
# ========================================
class Query_Model(tf.keras.Model):
    '''
    build sequential model for each feature
    pass outputs to dense/cross layers
    concatentate the outputs
    
    the produced embedding represents the features 
    of a Playlist known at query time 
    '''
    def __init__(
        self, 
        layer_sizes, 
        vocab_dict,
        embedding_dim,
        projection_dim,
        seed,
        use_cross_layer,
        use_dropout,
        dropout_rate,
        # max_playlist_length,
        max_tokens
    ):
        super().__init__()
        
        # ========================================
        # non-sequence playlist features
        # ========================================
        """
        # Feature: pl_name_src
        self.pl_name_src_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['pl_name_src'],
                    ngrams=2, 
                    name="pl_name_src_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, # + 1, 
                    output_dim=embedding_dim,
                    name="pl_name_src_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="pl_name_src_1d"),
            ], name="pl_name_src_text_embedding"
        )
        """
        
        """
        # Feature: pl_collaborative_src
        self.pl_collaborative_src_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=3), 
                tf.keras.layers.Embedding(
                    input_dim=3 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="pl_collaborative_emb_layer",
                    input_shape=()
                ),
            ], name="pl_collaborative_emb_model"
        )
        """

        # Feature: addr_number
        self.addr_number_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(20)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="addr_number_emb_layer",
                ),
            ], name="addr_number_emb_model"
        )
        
        # Feature: is_seller
        # vocab = [0,1]
        # self.is_seller_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.IntegerLookup(vocabulary=vocab),
        #         tf.keras.layers.Embedding(
        #             input_dim=2 + 1, 
        #             output_dim=embedding_dim,
        #             name="is_seller_emb_layer",
        #         ),
        #     ], name="is_seller_emb_model"
        # )
        
        # Feature: is_distributor
        # vocab = [0,1]
        # self.is_distributor_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.IntegerLookup(vocabulary=vocab),
        #         tf.keras.layers.Embedding(
        #             input_dim=2 + 1, 
        #             output_dim=embedding_dim,
        #             name="is_distributor_emb_layer",
        #         ),
        #     ], name="is_distributor_emb_model"
        # )
        
        # Feature: login_num_30d
        self.login_num_30d_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(2000)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="login_num_30d_emb_layer",
                ),
            ], name="login_num_30d_emb_model"
        )
        
         # Feature: last7d_login_num
        self.last7d_login_num_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(2000)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="last7d_login_num_emb_layer",
                ),
            ], name="last7d_login_num_emb_model"
        )
        
         # Feature: share_num_360d
        self.share_num_360d_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(2000)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="share_num_360d_emb_layer",
                ),
            ], name="share_num_360d_emb_model"
        )
        
        # Feature: gmv_30d
        self.gmv_30d_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(2000)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="gmv_30d_emb_layer",
                ),
            ], name="gmv_30d_emb_model"
        )
        
        # Feature: gmv_7d | n_songs_pl_new
        self.gmv_7d_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1000)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="gmv_7d_emb_layer",
                ),
            ], name="gmv_7d_emb_model"
        )
        
        # Feature: orders_30d
        self.orders_30d_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(40)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="orders_30d_emb_layer",
                ),
            ], name="orders_30d_emb_model"
        )
        
        # Feature: orders_7d
        self.orders_7d_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(244)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="orders_7d_emb_layer",
                ),
            ], name="orders_7d_emb_model"
        )
        
        # ========================================
        # sequence playlist features
        # ========================================
        
        # Feature: track_name_pl
        """
        self.track_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens,
                    ngrams=2, 
                    vocabulary=vocab_dict['track_name_pl'],
                    # vocabulary=np.array([vocab_dict['track_name_pl']]).flatten(),
                    # output_mode='int',
                    # output_sequence_length=MAX_PLAYLIST_LENGTH,
                    name="track_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, # + 1, 
                    output_dim=embedding_dim,
                    name="track_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="track_name_pl_2d"),
            ], name="track_name_pl_emb_model"
        )
        """
        
        
        # Feature: artist_name_pl
        """
        self.artist_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    ngrams=2, 
                    vocabulary=vocab_dict['artist_name_pl'],
                    name="artist_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, #  + 1, 
                    output_dim=embedding_dim,
                    name="artist_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="artist_name_pl_2d"),
            ], name="artist_name_pl_emb_model"
        )
        """
        
        # Feature: album_name_pl
        """
        self.album_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    ngrams=2, 
                    vocabulary=vocab_dict['album_name_pl'],
                    name="album_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, 
                    output_dim=embedding_dim,
                    name="album_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="album_name_pl_emb_layer_2d"),
            ], name="album_name_pl_emb_model"
        )
        """
        
        # Feature: duration_ms_songs_pl
        """
        self.duration_ms_songs_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(20744575)), # 20744575.0
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="duration_ms_songs_pl_emb_layer",
                    mask_zero=False
                ),
            tf.keras.layers.GlobalAveragePooling1D(name="duration_ms_songs_pl_emb_layer_pl_1d"),
            ], name="duration_ms_songs_pl_emb_model"
        )
        """
        
        # Feature: track_pop_pl
        """
        self.track_pop_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_pop_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_pop_pl_1d"),
            ], name="track_pop_pl_emb_model"
        )
        """
        
        # Feature: track_mode_pl
        """
        self.track_mode_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=3),
                tf.keras.layers.Embedding(
                    input_dim=3 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="track_mode_pl_emb_layer",
                    input_shape=()
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_mode_pl_1d"),
            ], name="track_mode_pl_emb_model"
        )
        """

        # Feature: time_signature_pl
        """
        self.time_signature_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=6),
                tf.keras.layers.Embedding(
                    input_dim=6 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="time_signature_pl_emb_layer",
                    input_shape=()
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="time_signature_pl_1d"),
            ], name="time_signature_pl_emb_model"
        )
        """
        
        # ========================================
        # dense and cross layers
        # ========================================

        # Cross Layers
        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_uniform", 
                name="pl_cross_layer"
            )
        else:
            self._cross_layer = None
            
        # Dense Layers
        self.dense_layers = tf.keras.Sequential(name="pl_dense_layers")
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                )
            )
            if use_dropout:
                self.dense_layers.add(tf.keras.layers.Dropout(dropout_rate))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
                )
            )
            
        ### ADDING L2 NORM AT THE END
        self.dense_layers.add(tf.keras.layers.LayerNormalization(name="normalize_dense"))
        
    # ========================================
    # call
    # ========================================
    def call(self, data):
        '''
        The call method defines what happens when
        the model is called
        '''
       
        all_embs = tf.concat(
            [   
                self.gmv_30d_embedding(data["gmv_30d"]),
                self.gmv_7d_embedding(data["gmv_7d"]), # gmv_7d | n_songs_pl_new
                self.orders_30d_embedding(data["orders_30d"]),
                self.orders_7d_embedding(data["orders_7d"]),
                
                self.login_num_30d_embedding(data["login_num_30d"]),
                self.last7d_login_num_embedding(data["last7d_login_num"]),
                self.share_num_360d_embedding(data["share_num_360d"]),
                
                self.addr_number_embedding(data["addr_number"]),
                # self.is_seller_embedding(data["is_seller"]),
                # self.is_distributor_embedding(data["is_distributor"]),
        
                # sequence features
                # self.track_uri_pl_embedding(data['track_uri_pl']),
                # self.track_name_pl_embedding(tf.reshape(data['track_name_pl'], [-1, MAX_PLAYLIST_LENGTH, 1])),
            ], axis=1)
        
        # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)

class Candidate_Model(tf.keras.Model):
    def __init__(
        self, 
        layer_sizes, 
        vocab_dict,
        embedding_dim,
        projection_dim,
        seed,
        use_cross_layer,
        use_dropout,
        dropout_rate,
        # max_playlist_length,
        max_tokens
    ):
        super().__init__()
        
        # ========================================
        # Candidate features
        # ========================================
        
        # Feature: activity_spu_code
        # 2249561
        self.activity_spu_code_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=500009), # TODO
                tf.keras.layers.Embedding(
                    input_dim=500009+1, 
                    output_dim=embedding_dim,
                    name="activity_spu_code_emb_layer",
                ),
            ], name="activity_spu_code_emb_model"
        )
        
        # Feature: track_name_can
        """
        self.track_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    vocabulary=vocab_dict['track_name_can'],
                    ngrams=2, 
                    name="track_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens+1,
                    output_dim=embedding_dim,
                    name="track_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_name_can_1d"),
            ], name="track_name_can_emb_model"
        )
        """
        
        # Feature: brand_id
        self.brand_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=100003),
                tf.keras.layers.Embedding(
                    input_dim=100003+1, 
                    output_dim=embedding_dim,
                    name="brand_id_emb_layer",
                ),
            ], name="brand_id_emb_model"
        )
        
        # Feature: back_first_ctgy_id
        self.back_first_ctgy_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=101),
                tf.keras.layers.Embedding(
                    input_dim=101+1, 
                    output_dim=embedding_dim,
                    name="back_first_ctgy_id_emb_layer",
                ),
            ], name="back_first_ctgy_id_emb_model"
        )
        
        # Feature: back_second_ctgy_id
        self.back_second_ctgy_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=1009),
                tf.keras.layers.Embedding(
                    input_dim=1009+1, 
                    output_dim=embedding_dim,
                    name="back_second_ctgy_id_emb_layer",
                ),
            ], name="back_second_ctgy_id_emb_model"
        )
        
        # Feature: back_third_ctgy_id
        self.back_third_ctgy_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=7001),
                tf.keras.layers.Embedding(
                    input_dim=7001+1, 
                    output_dim=embedding_dim,
                    name="back_third_ctgy_id_emb_layer",
                ),
            ], name="back_third_ctgy_id_emb_model"
        )
        
        # Feature: activity_mode_code
        self.activity_mode_code_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=11),
                tf.keras.layers.Embedding(
                    input_dim=11+1, 
                    output_dim=embedding_dim,
                    name="activity_mode_code_emb_layer",
                ),
            ], name="activity_mode_code_emb_model"
        )
        
        # Feature: activity_id
        self.activity_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=100003),
                tf.keras.layers.Embedding(
                    input_dim=100003+1, 
                    output_dim=embedding_dim,
                    name="activity_id_emb_layer",
                ),
            ], name="activity_id_emb_model"
        )
        
        
        """
        # Feature: artist_name_can
        self.artist_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['artist_name_can'],
                    ngrams=2, 
                    name="artist_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens+1,
                    output_dim=embedding_dim,
                    name="artist_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_name_can_1d"),
            ], name="artist_name_can_emb_model"
        )
        """
        

        """
        # Feature: album_name_can
        self.album_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['album_name_can'],
                    ngrams=2, 
                    name="album_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens+1,
                    output_dim=embedding_dim,
                    name="album_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="album_name_can_1d"),
            ], name="album_name_can_emb_model"
        )
        """
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/IntegerLookup
        # Feature: is_exchange
        vocab = [0,1]
        self.is_exchange_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_exchange_emb_layer",
                ),
            ], name="is_exchange_emb_model"
        )
        
        # Feature: is_high_commission
        vocab = [0,1]
        self.is_high_commission_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_high_commission_emb_layer",
                ),
            ], name="is_high_commission_emb_model"
        )
        
        # Feature: is_hot
        vocab = [0,1]
        self.is_hot_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_hot_emb_layer",
                ),
            ], name="is_hot_emb_model"
        )
        
        # Feature: is_ka_brand
        vocab = [0,1]
        self.is_ka_brand_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_ka_brand_emb_layer",
                ),
            ], name="is_ka_brand_emb_model"
        )
        
        # Feature: is_new
        vocab = [0,1]
        self.is_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_new_emb_layer",
                ),
            ], name="is_new_emb_model"
        )
        
        # Feature: is_oversea
        vocab = [0,1]
        self.is_oversea_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_oversea_emb_layer",
                ),
            ], name="is_oversea_emb_model"
        )
        
        # Feature: is_chaoji_pinpai
        vocab = [0,1]
        self.is_chaoji_pinpai_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_chaoji_pinpai_emb_layer",
                ),
            ], name="is_chaoji_pinpai_emb_model"
        )
        
        # Feature: is_wholesale_pop
        vocab = [0,1]
        self.is_wholesale_pop_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_wholesale_pop_emb_layer",
                ),
            ], name="is_wholesale_pop_emb_model"
        )
        
        # Feature: is_tuangou
        vocab = [0,1]
        self.is_tuangou_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_tuangou_emb_layer",
                ),
            ], name="is_tuangou_emb_model"
        )
        
        # Feature: is_virtual
        vocab = [0,1]
        self.is_virtual_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_virtual_emb_layer",
                ),
            ], name="is_virtual_emb_model"
        )
        
        # Feature: is_jifen_duihuan
        vocab = [0,1]
        self.is_jifen_duihuan_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_jifen_duihuan_emb_layer",
                ),
            ], name="is_jifen_duihuan_emb_model"
        )
        
        
        # Feature: is_n_x_discount
        vocab = [0,1]
        self.is_n_x_discount_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_n_x_discount_emb_layer",
                ),
            ], name="is_n_x_discount_emb_model"
        )
        
        # Feature: is_n_x_cny
        vocab = [0,1]
        self.is_n_x_cny_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_n_x_cny_emb_layer",
                ),
            ], name="is_n_x_cny_emb_model"
        )
        
        # Feature: is_youxuan_haowu
        vocab = [0,1]
        self.is_youxuan_haowu_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(
                    input_dim=2 + 1, 
                    output_dim=embedding_dim,
                    name="is_youxuan_haowu_emb_layer",
                ),
            ], name="is_youxuan_haowu_emb_model"
        )
        
    
        
        # Feature: max_c_sale_price
        self.max_c_sale_price_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(2000)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="max_c_sale_price_emb_layer",
                ),
            ], name="max_c_sale_price_emb_model"
        )
        
        """
        # Feature: artist_genres_can
        self.artist_genres_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['artist_genres_can'],
                    ngrams=2, 
                    name="artist_genres_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens + 1, 
                    output_dim=embedding_dim,
                    name="artist_genres_can_emb_layer",
                    mask_zero=True
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_genres_can_1d"),
            ], name="artist_genres_can_emb_model"
        )
        """
        
        # track_mode_can
        """
        self.track_mode_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=3),
                tf.keras.layers.Embedding(
                    input_dim=3 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="track_mode_can_emb_layer",
                    input_shape=()
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="track_mode_can_1d"),
            ], name="track_mode_can_emb_model"
        )
        """
        
        
        # ========================================
        # Dense & Cross Layers
        # ========================================
        
        # Cross Layers
        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_uniform", 
                name="can_cross_layer"
            )
        else:
            self._cross_layer = None
        
        # Dense Layer
        self.dense_layers = tf.keras.Sequential(name="candidate_dense_layers")
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                )
            )
            if use_dropout:
                self.dense_layers.add(tf.keras.layers.Dropout(dropout_rate))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
                )
            )
        ### ADDING L2 NORM AT THE END
        self.dense_layers.add(tf.keras.layers.LayerNormalization(name="normalize_dense"))
        
            
    # ========================================
    # Call Function
    # ========================================
            
    def call(self, data):

        all_embs = tf.concat(
            [
                self.activity_spu_code_embedding(data['activity_spu_code']),
                self.brand_id_embedding(data['brand_id']),
                self.back_first_ctgy_id_embedding(data['back_first_ctgy_id']),
                self.back_second_ctgy_id_embedding(data['back_second_ctgy_id']),
                self.back_third_ctgy_id_embedding(data['back_third_ctgy_id']),
                self.activity_mode_code_embedding(data['activity_mode_code']),
                self.activity_id_embedding(data['activity_id']),
				self.max_c_sale_price_embedding(data['max_c_sale_price']),
    
    			self.is_exchange_embedding(data['is_exchange']),
				self.is_high_commission_embedding(data['is_high_commission']),
				self.is_hot_embedding(data['is_hot']),
				self.is_ka_brand_embedding(data['is_ka_brand']),
				self.is_new_embedding(data['is_new']),
				self.is_oversea_embedding(data['is_oversea']),
				self.is_chaoji_pinpai_embedding(data['is_chaoji_pinpai']),
				self.is_wholesale_pop_embedding(data['is_wholesale_pop']),
				self.is_tuangou_embedding(data['is_tuangou']),
				self.is_virtual_embedding(data['is_virtual']),
				self.is_jifen_duihuan_embedding(data['is_jifen_duihuan']),
				self.is_n_x_discount_embedding(data['is_n_x_discount']),
				self.is_n_x_cny_embedding(data['is_n_x_cny']),
				self.is_youxuan_haowu_embedding(data['is_youxuan_haowu']),
            ], axis=1
        )
        
        # return self.dense_layers(all_embs)
                # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)


class TheTwoTowers(tfrs.models.Model):
    def __init__(
        self, 
        layer_sizes, 
        vocab_dict, 
        parsed_candidate_dataset,
        embedding_dim,
        projection_dim,
        seed,
        use_cross_layer,
        use_dropout,
        dropout_rate,
        max_tokens,
        compute_batch_metrics=False
    ):
        super().__init__()
        
        self.query_tower = Query_Model(
            layer_sizes=layer_sizes, 
            vocab_dict=vocab_dict,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            seed=seed,
            use_cross_layer=use_cross_layer,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            max_tokens=max_tokens,
        )

        self.candidate_tower = Candidate_Model(
            layer_sizes=layer_sizes, 
            vocab_dict=vocab_dict,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            seed=seed,
            use_cross_layer=use_cross_layer,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            max_tokens=max_tokens,
        )
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidate_dataset
                .batch(128)
                .map(lambda x: (x['activity_spu_code'], self.candidate_tower(x))), 
                ks=(10, 50, 100)),
            batch_metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(10, name='batch_categorical_accuracy_at_10'), 
                tf.keras.metrics.TopKCategoricalAccuracy(50, name='batch_categorical_accuracy_at_50')
            ],
            remove_accidental_hits=False,
            name="two_tower_retreival_task"
        )


    def compute_loss(self, data, training=False):
        
        query_embeddings = self.query_tower(data)
        candidate_embeddings = self.candidate_tower(data)
        
        return self.task(
            query_embeddings, 
            candidate_embeddings, 
            compute_metrics=not training,
            candidate_ids=data['activity_spu_code'],
            compute_batch_metrics=True
        ) # turn off metrics to save time on training
