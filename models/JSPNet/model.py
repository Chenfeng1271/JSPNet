import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_upsample
from pointconv_util import pointconv_encoding, pointconv_decoding_depthwise
from loss import *
from model_com import TopNet_create_decoder


def placeholder_inputs(batch_size, num_point, num_dims=9):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_dims))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    sem_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, sem_pl


def get_model(point_cloud, is_training, num_class, num_embed=5, sigma=0.05, bn_decay=None,is_dist=False):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3] #[batch_size,number_point,3] contains xyz 3D coordinate information
    l0_points = point_cloud[:, :, 3:]#[batch_size,number_point,6] contains other RGBL..
    end_points['l0_xyz'] = l0_xyz

    # shared encoder
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, scope='layer1')
    #l1_xyz [batch_size,1024,3] l1_points [batch_size,1024,64] l1_indices [batch_size,1024,32]
    l2_xyz, l2_points = pointconv_encoding(l1_xyz, l1_points, npoint=256, radius=0.2, sigma=2 * sigma, K=32, mlp=[ 64,  64, 128], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='layer2')
    #l2_xyz [batch_size,256,3] l2_points [batch_size,256,128] 
    l3_xyz, l3_points = pointconv_encoding(l2_xyz, l2_points, npoint=64,  radius=0.4, sigma=4 * sigma, K=32, mlp=[128, 128, 256], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='layer3')
    #l3_xyz [batch_size,64,3] l3_points [batch_size,64,256]
    l4_xyz, l4_points = pointconv_encoding(l3_xyz, l3_points, npoint=32,  radius=0.8, sigma=8 * sigma, K=32, mlp=[256, 256, 512], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='layer4')
    #l4_xyz [batch_size,32,3] l4_points [batch_size,32,512]

    # semantic decoder
    l3_points_sem = pointconv_decoding_depthwise(l3_xyz, l4_xyz, l3_points, l4_points,     radius=0.8, sigma=8*sigma, K=16, mlp=[512, 512], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='sem_fa_layer1')
    #l3_points_sem = [batch_size,64,512]
    l2_points_sem = pointconv_decoding_depthwise(l2_xyz, l3_xyz, l2_points, l3_points_sem, radius=0.4, sigma=4*sigma, K=16, mlp=[256, 256], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='sem_fa_layer2')  
    # batch_size x256x256
    l1_points_sem = pointconv_decoding_depthwise(l1_xyz, l2_xyz, l1_points, l2_points_sem, radius=0.2, sigma=2*sigma, K=16, mlp=[256, 128], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='sem_fa_layer3')  
    # batch_sizex1024x128
    l0_points_sem = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_sem, [128, 128, 128], is_training, bn_decay, is_dist=is_dist, scope='sem_fa_layer4')  
    # bx4096x128

    # instance decoder
    l3_points_ins = pointconv_decoding_depthwise(l3_xyz, l4_xyz, l3_points, l4_points,     radius=0.8, sigma=8*sigma, K=16, mlp=[512, 512], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='ins_fa_layer1')
    l2_points_ins = pointconv_decoding_depthwise(l2_xyz, l3_xyz, l2_points, l3_points_ins, radius=0.4, sigma=4*sigma, K=16, mlp=[256, 256], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='ins_fa_layer2')  # 48x256x256
    l1_points_ins = pointconv_decoding_depthwise(l1_xyz, l2_xyz, l1_points, l2_points_ins, radius=0.2, sigma=2*sigma, K=16, mlp=[256, 128], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='ins_fa_layer3')  # 48x1024x128
    l0_points_ins = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_ins, [128, 128, 128], is_training, bn_decay, is_dist=is_dist, scope='ins_fa_layer4')   # 48x4096x128   

    # FC layers F_sem
    l2_points_sem_up = pointnet_upsample(l0_xyz, l2_xyz, l2_points_sem, scope='sem_up1')#[b,4096,256]
    l1_points_sem_up = pointnet_upsample(l0_xyz, l1_xyz, l1_points_sem, scope='sem_up2')#[b,4096,128]
    net_sem_0 = tf.add(tf.concat([l0_points_sem, l1_points_sem_up], axis=-1, name='sem_up_concat'), l2_points_sem_up, name='sem_up_add')#[b,4096,256]
    net_sem_0 = tf_util.conv1d(net_sem_0, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc1', bn_decay=bn_decay)
    #[b,4096,128]

    # FC layers F_ins
    l2_points_ins_up = pointnet_upsample(l0_xyz, l2_xyz, l2_points_ins, scope='ins_up1')#[b,4096,256]
    l1_points_ins_up = pointnet_upsample(l0_xyz, l1_xyz, l1_points_ins, scope='ins_up2')#[b,4096,128]
    net_ins_0 = tf.add(tf.concat([l0_points_ins, l1_points_ins_up], axis=-1, name='ins_up_concat'), l2_points_ins_up, name='ins_up_add')#[b,4096,256]
    net_ins_0 = tf_util.conv1d(net_ins_0, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc1', bn_decay=bn_decay)
    #[b,4096,128]

    net_ins_4, net_sem_4 = JSPNet_SIFF_PIFF(net_sem_0,net_ins_0,bn_decay=bn_decay,is_dist=is_dist,is_training=is_training,num_embed=num_embed,num_point=num_point,num_class=num_class)



    return net_sem_4,net_ins_4





def JISS(sem_input,ins_input,is_training,is_dist,bn_decay,num_point,num_embed,num_class):
    #original JSNet, containing output generation
    # Adaptation
    ###ACFmodule test##
    sem_input = ACFModule(sem_input,ins_input,1,1,128,1,1,is_training,is_dist,bn_decay)
    net_sem_cache_0 = tf_util.conv1d(sem_input, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_cache_1', bn_decay=bn_decay)
    net_ins_1 = ins_input + net_sem_cache_0#[b,4096,128]

    net_ins_2 = tf.concat([ins_input, net_ins_1], axis=-1, name='net_ins_2_concat')#
    net_ins_atten = tf.sigmoid(tf.reduce_mean(net_ins_2, axis=-1, keep_dims=True, name='ins_reduce'), name='ins_atten') #[batch_size,4096,1]
    net_ins_3 = net_ins_2 * net_ins_atten#[b,4096,256]

    # Aggregation
    #ins_input = ACFModule(ins_input,sem_input,1,2,128,2,2,is_training,is_dist,bn_decay,name='ins')
    net_ins_cache_0 = tf_util.conv1d(net_ins_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_cache_1', bn_decay=bn_decay)
    ##[b,4096,128]
    net_ins_cache_1 = tf.reduce_mean(net_ins_cache_0, axis=1, keep_dims=True, name='ins_cache_2')#[b,4096,128]
    net_ins_cache_1 = tf.tile(net_ins_cache_1, [1, num_point, 1], name='ins_cache_tile')
    net_sem_1 = sem_input + net_ins_cache_1#[b,4096,128]

    net_sem_2 = tf.concat([sem_input, net_sem_1], axis=-1, name='net_sem_2_concat')#[b,4096,256]
    net_sem_atten = tf.sigmoid(tf.reduce_mean(net_sem_2, axis=-1, keep_dims=True, name='sem_reduce'), name='sem_atten')#[b,4096,1]
    net_sem_3 = net_sem_2 * net_sem_atten#[b,4096,128]

    # Output
    net_ins_3 = tf_util.conv1d(net_ins_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc2', bn_decay=bn_decay)
    #[b,4096,128]
    net_ins_4 = tf_util.dropout(net_ins_3, keep_prob=0.5, is_training=is_training, scope='ins_dp_4')
    net_ins_4 = tf_util.conv1d(net_ins_4, num_embed, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='ins_fc5')
    #[b,4096,5]

    net_sem_3 = tf_util.conv1d(net_sem_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc2', bn_decay=bn_decay)
    net_sem_4 = tf_util.dropout(net_sem_3, keep_prob=0.5, is_training=is_training, scope='sem_dp_4')
    net_sem_4 = tf_util.conv1d(net_sem_4, num_class, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='sem_fc5')

    return net_ins_4, net_sem_4

def JSPNet_PIFF(sem_input,ins_input,is_training,is_dist,bn_decay,num_point,num_embed,num_class):
    #original JSNet, containing output generation
    # Adaptation
    ###ACFmodule test##
    sem_acf = ACFModule(sem_input,ins_input,1,1,128,1,1,is_training,is_dist,bn_decay)
    sem_acf = tf_util.conv1d(sem_acf,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_cache_1', bn_decay=bn_decay)
    net_ins_1 = ins_input + sem_acf#[b,4096,128]

    net_ins_2 = tf.concat([ins_input, net_ins_1], axis=-1, name='net_ins_2_concat')#
    net_ins_atten = tf.sigmoid(tf.reduce_mean(net_ins_2, axis=-1, keep_dims=True, name='ins_reduce'), name='ins_atten') #[batch_size,4096,1]
    net_ins_3 = net_ins_2 * net_ins_atten#[b,4096,256]

    # Aggregation
    ins_acf = ACFModule(ins_input,sem_input,1,1,128,1,1,is_training,is_dist,bn_decay,name='ins')
    ins_acf = tf_util.conv1d(ins_acf, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_cache_1', bn_decay=bn_decay)
    ##[b,4096,128]
    #net_ins_cache_1 = tf.reduce_mean(ins_acf, axis=1, keep_dims=True, name='ins_cache_2')#[b,4096,128]
    #net_ins_cache_1 = tf.tile(net_ins_cache_1, [1, num_point, 1], name='ins_cache_tile')
    net_sem_1 = sem_input + ins_acf#[b,4096,128]

    net_sem_2 = tf.concat([sem_input, net_sem_1], axis=-1, name='net_sem_2_concat')#[b,4096,256]
    net_sem_atten = tf.sigmoid(tf.reduce_mean(net_sem_2, axis=-1, keep_dims=True, name='sem_reduce'), name='sem_atten')#[b,4096,1]
    net_sem_3 = net_sem_2 * net_sem_atten#[b,4096,128]

    # Output
    net_ins_3 = tf_util.conv1d(net_ins_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc2', bn_decay=bn_decay)
    #[b,4096,128]
    net_ins_4 = tf_util.dropout(net_ins_3, keep_prob=0.5, is_training=is_training, scope='ins_dp_4')
    net_ins_4 = tf_util.conv1d(net_ins_4, num_embed, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='ins_fc5')
    #[b,4096,5]

    net_sem_3 = tf_util.conv1d(net_sem_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc2', bn_decay=bn_decay)
    net_sem_4 = tf_util.dropout(net_sem_3, keep_prob=0.5, is_training=is_training, scope='sem_dp_4')
    net_sem_4 = tf_util.conv1d(net_sem_4, num_class, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='sem_fc5')

    return net_ins_4, net_sem_4

def JSPNet_SIFF_PIFF(sem_input,ins_input,is_training,is_dist,bn_decay,num_point,num_embed,num_class):
    #original JSNet, containing output generation
    # Adaptation
    ###ACFmodule test##
    sem_acf = ExchangeGateModule(sem_input,ins_input,'sem2ins',is_training,is_dist,bn_decay)
    sem_acf = tf_util.conv1d(sem_acf,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_cache_1', bn_decay=bn_decay)
    net_ins_1 = ins_input + sem_acf#[b,4096,128]

    net_ins_2 = tf.concat([ins_input, net_ins_1], axis=-1, name='net_ins_2_concat')#
    net_ins_atten = tf.sigmoid(tf.reduce_mean(net_ins_2, axis=-1, keep_dims=True, name='ins_reduce'), name='ins_atten') #[batch_size,4096,1]
    net_ins_3 = net_ins_2 * net_ins_atten#[b,4096,256]

    # Aggregation
    ins_acf = ExchangeGateModule(net_ins_3,sem_input,'ins2sem',is_training,is_dist,bn_decay)
    ins_acf = tf_util.conv1d(ins_acf, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_cache_1', bn_decay=bn_decay)
    ##[b,4096,128]
    #net_ins_cache_1 = tf.reduce_mean(ins_acf, axis=1, keep_dims=True, name='ins_cache_2')#[b,4096,128]
    #net_ins_cache_1 = tf.tile(net_ins_cache_1, [1, num_point, 1], name='ins_cache_tile')
    net_sem_1 = sem_input + ins_acf#[b,4096,128]

    net_sem_2 = tf.concat([sem_input, net_sem_1], axis=-1, name='net_sem_2_concat')#[b,4096,256]
    net_sem_atten = tf.sigmoid(tf.reduce_mean(net_sem_2, axis=-1, keep_dims=True, name='sem_reduce'), name='sem_atten')#[b,4096,1]
    net_sem_3 = net_sem_2 * net_sem_atten#[b,4096,256]

    ###########acf###
    sem_acf = ACFModule(net_sem_3,net_ins_3,1,1,128,1,1,is_training,is_dist,bn_decay,name='sem',concat=True)
    ins_acf = ACFModule(net_ins_3,net_sem_3,1,1,128,1,1,is_training,is_dist,bn_decay,name='ins',concat=True)

    # Output
    net_ins_3 = tf_util.conv1d(sem_acf, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc2', bn_decay=bn_decay)
    #[b,4096,128]
    net_ins_4 = tf_util.dropout(net_ins_3, keep_prob=0.5, is_training=is_training, scope='ins_dp_4')
    net_ins_4 = tf_util.conv1d(net_ins_4, num_embed, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='ins_fc5')
    #[b,4096,5]

    net_sem_3 = tf_util.conv1d(ins_acf, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc2', bn_decay=bn_decay)
    net_sem_4 = tf_util.dropout(net_sem_3, keep_prob=0.5, is_training=is_training, scope='sem_dp_4')
    net_sem_4 = tf_util.conv1d(net_sem_4, num_class, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='sem_fc5')

    return net_ins_4, net_sem_4

def JSPNet_SIFF(sem_input,ins_input,is_training,is_dist,bn_decay,num_point,num_embed,num_class):
    #original JSNet, containing output generation
    # Adaptation
    ###ACFmodule test##
    sem_acf = ExchangeGateModule(sem_input,ins_input,'sem2ins',is_training,is_dist,bn_decay)
    sem_acf = tf_util.conv1d(sem_acf,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_cache_1', bn_decay=bn_decay)
    net_ins_1 = ins_input + sem_acf#[b,4096,128]

    net_ins_2 = tf.concat([ins_input, net_ins_1], axis=-1, name='net_ins_2_concat')#
    net_ins_atten = tf.sigmoid(tf.reduce_mean(net_ins_2, axis=-1, keep_dims=True, name='ins_reduce'), name='ins_atten') #[batch_size,4096,1]
    net_ins_3 = net_ins_2 * net_ins_atten#[b,4096,256]

    # Aggregation
    ins_acf = ExchangeGateModule(ins_input,sem_input,'ins2sem',is_training,is_dist,bn_decay)
    ins_acf = tf_util.conv1d(ins_acf, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_cache_1', bn_decay=bn_decay)
    ##[b,4096,128]
    #net_ins_cache_1 = tf.reduce_mean(ins_acf, axis=1, keep_dims=True, name='ins_cache_2')#[b,4096,128]
    #net_ins_cache_1 = tf.tile(net_ins_cache_1, [1, num_point, 1], name='ins_cache_tile')
    net_sem_1 = sem_input + ins_acf#[b,4096,128]

    net_sem_2 = tf.concat([sem_input, net_sem_1], axis=-1, name='net_sem_2_concat')#[b,4096,256]
    net_sem_atten = tf.sigmoid(tf.reduce_mean(net_sem_2, axis=-1, keep_dims=True, name='sem_reduce'), name='sem_atten')#[b,4096,1]
    net_sem_3 = net_sem_2 * net_sem_atten#[b,4096,128]

    # Output
    net_ins_3 = tf_util.conv1d(net_ins_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc2', bn_decay=bn_decay)
    #[b,4096,128]
    net_ins_4 = tf_util.dropout(net_ins_3, keep_prob=0.5, is_training=is_training, scope='ins_dp_4')
    net_ins_4 = tf_util.conv1d(net_ins_4, num_embed, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='ins_fc5')
    #[b,4096,5]

    net_sem_3 = tf_util.conv1d(net_sem_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc2', bn_decay=bn_decay)
    net_sem_4 = tf_util.dropout(net_sem_3, keep_prob=0.5, is_training=is_training, scope='sem_dp_4')
    net_sem_4 = tf_util.conv1d(net_sem_4, num_class, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='sem_fc5')

    return net_ins_4, net_sem_4

def JSPNet_PIFF_later(sem_input,ins_input,is_training,is_dist,bn_decay,num_point,num_embed,num_class):
    #original JSNet, containing output generation
    # Adaptation
    ###ACFmodule test##
    #sem_acf = ACFModule(sem_input,ins_input,1,1,128,1,1,is_training,is_dist,bn_decay)
    #sem_acf = tf_util.conv1d(sem_acf,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_cache_1', bn_decay=bn_decay)
    #net_ins_1 = ins_input + sem_acf#[b,4096,128]

    sem2ins = tf_util.conv1d(sem_input,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem2ins_cache_1', bn_decay=bn_decay)
    net_ins_1 = ins_input + sem2ins

    net_ins_2 = tf.concat([ins_input, net_ins_1], axis=-1, name='net_ins_2_concat')#
    net_ins_atten = tf.sigmoid(tf.reduce_mean(net_ins_2, axis=-1, keep_dims=True, name='ins_reduce'), name='ins_atten') #[batch_size,4096,1]
    net_ins_3 = net_ins_2 * net_ins_atten#[b,4096,256]

    # Aggregation
    #ins_acf = ACFModule(ins_input,sem_input,1,1,128,1,1,is_training,is_dist,bn_decay,name='ins')
    #ins_acf = tf_util.conv1d(ins_acf, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_cache_1', bn_decay=bn_decay)
    ##[b,4096,128]
    #net_ins_cache_1 = tf.reduce_mean(ins_acf, axis=1, keep_dims=True, name='ins_cache_2')#[b,4096,128]
    #net_ins_cache_1 = tf.tile(net_ins_cache_1, [1, num_point, 1], name='ins_cache_tile')
    #net_sem_1 = sem_input + ins_acf#[b,4096,128]
    net_ins_3_ada = tf_util.conv1d(net_ins_3,128,1,padding='VALID',bn=True, is_training=is_training, is_dist=is_dist, scope='net_ins_3_ada', bn_decay=bn_decay)
    ins2sem = tf.reduce_mean(net_ins_3_ada, axis=1, keep_dims=True, name='ins2sem_cache_1')
    ins2sem = tf.tile(ins2sem, [1, num_point, 1], name='ins2sem_cache_tile')
    net_sem_1 = sem_input + ins2sem

    net_sem_2 = tf.concat([sem_input, net_sem_1], axis=-1, name='net_sem_2_concat')#[b,4096,256]
    net_sem_atten = tf.sigmoid(tf.reduce_mean(net_sem_2, axis=-1, keep_dims=True, name='sem_reduce'), name='sem_atten')#[b,4096,1]
    net_sem_3 = net_sem_2 * net_sem_atten#[b,4096,256]

    # ACF
    net_sem_4 = tf_util.conv1d(net_sem_3,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='net_sem_4_ada', bn_decay=bn_decay)
    net_ins_4 = tf_util.conv1d(net_ins_3,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='net_ins_4_ada', bn_decay=bn_decay)

    sem_acf = ACFModule(net_sem_4,net_ins_4,1,1,128,1,1,is_training,is_dist,bn_decay,name='sem',concat=True)
    ins_acf = ACFModule(net_ins_4,net_sem_4,1,1,128,1,1,is_training,is_dist,bn_decay,name='ins',concat=True)

    sem_acf = sem_acf + net_sem_3
    ins_acf = ins_acf + net_ins_3

    sem_acf = tf_util.conv1d(sem_acf,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_acf_ada', bn_decay=bn_decay)
    ins_acf = tf_util.conv1d(ins_acf,128,1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_acf_ada', bn_decay=bn_decay)



    # Output
    net_ins_3 = tf_util.conv1d(ins_acf, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc2', bn_decay=bn_decay)
    #[b,4096,128]
    net_ins_4 = tf_util.dropout(net_ins_3, keep_prob=0.5, is_training=is_training, scope='ins_dp_4')
    net_ins_4 = tf_util.conv1d(net_ins_4, num_embed, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='ins_fc5')
    #[b,4096,5]

    net_sem_3 = tf_util.conv1d(sem_acf, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc2', bn_decay=bn_decay)
    net_sem_4 = tf_util.dropout(net_sem_3, keep_prob=0.5, is_training=is_training, scope='sem_dp_4')
    net_sem_4 = tf_util.conv1d(net_sem_4, num_class, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='sem_fc5')

    return net_ins_4, net_sem_4



def SIFF(input1,input2,name,is_training,is_dist,bn_decay):
    #input1 is the main branch
    num_point = input1.shape[1].value
    C = input1.shape[-1].value
    input_1_gb = tf.reduce_mean(input1, axis=1, keep_dims=True, name=name+'_globalpooling')
    input_1_gb = tf.tile(input_1_gb, [1, num_point, 1], name=name+'_tile')
    gb_sim = tf.square((input_1_gb - input1),name=name+'_square') 
    gb_sim_min = tf.reduce_min(gb_sim,axis=-1,keep_dims=True,name=name+'gb_sim_min')
    aligned_gb_sim = gb_sim - gb_sim_min 
    ####how to handle the similarity distance
    gb_sim_mask = tf.sigmoid(aligned_gb_sim,name=name+'sigmoid')

    #######################################
    input2 = tf_util.conv1d(input2,C,1,padding='VALID',bn=True,bn_decay=bn_decay,is_training=is_training,is_dist=is_dist,scope=name+'_channel_adaption1')
    aligned_input2 = gb_sim_mask * input2

    output = input1 + aligned_input2

    return output



def PIFF(input1,input2,n_head,n_mix,d_model,d_k,d_v,is_training,is_dist,bn_decay,kq_transform='conv',value_transform='conv',\
    pooling=True,concat=False,dropout=0.1,name='sem'):

    resudial = input1# set the input1 as main branch, i.e., kt,vt, input2 is qt
    B, N, F = input1.shape
    if kq_transform != 'conv':
        raise NotImplemented

    if pooling == True:
        qt = tf_util.conv1d(input1,n_head*d_k,1,scope=name+'qt_transform_pooling',is_training=is_training,is_dist=is_dist,bn=False,stddev=np.sqrt(2.0 / (d_model + d_k)))
        qt = tf.reshape(qt,[B*n_head,d_k,N])
        kt = tf.layers.average_pooling1d(input2,pool_size=2,strides=2,padding='SAME')
        kt = tf_util.conv1d(kt,n_head*d_k,1,scope=name+'kt_transorm_conv',is_training=is_training,is_dist=is_dist,bn=False)
        kt = tf.reshape(kt,[B*n_head,d_k,N//2])

        vt = tf.layers.average_pooling1d(input2,pool_size=2,strides=2,padding='SAME')
        vt = tf_util.conv1d(vt,n_head*d_k,1,scope=name+'vt_transorm_conv',is_training=is_training,is_dist=is_dist,bn=False)
        vt = tf.reshape(vt,[B*n_head,d_k,N//2])
    else:
        kt = tf_util.conv1d(input1,n_head*d_k,1,scope=name+'kt_transform_pooling',is_training=is_training,is_dist=is_dist,bn=False,stddev=np.sqrt(2.0 / (d_model + d_k)))
        qt = kt
        vt = tf_util.conv1d(input2,n_head*d_k,1,scope=name+'vt_transform_pooling',is_training=is_training,is_dist=is_dist,bn=False,stddev=np.sqrt(2.0 / (d_model + d_k)))

    output, attn = get_atten(qt,kt,vt,n_mix,is_training=is_training,is_dist=is_dist,bn_decay=bn_decay,name=name)
    output = tf_util.conv1d(output,d_model,1,is_training=is_training,is_dist=is_dist,bn=True,bn_decay=bn_decay,scope=name+'attn_output_conv')

    if concat == True:
        output = tf.concat([output,resudial],-1)
    else:
        output = output + resudial
    return output

def get_atten(qt,kt,vt,m,is_dist,is_training,bn_decay,name):
    B, d_k, N = qt.shape
    d = d_k // m
    if m > 1:
        bar_qt = tf.reduce_mean(qt,2,keep_dims=False)
        std = np.power(m,-0.5)
        weight = tf.Variable(tf.random_uniform(shape=[m,int(d_k)],minval=-std,maxval=std))
        #pi = tf.nn.softmax(tf.matmul(weight,bar_qt))
        pi = tf.nn.softmax(tf.matmul(bar_qt,weight))
        pi = tf.reshape(pi,[B*m,1,1])
    
    q = tf.transpose(tf.reshape(qt,[B*m,d,N]),[0,2,1])
    N2 = int(kt.shape[2])
    kt = tf.reshape(kt,[B*m,d,N2])
    v = tf.transpose(vt,[0,2,1])
    
    attn = tf.matmul(q,kt)
    attn = attn / np.power(int(d_k),0.5)
    attn = tf.nn.softmax(attn)
    attn = tf_util.dropout(attn,is_training=is_training,scope=name+"atten_dropout")

    if m > 1:
        attn = tf.reshape((attn * pi),[B,m,N,N2])
        attn = tf.reduce_sum(attn,1)
    output = tf.matmul(attn,v)

    return output, attn


def get_loss(pred, ins_label, pred_sem_label, pred_sem, sem_label, weights=1.0, disc_weight=1.0):
    """ pred:   BxNxE,
        ins_label:  BxN
        pred_sem_label: BxN
        pred_sem: BxNx13
        sem_label: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_label, logits=pred_sem, weights=weights)
    tf.summary.scalar('classify loss', classify_loss)

    feature_dim = pred.get_shape()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.

    disc_loss, l_var, l_dist = discriminative_loss(pred, ins_label, feature_dim, delta_v, delta_d, param_var, param_dist)

    disc_loss = disc_loss * disc_weight
    l_var = l_var * disc_weight
    l_dist = l_dist * disc_weight
    
    loss = classify_loss + disc_loss

    tf.add_to_collection('losses', loss)
    return loss, classify_loss, disc_loss, l_var, l_dist


def get_loss_without_completion(pred, ins_label, pred_sem_label, pred_sem, sem_label,weights=1.0, disc_weight=1.0):
    """ pred:   BxNxE,
        ins_label:  BxN
        pred_sem_label: BxN
        pred_sem: BxNx13
        sem_label: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_label, logits=pred_sem, weights=weights)
    tf.summary.scalar('classify loss', classify_loss)

    feature_dim = pred.get_shape()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.

    disc_loss, l_var, l_dist = discriminative_loss(pred, ins_label, feature_dim, delta_v, delta_d, param_var, param_dist)

    disc_loss = disc_loss * disc_weight
    l_var = l_var * disc_weight
    l_dist = l_dist * disc_weight

    loss = classify_loss + disc_loss 

    tf.add_to_collection('losses', loss)

    return loss, classify_loss, disc_loss, l_var, l_dist


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 2048, 3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)