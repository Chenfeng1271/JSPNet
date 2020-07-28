import os
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import time
import cv2

COLOR = [np.array([255, 0, 0]),
         np.array([0, 255, 0]),
         np.array([0, 0, 255]),
         np.array([125, 125, 0]),
         np.array([0, 125, 125]),
         np.array([125, 0, 125]),
         np.array([50, 100, 50]),
         np.array([100, 50, 100])]


def cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth, bin_seeding=True)#核加权实现质心漂移，迭代收敛，最终收敛在密集的点上，bandwidth是半径，bin_seeding=True是离散点的平均
    # print ('Mean shift clustering, might take some time ...')
    # tic = time.time()
    ms.fit(prediction)#聚类
    # print ('time for clustering', time.time() - tic)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    num_clusters = cluster_centers.shape[0]

    return num_clusters, labels, cluster_centers


def get_instance_masks(prediction, bandwidth):#聚类成instance后clolrize mask的过程
    batch_size, h, w, feature_dim = prediction.shape

    instance_masks = []
    for i in range(batch_size):#逐个对每个batch的feature进行cluster
        num_clusters, labels, cluster_centers = cluster(prediction[i].reshape([h * w, feature_dim]), bandwidth)
        print('Number of predicted clusters', num_clusters)
        labels = np.array(labels, dtype=np.uint8).reshape([h, w])#实现instance聚类后返回label信息的point-wise feature
        mask = np.zeros([h, w, 3], dtype=np.uint8)

        num_clusters = min([num_clusters, 8])#instance至少8个
        for mask_id in range(num_clusters):
            ind = np.where(labels == mask_id)
            mask[ind] = COLOR[mask_id]

        instance_masks.append(mask)

    return instance_masks


def save_instance_masks(prediction, output_dir, bandwidth, count):#和get_instance_mask()一样，不过save mask
    batch_size, h, w, feature_dim = prediction.shape

    instance_masks = []
    for i in range(batch_size):
        num_clusters, labels, cluster_centers = cluster(prediction[i].reshape([h * w, feature_dim]), bandwidth)
        print('Number of predicted clusters', num_clusters)
        labels = np.array(labels, dtype=np.uint8).reshape([h, w])
        mask = np.zeros([h, w, 3], dtype=np.uint8)

        num_clusters = min([num_clusters, 8])
        for mask_id in range(num_clusters):
            mask = np.zeros([h, w, 3], dtype=np.uint8)
            ind = np.where(labels == mask_id)
            mask[ind] = np.array([255, 255, 255])
            output_file_name = os.path.join(output_dir, 'cluster_{}_{}.png'.format(str(count).zfill(4), str(mask_id)))
            cv2.imwrite(output_file_name, mask)

        instance_masks.append(mask)

    return instance_masks
