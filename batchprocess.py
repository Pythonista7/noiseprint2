import milvus
import numpy as np
from PIL import Image, ImageOps
import os
import random

from tensorflow.python.ops.gen_array_ops import shape
from noiseprint2 import gen_noiseprint, NoiseprintEngine
from milvus import Milvus, IndexType, MetricType, Status
from sklearn.decomposition import PCA
import tensorflow as tf
import redis

"""
docker run --rm --name redis-commander -d \
  -p 8081:8081 \
  rediscommander/redis-commander:latest
"""


class BatchProcess():
    def __init__(self) -> None:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print(tf.config.experimental.set_memory_growth(
                physical_devices[0], True))
        # init milvus
        self._milvus = Milvus(host='127.0.0.1', port=19530)
        self._collection_name = "local_test"
        self._path = "/home/ash/Desktop/7thSem/7thsem/FinalYearProj/exp/camera_identification/train/"
        self._milvus_dimension = 700  # Change accrodingly wrt transform
        self._engine = NoiseprintEngine()
        self._engine.load_quality(80)
        self._pca = PCA(1)
        self._r = redis.Redis(host='localhost', port=6379, db=0)

    def crop_center(self, img, cropx=700, cropy=700):
        y, x = img.shape
        low = min(x, y)
        if low < 700:
            print("UNDER 700 image crop")
            cropx = low
            cropy = low

        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy, startx:startx+cropx]

    def transform(self, noiseprint):
        converted_data = self._pca.fit_transform(noiseprint).flatten()
        print("Shape of pca :", converted_data.shape)
        return converted_data

    def start(self, folder, limit):
        ls = random.sample(os.listdir(self._path+folder), limit)
        print("Processing " + folder + " : " + str(ls))
        res_vec = []
        res_filename = []
        for img in ls:
            input_image = self.crop_center(np.asarray(ImageOps.grayscale(
                Image.open(self._path+folder+img))
            ))
            noiseprint = self._engine.predict(input_image)
            print("Shape of Noiseprint : ", noiseprint.shape)
            vec = self.transform(noiseprint)
            res_vec.append(vec.tolist())
            res_filename.append(folder+img)

        milvus_id = self._milvus.insert(
            collection_name=self._collection_name, records=res_vec
        )

        print("milvus ids : ", milvus_id[1])
        # store id in redis and map to file name {key:milvus_id - val:filename}
        for mid, fname in zip(milvus_id[1], res_filename):
            self._r.set(mid, fname)

    def search_job(self,folder):
        ls = random.sample(os.listdir(self._path+folder), 1)
        input_image = self.crop_center(np.asarray(ImageOps.grayscale(
                Image.open(self._path+folder+ls[0]))
            ))
        noiseprint = self._engine.predict(input_image)
        query_vec=self.transform(noiseprint)
        res = self._milvus.search(self._collection_name,5,[query_vec.tolist()])
        print("SEARCH RESULT : ",res)

if __name__ == '__main__':
    job = BatchProcess()
    #job.start("HTC-1-M7/", 5)
    #job.start("Motorola-X/", 5)
    job.search_job("Motorola-X/")
