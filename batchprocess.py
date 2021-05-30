import numpy as np
from PIL import Image, ImageOps
import os
import random
from noiseprint2 import gen_noiseprint,NoiseprintEngine
from milvus import Milvus, IndexType, MetricType, Status
import tensorflow as tf
import redis

class BatchProcess():
    def __init__(self) -> None:
        # init milvus
        self._milvus = Milvus(host='192.168.0.105', port=19530)
        self._collection_name = "test"
        self._path = "/home/ash/Desktop/7thSem/7thsem/FinalYearProj/exp/data/"
        self._milvus_dimension = 4 # Change accrodingly wrt transform
        self._engine = NoiseprintEngine()
        self._engine.load_quality(80)
        self._r = redis.Redis(host='localhost', port=6379, db=0)

    def transform(self,noiseprint):
        return [0]*self._milvus_dimension

    def start(self,folder,limit):
        ls = random.sample(os.listdir(self._path+folder),limit)
        print("Processing "+ folder +" : "+ str(ls))
        res_vec=[]
        res_filename=[]
        for img in ls :
            input_image = np.asarray(ImageOps.grayscale(Image.open(self._path+folder+img)))
            noiseprint = tf.reshape(self._engine.predict(input_image),-1).numpy()
            vec =  self.transform(noiseprint)
            res_vec.append(vec)
            res_filename.append(folder+img)
        
        milvus_id = self._milvus.insert(collection_name=self._collection_name,records = res_vec )
        print("milvus ids : ",milvus_id[1])
        # store id in redis and map to file name {key:milvus_id - val:filename}
        for mid,fname in zip(milvus_id[1],res_filename):
            self._r.set(mid,fname)


if __name__ == '__main__':
    job = BatchProcess()
    job.start("HTC-1-M7/",10)