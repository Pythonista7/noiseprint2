import numpy as np
from PIL import Image, ImageOps
import os
import random
from noiseprint2 import gen_noiseprint,NoiseprintEngine
from milvus import Milvus, IndexType, MetricType, Status
import tensorflow as tf
from bert_serving.client import BertClient

"""
bert - 1b713c3913298bc936f6098e172eea502410ca915d75ade72a0996be75b7e34b
       eb2ce046b3c6937f8e5f90ada9d76533909699a5d1b3bd0203b1dffafd93d472
sudo docker run -d --name milvus_cpu_1.1.0 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:1.1.0-cpu-d050721-5e559c

"""
#docker run -p 3000:80 milvusdb/milvus-admin:v0.3.0


class BatchProcess():
    def __init__(self) -> None:
        # init milvus
        self._milvus = Milvus(host='192.168.0.105', port=19530)
        self._collection_name = "test"
        self._path = "/home/ash/Desktop/7thSem/7thsem/FinalYearProj/exp/data/"
        # print(Milvus(host="0.0.0.0",port="19530"))
        self._engine = NoiseprintEngine()
        self._engine.load_quality(80)
        self._bc = BertClient()

    
    def start(self,folder,limit):
        ls = random.sample(os.listdir(self._path+folder),limit)
        print("Processing "+ folder +" : "+ str(ls))
        res=[]
        for img in ls :
            input_image = np.asarray(ImageOps.grayscale(Image.open(self._path+folder+img)))
            noiseprint = tf.reshape(self._engine.predict(input_image),-1).numpy()
            vec =  self._bc.encode(noiseprint)
            res.append(vec)
        self._milvus.insert(collection_name=self._collection_name,records = res )


if __name__ == '__main__':
    job = BatchProcess()
    job.start("HTC-1-M7/",10)