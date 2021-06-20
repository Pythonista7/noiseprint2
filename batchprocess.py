import milvus
import numpy as np
from PIL import Image, ImageOps
import os
import random

from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.ops.gen_nn_ops import top_k
from noiseprint2 import gen_noiseprint, NoiseprintEngine
from milvus import Milvus, IndexType, MetricType, Status
from sklearn.decomposition import PCA
import tensorflow as tf
import redis
from tqdm import tqdm
from collections import Counter

class BatchProcess():
    def __init__(self) -> None:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print(tf.config.experimental.set_memory_growth(
                physical_devices[0], True))
        # init milvus
        self._milvus = Milvus(host='127.0.0.1', port=19530)
        self._collection_name = "local_pca1"
        self._path = "/home/ash/Desktop/7thSem/7thsem/FinalYearProj/exp/camera_identification/train/"
        self._milvus_dimension = 700  # Change accrodingly wrt transform
        self._engine = NoiseprintEngine()
        self._engine.load_quality(80)
        self._pca = PCA(1)
        self._r = redis.Redis(host='localhost', port=6379, db=0)
        self._cache = []
        self._correct = 0
        self._wrong = 0
        self.topk=5

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
        # print("Shape of pca :", converted_data.shape)
        return converted_data

    def start(self, folder, limit):
        ls = random.sample(os.listdir(self._path+folder), limit)
        print("Processing " + folder + " : " + str(ls))
        res_vec = []
        res_filename = []
        for img in tqdm(ls):
            input_image = self.crop_center(np.asarray(ImageOps.grayscale(
                Image.open(self._path+folder+"/"+img))
            ))
            noiseprint = self._engine.predict(input_image)
            vec = self.transform(noiseprint)
            res_vec.append(vec.tolist())
            res_filename.append(folder+"/"+img)

        milvus_id = self._milvus.insert(
            collection_name=self._collection_name, records=res_vec
        )

        # store id in redis and map to file name {key:milvus_id - val:filename}
        for mid, fname in zip(milvus_id[1], res_filename):
            self._r.set(mid, fname)

        print(f"{folder} is now complete ! \n\n\n\n\n")

    def search_job(self, folders=["HTC-1-M7", "Motorola-X", "iPhone-6", "LG-Nexus-5x", "Sony-NEX-7"]):
        folder = random.sample(folders, 1)[0]
        # print("Predicting for Target Label - ", folder)
        ls = random.sample(os.listdir(self._path+folder), 1)
        input_image = self.crop_center(np.asarray(ImageOps.grayscale(
            Image.open(self._path+folder+"/"+ls[0]))
        ))
        noiseprint = self._engine.predict(input_image)
        self.evaluateTopk(noiseprint,folder)
        

    def evaluateTopk(self, noiseprint,target):
        print("evaluating topk for image of shape ",noiseprint.shape)
        query_vec = self.transform(noiseprint)
        status, res = self._milvus.search(
            self._collection_name, self.topk, [query_vec.tolist()]
        )
        if status.OK():
            if self.topk == 1:
                predicted_label = self._r.get(res[0][0].id).decode("utf-8")
                # Storing to cache  
                predicted = str(predicted_label.split("/")[0])

            else:
                predicted = self.evalTargetClass(res[0])

                # self._cache.append(f"True Label ---> {folder} , {predicted} <--- Predicted class")
                self._cache.append(f"{target},{predicted}")

                if predicted == target:
                    self._correct += 1
                else:
                    self._wrong += 1
        return
    
    def evalTargetClass(self,topk):
        topk_labels = []
        print("TOPK = ",topk)
        for k in topk:
            print("k=",k.id)
            label = self._r.get(k.id)
            print(label)
            label = label.decode('utf-8').split("/")[0]
            print("Label : ",label)
            topk_labels.append(label)

        top = Counter(topk_labels).most_common()
        print("TOP = ",topk_labels)
        return top[0][0]


if __name__ == '__main__':
    job = BatchProcess()
    sample_size = 100
    loaddata = False
    if loaddata== True:
        job.start("HTC-1-M7", sample_size)
        job.start("Motorola-X", sample_size)
        job.start("iPhone-6", sample_size)
        job.start("LG-Nexus-5x", sample_size)
        job.start("Sony-NEX-7", sample_size)

    test_size = 25
    for i in range(test_size):
        job.search_job()
    print("FINAL RESULTS \n Target Label,Predicted Label")
    for res in tqdm(job._cache):
        print(res)
    print(f"Score : {job._correct}/{(job._correct+job._wrong)}")
    print("Final accuracy on test : ", job._correct/(job._correct+job._wrong))
