from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2 as cv
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

###
#Dataset Exploration
###
dataDir='dataset'
dataTrain ='train'
dataVal='val'
dataTest='test'
annFileVal='{}/{}/COCO_{}_annos.json'.format(dataDir,dataVal,dataVal)
annFileTrain='{}/{}/COCO_{}_annos.json'.format(dataDir,dataTrain,dataTrain)

# Initialize the COCO api for instance annotations
coco=COCO(annFileVal)

# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats)

imgIds = coco.getImgIds(catIds=catIDs)
print("Number of images containing all the  classes:", len(imgIds))

# load and display a random image
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
print(img)
I = io.imread('{}/{}/{}'.format(dataDir,dataVal,img['file_name']))/255.0

# #Teste de OpenCV
# I_uint8 = (I * 255).astype(np.uint8)
# I_bgr = cv.cvtColor(I_uint8, cv.COLOR_RGB2BGR)
# cv.imshow('teste', I_bgr)

# #Mostrar imagem aleatório
# plt.axis('off')
# plt.imshow(I)
# plt.show()

# # Load and display instance annotations
# plt.imshow(I)
# plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIDs, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
# plt.show()

###
#END
#Dataset Exploration
###



def inference(listImg):
    for img in listImg:  
        im = io.imread(img)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=val_metadata_dicts, 
                       scale=0.5, 
    #                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(50,50))
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(out.get_image()[:, :, ::-1])
        masks = outputs['instances'].pred_masks.cpu().numpy().astype('uint8')
        total_area = outputs['instances'].pred_boxes.area().sum()
        plt.text(20, 40, 'Number of Damaged: {}\n Damaged Area: {} px^2'.format(len(masks), total_area), fontsize = 40, color = 'white', bbox = dict(facecolor = 'red', alpha = 0.5))
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(im)
        plt.show()