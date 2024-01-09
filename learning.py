print('Package 호출 시작')
import tensorflow as tf
import numpy as np
#파일 호출
import tkinter as tk
from tkinter import filedialog
#구분용 폴더 만들기
import os
import shutil
#이미지 사이즈 수정
import cv2  #OpenCV
#학습, 테스트 데이터 분할
import glob
import shutil
#Generator 설정
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#모델 만들기
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
#Model 학습
from tensorflow.keras.callbacks import EarlyStopping
#모델 저장
import datetime
import joblib
print('Package 호출 종료')



# pip 설치 목록(설치일 : 2023-12-11)
# tensorflow
# cv2
# opencv-python
# pillow
# scipy
# joblib
# matplotlib


#파일 호출
# Python 기본 윈도우 숨기기, jupyter에서 작동 안됨
print('폴더 지정')
root = tk.Tk()
root.withdraw() 
openFolderPath = filedialog.askdirectory(
    # filetypes=(("xlsx File", "*.xlsx"), ("xls File", "*.xls")),
    title='적용파일 불러오기'
)
# openFolderPath = 'E:\\00.Storage\\Study\\Programing\\Project\\Img_Process\\Image_List'
print('폴더 지정 종료')


#구분용 폴더 만들기
#ReSize한 폴더 삭제 후 생성
resizeFolderName = os.getcwd() + '\\Train'
if os.path.exists(resizeFolderName):
    shutil.rmtree(resizeFolderName)
os.mkdir(resizeFolderName)


#ReSzie한 폴더에 하위 폴더 삭제 후 생성
for labelName in os.listdir(openFolderPath):
    labelFolder = resizeFolderName + '\\' + labelName

    #폴더 리스트 경로에 파일 있으면 해당 파일면으로 폴더생성됨(내 폴더 기준으로 os.path.isdir 써도 안먹힘)
    if os.path.exists(labelFolder):
        shutil.rmtree(labelFolder)
    os.mkdir(labelFolder)


testFolderName = os.getcwd() + '\\Test'
if os.path.exists(testFolderName):
    shutil.rmtree(testFolderName)
os.mkdir(testFolderName)

#Test 폴더에 하위 폴더 삭제 후 생성
for labelName in os.listdir(openFolderPath):
    labelFolder = testFolderName + '\\' + labelName

    #폴더 리스트 경로에 파일 있으면 해당 파일면으로 폴더생성됨(내 폴더 기준으로 os.path.isdir 써도 안먹힘)
    if os.path.exists(labelFolder):
        shutil.rmtree(labelFolder)
    os.mkdir(labelFolder)



#이미지 사이즈 수정하기

#파일 사이즈 설정 및 수정본 저장
for labelName in os.listdir(openFolderPath):
    labelFile = openFolderPath + '\\' + labelName
    savePath = resizeFolderName + '\\'  + labelName

    print('Focused Folder : ', labelFile)
    print('Saving Folder : ', savePath)

    for fileName in os.listdir(labelFile):
        filePath = labelFile + '\\' + fileName

        #이미지 크기 변경
        img = cv2.imread(filePath, cv2.IMREAD_COLOR) #이미지 가져오기

        if (type(img) == np.ndarray) : #꺠진파일 거르기
            if (str(img.shape) != '(1000, 1000, 3)') : 
                print('BfSize :', img.shape)
                img = cv2.resize(img, dsize=(1000,1000), interpolation=cv2.INTER_AREA)
                print('AfSize :', img.shape)
                print('ReSize Image :', fileName)

            #이미지 저장
            cv2.imwrite(savePath + '\\' + fileName, img)
            print('Saved Image :', fileName)


#학습, 테스트 데이터 분할

extensions = ['jpg', 'png']
ratio = 0.1

#테스트파일 이동
for labelName in os.listdir(openFolderPath):
    for extension in extensions:#이미지 확장자
        imgList = glob.glob(resizeFolderName + '\\' + labelName + '\\*.' + extension)

        #파일 이동
        for img in imgList[0:int(ratio*len(imgList))]:
            if not os.path.exists(testFolderName + '\\' + labelName):
                shutil.move(img, testFolderName + '\\' + labelName)
                # print('Moved Img from ReSize to Test : ', img)

        print('Move End .' + extension, '/ Cnt :', len(imgList))
    print('Move End', labelName, 'Label')


#Generator 설정
imgMaxWidth = 1000
imgMaxHeight = 1000
rateValidation = 0.15

#DataGenerator 설정
#: Tensor 이미지 데이터 배치 생성
trainDataGen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  validation_split=rateValidation)
validationDataGen = ImageDataGenerator(rescale=1./255,
                                       validation_split = rateValidation)

#Generator 설정
#: Directory 경로를 사용하여 증강 데이터 배치를 생성
trainGen = trainDataGen.flow_from_directory(directory=resizeFolderName + '\\',
                                            batch_size=16,
                                            color_mode='rgb',
                                            class_mode='sparse',
                                            target_size=(imgMaxWidth, imgMaxHeight),
                                            subset='training')
validationGen = validationDataGen.flow_from_directory(directory=resizeFolderName + '\\',
                                                      batch_size=16,
                                                      color_mode='rgb',
                                                      class_mode='sparse',
                                                      target_size=(imgMaxWidth, imgMaxHeight),
                                                      subset='validation')

print(trainGen.class_indices)



#모델 만들기

#강의 그저 따라하기
baseModel = MobileNet(weights='imagenet',
                      include_top=False,
                      input_shape=(imgMaxWidth, imgMaxHeight, 3))

model = Sequential()
model.add(baseModel)
model.add(Flatten())

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(2e-5),
              metrics=['accuracy'])

model.summary()


#Model 학습
print('Model Learning')
earlystopping = EarlyStopping(monitor='val_loss', patience=5)
print('learning fit')
hist = model.fit(trainGen,
                 validation_data=validationGen,
                 epochs=10
                ,callbacks=[earlystopping])
print('Model Learning End')



#모델 저장
pathModel = os.getcwd() + '//Models'
if not os.path.exists(pathModel): #모델폴더 생성
    print('Not Exists LabelFolder, Create Folder : Models')
    os.mkdir(pathModel)

#모델명 : 날짜_시분
modelName = str(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
joblib.dump(model, pathModel + "\\" + modelName + ".pkl")