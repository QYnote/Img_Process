import tkinter as tk
from tkinter import filedialog
#구분한 폴더 만들기
import os
import shutil
#이미지 사이즈 수정 및 폴더 저장
import cv2  #OpenCV
import numpy as np
#모델 불러오기 및 적용하기
import joblib
#결과저장
from PIL import Image

#분류 시킬 폴더 선택
#파일 호출
# Python 기본 윈도우 숨기기, jupyter에서 작동 안됨
root = tk.Tk()
root.withdraw() 
openFolderPath = filedialog.askdirectory(
    # filetypes=(("xlsx File", "*.xlsx"), ("xls File", "*.xls")),
    title='적용폴더 불러오기'
)
# openFolderPath = 'E:\\00.Storage\\Study\\Programing\\Project\\Img_Process\\Image_List\\pass'



#구분한 폴더 만들기
#구분된 통합폴더 존재 확인
applyFolderName = os.getcwd() + '\\Apply'
if os.path.exists(applyFolderName):
    shutil.rmtree(applyFolderName)
os.mkdir(applyFolderName)

#구분된 하위 폴더 존재 확인
for labelName in ['border', 'pass', 'small', 'text', 'white_bg']:
    labelFolder = applyFolderName + '\\' + labelName
    os.mkdir(labelFolder)



#이미지 사이즈 수정 및 폴더 저장
#파일 사이즈 설정 및 수정본 저장
imgList = []
for file in os.listdir(openFolderPath):
    #이미지 가져오기
    FocusedFile = openFolderPath + '\\' + file
    img = cv2.imread(FocusedFile, cv2.IMREAD_COLOR) 

    #파일 Size 1000, 1000 변경
    if (type(img) == np.ndarray) : #꺠진파일 거르기
        if (str(img.shape) != '(1000, 1000, 3)') : 
            print('BfSize :', img.shape)
            img = cv2.resize(img, dsize=(1000,1000), interpolation=cv2.INTER_AREA)
            print('ReSize Image :', file, '/ AfSize :', img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgList.append(img)
        # cv2.imwrite(resizeFolderName + '\\' + file, img)

imgAry = np.array(imgList)



#모델 불러오기 및 적용하기

path = os.getcwd() + '//Models'
files = os.listdir(path)

fileName = files[len(files) - 1]

model = joblib.load(path + "\\" + fileName)

#모델 적용
label = model.predict(imgAry)
print(label.shape)



#결과저장
minAcc = 0.4 #최소 정확도
saveCnt = 0 #정확도에 적합된 데이터 수
labelList = ['border', 'pass', 'small', 'text', 'white_bg']

for idx in range(len(label)):
    print('Accuray :', label[idx][np.argmax(label[idx])], '/ Result :', labelList[np.argmax(label[idx])])

    if(label[idx][np.argmax(label[idx])] > minAcc):
        plt.imshow(imgList[idx])

        savePath = applyFolderName + '\\' + labelList[np.argmax(label[idx])] + '\\' + str(idx) + '.jpg'
        # print('SavePath :', savePath)

        imgList[idx] = Image.fromarray(imgList[idx])
        imgList[idx].save(savePath)

        saveCnt += 1
    
print('SaveCnt :', saveCnt)
