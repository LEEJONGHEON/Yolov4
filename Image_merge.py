import os
import cv2
import numpy as np
import glob
dir = 'D:\\Image_project\\UECFOOD100'

# 모든 파일 경로 저장하기
file_list = []
for i in range(1,101):
  file_list.append(glob.glob(dir+'/'+str(i)+'/*jpg')) # 리스트가 리스트에 저장

# 이미지 데이터 한군데로 모으기
import shutil
file_count = 0
for f in file_list: # 1~100
  for temp in f: # 각 폴더당 이미지 파일경로
    try:
      shutil.move(temp,'D:\\Image_project\\Image\\') # dataes 파일 생성후 해당폴더로 이동
      file_count+=1
    except:
      pass
print(file_count)

# 파일 이름과 동일한 txt파일 만들기
dir = 'D:\\Image_project\\Image\\'
file_list = os.listdir(dir)
new_list = []
for f in file_list:
  new_list.append(f.replace('jpg','txt'))

# 파일 이름으로 파일 생성
dir = 'D:\\Image_project\\Image\\'
for n in new_list:
  f = open(dir+n,'w')
  f.close()

# 파일 개수세기
dir = 'D:\\Image_project\\Image\\*'
print(len(glob.glob(dir)))
