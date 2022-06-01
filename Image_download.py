import wget
import math
def bar_custom(current, total, width=80):
  width=30
  avail_dots = width-2
  shaded_dots = int(math.floor(float(current) / total * avail_dots))
  percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
  progress = "%d%% %s [%d / %d]" % (current / total * 100, percent_bar, current, total)
  return progress

url = "http://foodcam.mobi/dataset100.zip" # 다운받을 파일 경로
wget.download(url,out='D:\Image_project/',bar=bar_custom) # out : 다운받을 경로

# 다운로드된 이미지 개수세기
import glob
dir = 'D:\Image_project/UECFOOD100/*/*.jpg'
length = len(glob.glob(dir))
print(length)