from PIL import Image

# x,y,w,h 정규화(0~1)하는 코드
def convert_yolo_bbox(img_size, box):

    dw = 1./img_size[0]
    dh = 1./img_size[1]
    # 센터 x, y 좌표
    x = (int(box[1]) + int(box[3]))/2.0
    y = (int(box[2]) + int(box[4]))/2.0

    # w,h 값
    w = abs(int(box[3])-int(box[1]))
    h = abs(int(box[4])-int(box[2]))

    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x,y,w,h)

# 사진 경로 가져오기
import glob
dir = 'D:\Image_project\Image'
file_list = glob.glob(dir+'/*.jpg')

# ex) '1' : D:\\Image_project\\Image\\1.jpg'
new_list = {}
for image in file_list:
  index = image.split('\\')[-1].split('.jpg')[0] # 각 사진 이미지 이름가져오기
  new_list[index] = image # 이미지에 경로매칭하기

# 텍스트 파일 가져오기
import glob
dir = 'D:\Image_project\Image'
txt_list = glob.glob(dir+'/*.txt')

# ex) '1' : D:\\Image_project\\Image\\1.txt
new_txt = {}
for txt in txt_list:
  index = txt.split('\\')[-1].split('.txt')[0] # 각 텍스트 이름과 경로 매칭
  new_txt[index] = txt

# 텍스트에 데이터추가하기
from tqdm import tqdm
for i in tqdm(range(1, 101)):
    dir = f'D:\\Image_project\\UECFOOD100\\{i}\\bb_info.txt'
    f = open(dir, 'r')
    lines = f.readlines()
    count = 0
    for line in lines:
        count += 1
        if count == 1: continue  # 처음 img x1,y1,x2,y2 삭제
        line = line.split(' ')  # list 화
        file_name = new_txt[line[0]]
        wr = open(file_name, 'a+')

        # 사진불러오기
        f = Image.open(new_list[line[0]])
        # 사진 사이즈
        size = f.size
        f.close()

        temp = line[:-1]
        temp.append(line[-1].replace('\n', ''))
        x, y, w, h = convert_yolo_bbox(size, temp)
        data = str(i-1) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
        # print(data+ "이름" +line[0])
        wr.write(data)
        wr.close()