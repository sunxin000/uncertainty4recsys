from PIL import Image
from pyparsing import conditionAsParseAction
COL = 5
ROW = 2

COL = 5 #指定拼接图片的列数
ROW = 2 #指定拼接图片的行数
UNIT_HEIGHT_SIZE = 600 #图片高度
UNIT_WIDTH_SIZE = 800 #图片宽度
PATH = "" #需要拼接的图片所在的路径
NAME = "" #拼接出的图片保存的名字
RANDOM_SELECT = False #设置是否可重复抽取图片
SAVE_QUALITY = 50 #保存的图片的质量 可选0-100

def concat_images(name, path):
    image_files = []
    for index in range(1, COL*ROW+1):
        image_files.append(Image.open(f'pic/coat/small/neumf_{index * 10}_20_quantile.jpg')) #读取所有用于拼接的图片
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW)) #创建成品图的画布
    #第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    for row in range(ROW):
        for col in range(COL):
            #对图片进行逐行拼接
            #paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
            #或四元元组（指定复制位置的左上角和右下角坐标）
            target.paste(image_files[COL*row+col], (0 + UNIT_WIDTH_SIZE*col, 0 + UNIT_HEIGHT_SIZE*row))
    target.save(path + name + '.jpg', quality=SAVE_QUALITY) #成品图保存

if __name__ == "__main__":
    concat_images('epoch', './')