from DataProcess import *
from naca import generate_foil_body
from network_conv import network_conv
from network_mlp import network_mlp
from mnist import mnist

if __name__ == '__main__':
    # 生成输入参数全排列
    # generate_input()

    # 提取冰型、结冰部分翼型以及完整翼型，并保存
    raw_data_path = 'D:/Project/data/naca/test_raw_data/'
    new_data_path = 'D:/Project/data/naca/test_data/'

    # 第零步
    # 生成翼型数据
    # generate_foil_body()
    # generate_input(new_data_path)

    # 第一步
    # 从原始数据中提取冰型
    # extraction_ice_shape(raw_data_path, new_data_path)
    # generate_ice_img(raw_data_path, new_data_path)

    # 第二步
    # 将 x-y 坐标数据转换为 ξ-η 坐标数据，然后将其拟合成傅里叶级数，并保存为 csv 文件
    # save_fourier_coeffient(new_data_path)

    # 第三步
    # 根据翼型数据生成对应的图像
    # generate_foil_images(new_data_path)

    # 第四步
    # 训练神经网络
    network_conv()
    # network_mlp()
    # mnist()

    # 第五步
    # 读入预测的傅里叶级数，将其通过反傅里叶变换转换为 ξ-η 坐标数据，进一步转换为 x-y 坐标数据，并绘制冰型图
    # data_path = './output/mse_loss/output.csv'
    # seq_path = './data/seq.txt'
    # foil_info_path = './data/img_path.csv'
    # convert_coordinate_system_inversed(data_path, seq_path, foil_info_path)

    # 第六步
    # 对结果进行分析
    # generate_feature_batch()
    # 画对比图
    # mse_result = './output/mse_loss/ice/'
    # huber_result = './output/huber_loss/ice/'
    # compare(mse_result, huber_result, new_data_path)
