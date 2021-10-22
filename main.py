from DataProcess import *


if __name__ == '__main__':
    # 生成输入参数全排列
    # generate_input()

    # 提取冰型、结冰部分翼型以及完整翼型，并保存
    raw_data_path = 'C:/Users/Administrator/project/naca/raw_data/'
    new_data_path = 'C:/Users/Administrator/project/naca/data/'
    # types = ['0006', '0008', '0009', '0010', '0012', '0015', '0018', '0021', '0024', '1408', '1410', '1412', '2408',
    #          '2410', '2411', '2412', '2414', '2415', '2418', '2421', '2424', '4412', '4415', '4418', '4421', '4424',
    #          '6409', '6412', '22112', '23012', '23015', '23018', '23021', '23024', '23112', '24112', '25112', ]
    types = ['23015']
    rows = 401
    extraction_ice_shape(raw_data_path, new_data_path, types, rows)

    # 将 x-y 坐标数据转换为 ξ-η 坐标数据，然后将其拟合成傅里叶级数，并保存为 csv 文件
    # save_fourier_coeffient(types)

    # 读入预测的傅里叶级数，将其通过反傅里叶变换转换为 ξ-η 坐标数据，进一步转换为 x-y 坐标数据，并绘制冰型图
    # test = np.genfromtxt('./output/test.txt', delimiter='\t')
    # convert_coordinate_system_inversed('./data/pingwei4.45/body/556656.csv', './data/pingwei4.45/foil/556656.csv', '', test[0], test[1:31], test[31:61], test[61:63])

    # new_data_path = './feature/'
    # save_glaze_ice(raw_data_path, new_data_path, types, rows)
