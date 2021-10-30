import cv2
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, atan
from scipy import interpolate
from scipy.interpolate import interp1d
from FourierCoefficient import *
from numba import jit
import os


# matplotlib.use('module://backend_interagg')


@jit
def my_ceil(obj, precision=4):
    """
    将所给数据向上取整，默认保留精度为 4 位小数

    :param obj: 取整对象
    :param precision: 精度
    :return: 取整后的对象
    """
    return np.true_divide(np.ceil(obj * 10 ** precision), 10 ** precision)


@jit
def my_floor(obj, precision=4):
    """
    将所给数据向下取整，默认保留精度为 4 位小数

    :param obj: 取整对象
    :param precision: 精度
    :return: 取整后的对象
    """
    return np.true_divide(np.floor(obj * 10 ** precision), 10 ** precision)


def check_dir(paths):
    """
    检查路径里的文件夹是否存在，若不存在，则按层级依次新建文件夹
    :param paths: 路径 list
    :return:
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


@jit
def generate_sequence(
        alpha=np.arange(1, 4),  # [1-3]
        velocity=np.arange(1, 6),  # [1-5]
        temperature=np.arange(1, 6),  # [1-5]
        lwc=np.arange(1, 5),  # [1-4]
        mvd=np.arange(1, 5),  # [1-4]
):
    """
    生成结冰数据的文件名序列

    :param alpha: 攻角
    :param velocity: 飞行速度
    :param temperature: 温度
    :param lwc: 液态水含量
    :param mvd: 水滴平均直径
    :return: 文件名序列
    time = 1800s
    """
    sequence = []
    for a in alpha:
        for b in velocity:
            for c in temperature:
                for d in lwc:
                    for e in mvd:
                        sequence.append(a * 10000 + b * 1000 + c * 100 + d * 10 + e)
    return sequence


# @jit
def generate_input(new_data_path):
    """
    输入参数的全排列，并保存到 csv 文件中
    """
    alpha = [-3, 1, 3]
    velocity = [60, 75, 90, 105, 125]
    temperature = [-25, -20, -15, -10, -5]
    lwc = [0.2, 0.35, 0.55, 0.8]
    mvd = [10, 15, 20, 30]

    data, foil_paras, input_mlp, img_path, foil_path = [], [], [], [], []
    types = os.listdir(new_data_path)
    for type in types:
        for a in alpha:
            for b in velocity:
                for c in temperature:
                    for d in lwc:
                        for e in mvd:
                            data.append([a, b, c, d, e])
                            if len(type) == 4:
                                foil_paras.append([type[0], type[1], type[2:]])
                                index = str(alpha.index(a) + 1) + str(velocity.index(b) + 1) \
                                        + str(temperature.index(c) + 1) + str(lwc.index(d) + 1) \
                                        + str(mvd.index(e) + 1)
                                foil_path.append([type, index])
                                input_mlp.append([a, b, c, d, e])
                            img_path.append(type + '.png')
    df = pd.DataFrame(data)
    df.to_csv('./data/input.csv', sep=',', index=False, header=False)
    df = pd.DataFrame(foil_paras)
    df.to_csv('./data/foil_paras.csv', sep=',', index=False, header=False)
    df = pd.DataFrame(input_mlp)
    df.to_csv('./data/input_mlp.csv', sep=',', index=False, header=False)
    df = pd.DataFrame(img_path)
    df.to_csv('./data/img_path.csv', sep=',', index=False, header=False)
    df = pd.DataFrame(foil_path)
    df.to_csv('./data/foil_path_mlp.csv', sep=',', index=False, header=False)


@jit
def load_data(data_rows, annotation_rows, data_path):
    """
    加载数据，提取冰型数据，以及结冰部分翼型数据

    :param data_rows: 翼型数据行数
    :param annotation_rows: 数据中注释所占行数，需跳过读取
    :param data_path: 数据路径
    :return ice: 冰型数据
    :return: foil 结冰部分的翼型数据
    """

    # 翼型
    body = np.loadtxt(data_path,
                      skiprows=annotation_rows,
                      max_rows=data_rows)
    # 翼型+冰型
    body_plus_ice = np.loadtxt(data_path,
                               skiprows=data_rows + annotation_rows * 2,
                               max_rows=data_rows)

    # plt.plot(body[:, 0], body[:, 1], 'blue', linestyle=':')
    # plt.plot(body_plus_ice[:, 0], body_plus_ice[:, 1], 'black', linestyle='--')
    # plt.show()

    ice = []
    foil = []
    rows = body.shape[0]
    # 提取冰型
    for i in range(1, rows - 1):
        # 冰型的上极限点
        is_first_point = (body[i, 0] == body_plus_ice[i, 0] and body[i, 1] == body_plus_ice[i, 1]) \
                         and (body[i + 1, 0] != body_plus_ice[i + 1, 0] and body[i + 1, 1] != body_plus_ice[i + 1, 1])
        # 冰型中间点
        is_middle_point = body[i, 0] != body_plus_ice[i, 0] and body[i, 1] != body_plus_ice[i, 1]

        # 冰型的下极限点
        is_last_point = (body[i - 1, 0] != body_plus_ice[i - 1, 0] and body[i - 1, 1] != body_plus_ice[i - 1, 1]) \
                        and (body[i, 0] == body_plus_ice[i, 0] and body[i, 1] == body_plus_ice[i, 1])
        # 这三种点都需要保留
        condition = is_first_point or is_middle_point or is_last_point
        if condition:
            # 冰型
            ice.append([body_plus_ice[i, 0], body_plus_ice[i, 1]])
            # 结冰部分的翼型
            foil.append([body[i, 0], body[i, 1]])

    start = np.where((body == foil[0]).all(axis=1))[0][0]
    end = np.where((body == foil[-1]).all(axis=1))[0][0]
    foil = body[start:end + 1, :]

    start = np.where((body_plus_ice == ice[0]).all(axis=1))[0][0]
    end = np.where((body_plus_ice == ice[-1]).all(axis=1))[0][0]
    ice = body_plus_ice[start:end + 1, :]

    return ice, foil, body


@jit
def extraction_ice_shape(raw_data_path, new_data_path):
    """
    提取冰型及结冰部分的翼型，分别保存为 csv 并生成图像

    :param raw_data_path: 源文件地址
    :param new_data_path: 生成数据保存地址
    :param types: 翼型类型
    :param rows: 需要读取的行数
    """
    rows = 401
    types = os.listdir(raw_data_path)
    sequence = generate_sequence()
    for type in types:
        for number in sequence:
            print(type, number)
            ice, foil, body = load_data(data_rows=rows,
                                        annotation_rows=2,
                                        data_path=raw_data_path + type + '/' + str(number) + '.dat')
            df_ice = pd.DataFrame(ice)
            df_foil = pd.DataFrame(foil)
            df_body = pd.DataFrame(body)
            data_path = new_data_path + type + '/'
            path_ice = data_path + 'ice/'
            path_foil = data_path + 'foil/'
            path_body = data_path + 'body/'
            path_img = data_path + 'img/'
            check_dir([path_ice, path_foil, path_body, path_img])

            df_ice.to_csv(path_ice + str(number) + '.csv', sep=',', index=False, header=False)
            df_foil.to_csv(path_foil + str(number) + '.csv', sep=',', index=False, header=False)
            df_body.to_csv(path_body + str(number) + '.csv', sep=',', index=False, header=False)

            plt.plot(ice[:, 0], ice[:, 1], 'red')
            plt.plot(foil[:, 0], foil[:, 1], 'green')
            plt.legend(['ice', 'foil'])
            plt.savefig(path_img + str(number) + '.png')
            plt.clf()


# @jit
def convert_coordinate_system(full_foil_path, partial_foil_path, ice_path):
    """
    将机翼结冰数据从 x-y 坐标系转换到 ξ-η 坐标系

    :param full_foil_path: 翼型的整体数据
    :param partial_foil_path: 结冰部分的翼型数据
    :param ice_path: 冰型数据
    :return: ksi 翼型横坐标
    :return: eta 翼型纵坐标
    """
    full_foil = np.loadtxt(full_foil_path, delimiter=',')
    partial_foil = np.loadtxt(partial_foil_path, delimiter=',')
    ice = np.loadtxt(ice_path, delimiter=',')

    # 如果是空文件，则直接返回 None
    if len(partial_foil) == 0 or len(ice) == 0:
        return None, None

    full_foil_x, full_foil_y = full_foil[:, 0], full_foil[:, 1]
    foil_x, foil_y = partial_foil[:, 0], partial_foil[:, 1]
    ice_x, ice_y = ice[:, 0], ice[:, 1]
    length = len(partial_foil)

    # 计算冰厚 η(eta)
    eta = np.zeros(length)
    for i in range(length):
        eta[i] = sqrt((ice_x[i] - foil_x[i]) ** 2 + (ice_y[i] - foil_y[i]) ** 2)
        if eta[i] < 1e-4:
            eta[i] = 0

    # 计算翼型相邻两点之间的总距离（累加得到）
    d, dis = np.zeros(len(full_foil)), np.zeros(len(full_foil))
    for i in range(len(full_foil)):
        if i > 0:
            d[i] = sqrt((full_foil_x[i] - full_foil_x[i - 1]) ** 2 + (full_foil_y[i] - full_foil_y[i - 1]) ** 2)
            dis[i] = dis[i - 1] + d[i]

    # 计算结冰部分翼型 ξ(ksi) 坐标
    # 计算翼型 x-y 坐标原点（最左端点）
    origin = len(full_foil) // 2
    # 结冰部分的翼型坐标起点，即 partial_foil 第一个坐标在 full_foil 中出现的位置
    starting_point = np.where((full_foil == partial_foil[0]).all(axis=1))[0][0]

    # 保存翼型 ξ(ksi) 坐标
    ksi = np.zeros(length)
    for i in range(length):
        ksi[i] = dis[i + starting_point] - dis[origin]
        # 上翼面弧长为正，下翼面弧长为负，故需要取反
        ksi[i] = -ksi[i]

    return ksi, eta


# @jit
def convert_coordinate_system_inversed(data_path, seq_path, foil_info_path):
    """
    将机翼结冰数据从 ξ-η 坐标系转换到  x-y 坐标系

    :param data: 预测生成的傅里叶系数
    :return:
    """
    # constant: 预测的傅里叶系数常数
    # an: 预测的傅里叶系数余弦项
    # bn: 预测的傅里叶系数正弦项
    # limit: 预测的结冰上下极限点

    data = np.loadtxt(data_path, delimiter=',')
    seq = np.loadtxt(seq_path, delimiter=',', dtype=int)
    foil_path = np.loadtxt(foil_info_path, delimiter=',', dtype=str)
    total_samples = data.shape[0]
    foil_path = foil_path[seq[-total_samples:]]
    original_data_path = 'D:/Project/data/naca/data/'

    for index in range(total_samples):
        constant, an, bn, limit = data[index, 0], data[index, 1:31], data[index, 31:61], data[index, 61:63]
        foil_name, foil_index = foil_path[index, 0], foil_path[index, 1]

        full_foil_path = original_data_path + foil_name + '/body/' + foil_index + '.csv'
        partial_foil_path = original_data_path + foil_name + '/foil/' + foil_index + '.csv'
        ground_truth_path = original_data_path + foil_name + '/ice/' + foil_index + '.csv'
        full_foil = np.loadtxt(full_foil_path, delimiter=',')
        partial_foil = np.loadtxt(partial_foil_path, delimiter=',')
        ground_truth = np.loadtxt(ground_truth_path, delimiter=',')

        full_foil_x, full_foil_y = full_foil[:, 0], full_foil[:, 1]
        foil_x, foil_y = partial_foil[:, 0], partial_foil[:, 1]
        ground_truth_x, ground_truth_y = ground_truth[:, 0], ground_truth[:, 1]
        length = len(partial_foil)

        # 计算翼型相邻两点之间的总距离（累加得到）
        d, dis = np.zeros(len(full_foil)), np.zeros(len(full_foil))
        for i in range(len(full_foil)):
            if i > 0:
                d[i] = sqrt((full_foil_x[i] - full_foil_x[i - 1]) ** 2 + (full_foil_y[i] - full_foil_y[i - 1]) ** 2)
                dis[i] = dis[i - 1] + d[i]

        # 计算翼型 x-y 坐标原点（最左端点）
        origin = len(full_foil) // 2
        # 结冰部分的翼型坐标起点，即结冰部分翼型数据的第一个坐标在全部翼型数据中出现的位置
        starting_point = np.where((full_foil == partial_foil[0]).all(axis=1))[0][0]

        # 保存翼型弧度坐标
        radian = np.zeros(length)
        for i in range(length):
            radian[i] = dis[i + starting_point] - dis[origin]
            # 上翼面弧长为正，下翼面弧长为负，故需要取反
            radian[i] = -radian[i]

        # 获取结冰的上下极限点
        ice_limit_upper = limit[0]
        ice_limit_lower = limit[1]

        # 按照 0.00001 间距插值
        x = np.linspace(ice_limit_lower, ice_limit_upper, int((ice_limit_upper - ice_limit_lower) / 0.00001))

        # 反傅里叶变换，得到冰型曲线
        predicted_ice = inverse_fourier(constant, an, bn, x)

        # 将冰型曲线小于 0 的部分重置为 0
        _, cols = np.where(predicted_ice.reshape((1, len(predicted_ice))) < 0)
        for col in cols:
            predicted_ice[col] = 0

        # 三次样条插值，得到插值后的冰型 ξ-η 坐标曲线，并处理异常数据（小于 0 的部分重置为 0）
        tck = interpolate.splrep(x, predicted_ice)
        bspline = interpolate.splev(radian, tck)
        _, cols = np.where(bspline.reshape((1, len(bspline))) < 0)
        for col in cols:
            bspline[col] = 0

        # 将冰型数据以及翼型数据转换到 x-y 坐标系中，并将两者结合
        ice_x, ice_y = np.zeros(length), np.zeros(length)
        for i in range(length):
            if i == 0:
                ice_x[i] = foil_x[i] + bspline[i] * (foil_y[i] - full_foil_y[i - 1 + starting_point]) / d[
                    i + starting_point]
                ice_y[i] = foil_y[i] + bspline[i] * (full_foil_x[i - 1 + starting_point] - foil_x[i]) / d[
                    i + starting_point]
            else:
                ice_x[i] = foil_x[i] + bspline[i] * (foil_y[i] - foil_y[i - 1]) / d[i + starting_point]
                ice_y[i] = foil_y[i] + bspline[i] * (foil_x[i - 1] - foil_x[i]) / d[i + starting_point]

        plt.gcf().set_size_inches(8, 6)
        plt.plot(ice_x, ice_y, linestyle='--', linewidth=3)
        plt.plot(ground_truth_x, ground_truth_y, linestyle='-.')
        plt.plot(foil_x, foil_y, color='gray')
        plt.legend(['predicted', 'ground truth', 'foil ' + foil_name])
        plt.savefig('./output/img/' + foil_name + '-' + foil_index + '.png', dpi=100)
        # plt.show()
        plt.clf()


@jit
def generate_fourier_coefficient(ksi, eta):
    """
    将 ξ-η 坐标数据，拟合成傅里叶级数，返回一个 1*63 的 numpy 数组

    :param ksi: 翼型横坐标
    :param eta: 翼型纵坐标
    :return: None
    """
    # 获取结冰的上下极限点
    ice_limit_upper = ksi[0]
    ice_limit_lower = ksi[-1]

    # 按照 0.00001 间距插值
    new_ksi = np.linspace(ice_limit_lower, ice_limit_upper, int((ice_limit_upper - ice_limit_lower) / 0.00001))
    function = interp1d(ksi, eta, kind='slinear')
    new_eta = function(new_ksi)

    plt.plot(ksi, eta)
    plt.plot(new_ksi, new_eta, linestyle='--')
    plt.xlabel('ξ')
    plt.ylabel('η')

    constant, an, bn = fourier(new_eta, new_ksi, 30)
    all = np.zeros(63)
    all[0] = constant
    all[1:31] = an
    all[31:61] = bn
    all[61] = ice_limit_upper
    all[62] = ice_limit_lower
    return all.reshape((1, 63))

    # fx = inverse_fourier(constant, an, bn, new_ksi)
    # plt.plot(new_ksi, fx)
    # plt.legend(['old', 'new'])
    # plt.show()


@jit
def save_fourier_coeffient(new_data_path):
    """
    保存傅里叶级数到 csv 文件

    :return: None
    """
    sequence = generate_sequence()
    output, output_mlp = [], []
    types = os.listdir(new_data_path)
    for type in types:
        print(type)
        for number in sequence:
            # 转换坐标系
            full_foil_path = new_data_path + type + '/body/' + str(number) + '.csv'
            partial_foil_path = new_data_path + type + '/foil/' + str(number) + '.csv'
            ice_path = new_data_path + type + '/ice/' + str(number) + '.csv'
            ksi, eta = convert_coordinate_system(full_foil_path, partial_foil_path, ice_path)
            if ksi is None or eta is None:
                output.append([])
                print('error ', type)
                continue

            # 将坐标数据展开为傅里叶级数
            coefficient = generate_fourier_coefficient(ksi, eta)
            output.append(coefficient)
            if len(type) == 4:
                output_mlp.append(coefficient)

    df = pd.DataFrame(np.concatenate(output))
    df.to_csv('data/output.csv', sep=',', index=False, header=False)
    df = pd.DataFrame(np.concatenate(output_mlp))
    df.to_csv('data/output_mlp.csv', sep=',', index=False, header=False)


@jit
def get_stationary_point(ksi, eta, threshold=7):
    """
    获取冰型驻点，若算法获得多个，则取最靠近零点的那一个

    :param ksi: 横坐标
    :param eta: 纵坐标
    :param threshold: 驻点阈值，用于确定驻点
    :return: 最靠近零点的冰型驻点
    """
    points = []
    for i in range(threshold, len(ksi) - threshold):
        status = np.zeros(threshold)
        for j in range(0, threshold):
            if eta[i - j - 1] > eta[i] < eta[i + j + 1]:
                status[j] = 1
        if np.sum(status) == threshold:
            points.append([ksi[i], eta[i]])
    if len(points) == 0:
        return None
    else:
        points = np.array(points)
        points = sorted(points, key=lambda row: np.abs(row[0]))
        return points[0]


@jit
def generate_feature(types, radius, angle):
    sequence = generate_sequence()
    for index in range(len(types)):
        list = []
        aoa = []
        if index <= 3:
            aoa = [-4, -2, 0, 2, 4]
        else:
            aoa = [0, 2, 4, 6, 7]
        for number in sequence:
            print(types[index], number)
            full_foil_path = './body/' + types[index] + '/body/' + str(number) + '.csv'
            partial_foil_path = './body/' + types[index] + '/foil/' + str(number) + '.csv'
            ice_path = './body/' + types[index] + '/ice/' + str(number) + '.csv'
            ksi, eta = convert_coordinate_system(full_foil_path, partial_foil_path, ice_path)

            # 找冰型驻点，如果为空则返回
            if ksi is None or eta is None:
                continue
            point = get_stationary_point(ksi, eta, threshold=7)
            if point is None:
                continue

            # 求驻点、上下冰角点及其高度
            stationary_point = np.where(ksi == point[0])[0][0]  # 驻点
            stationary_ice_horn = eta[stationary_point]
            upper_ice_horn = np.max(eta[:stationary_point])  # 上冰角
            upper_index = np.argmax(eta[:stationary_point])
            lower_ice_horn = np.max(eta[stationary_point:])  # 下冰角
            lower_index = np.argmax(eta[stationary_point:]) + stationary_point

            full_foil = np.loadtxt(full_foil_path, delimiter=',')
            partial_foil = np.loadtxt(partial_foil_path, delimiter=',')
            full_foil_x, full_foil_y = full_foil[:, 0], full_foil[:, 1]
            length = len(partial_foil)

            # 计算翼型相邻两点之间的总距离（累加得到）
            d, dis = np.zeros(len(full_foil)), np.zeros(len(full_foil))
            for i in range(len(full_foil)):
                if i > 0:
                    d[i] = sqrt((full_foil_x[i] - full_foil_x[i - 1]) ** 2 + (full_foil_y[i] - full_foil_y[i - 1]) ** 2)
                    dis[i] = dis[i - 1] + d[i]

            # 计算翼型 x-y 坐标原点（最左端点）
            origin = len(full_foil) // 2
            # 结冰部分的翼型坐标起点，即结冰部分翼型数据的第一个坐标在全部翼型数据中出现的位置
            starting_point = np.where((full_foil == partial_foil[0]).all(axis=1))[0][0]

            # 保存翼型弧度坐标
            radian = np.zeros(length)
            for i in range(length):
                radian[i] = dis[i + starting_point] - dis[origin]
                # 上翼面弧长为正，下翼面弧长为负，故需要取反
                radian[i] = -radian[i]

            # 获取结冰的上下极限点
            ice_limit_upper = ksi[0]
            ice_limit_lower = ksi[-1]

            # 按照 0.00001 间距插值
            new_ksi = np.linspace(ice_limit_lower, ice_limit_upper, int((ice_limit_upper - ice_limit_lower) / 0.00001))
            function = interp1d(ksi, eta, kind='slinear')
            new_eta = function(new_ksi)

            # 求结冰区域面积
            area = 0
            length = len(new_ksi)
            for i in range(length - 1):
                area += (new_ksi[i + 1] - new_ksi[i]) * (new_eta[i + 1] + new_eta[i]) / 2

            # 将三个极值点画到 x-y 坐标系
            foil = np.loadtxt(partial_foil_path, delimiter=',')
            ice = np.loadtxt(ice_path, delimiter=',')
            staionary_x, stationary_y = ice[stationary_point][0], ice[stationary_point][1]
            upper_horn_x, upper_horn_y = ice[upper_index][0], ice[upper_index][1]
            lower_horn_x, lower_horn_y = ice[lower_index][0], ice[lower_index][1]

            # 三个极值点在机翼上对应的坐标
            foil_staionary_x, foil_staionary_y = foil[stationary_point][0], foil[stationary_point][1]
            foil_upper_horn_x, foil_upper_horn_y = foil[upper_index][0], foil[upper_index][1]
            foil_lower_horn_x, foil_lower_horn_y = foil[lower_index][0], foil[lower_index][1]

            # 获取机翼原点
            origin_x = full_foil_x[(len(full_foil_x) - 1) // 2]
            origin_y = full_foil_y[(len(full_foil_y) - 1) // 2]

            # 获取机翼前缘点
            leading_edge_point_x = radius[index] * cos((angle[index] + aoa[number // 100000 - 1]) / 180)
            leading_edge_point_y = origin_y - radius[index] * sin((angle[index] + aoa[number // 100000 - 1]) / 180)
            AssertionError(leading_edge_point_x >= 0)

            # 求上冰角角度，以机翼前缘点为新的原点，划分四个象限，对四个象限中的冰角分开计算
            if upper_horn_y >= leading_edge_point_y:
                if upper_horn_x >= leading_edge_point_x:  # 第一象限
                    theta_upper = atan(
                        (upper_horn_y - leading_edge_point_y) / (upper_horn_x - leading_edge_point_x)) / pi * 180
                else:  # 第二象限
                    theta_upper = 180 - atan(
                        (upper_horn_y - leading_edge_point_y) / (leading_edge_point_x - upper_horn_x)) / pi * 180
            else:
                if upper_horn_x < leading_edge_point_x:  # 第三象限
                    theta_upper = 180 + atan(
                        (leading_edge_point_y - upper_horn_y) / (leading_edge_point_x - upper_horn_x)) / pi * 180
                else:  # 第四象限
                    theta_upper = 360 - atan(
                        (leading_edge_point_y - upper_horn_y) / (upper_horn_x - leading_edge_point_x)) / pi * 180

            # 求下冰角角度
            if lower_horn_y >= leading_edge_point_y:
                if lower_horn_x >= leading_edge_point_x:  # 第一象限
                    theta_lower = atan(
                        (lower_horn_y - leading_edge_point_y) / (lower_horn_x - leading_edge_point_x)) / pi * 180
                else:  # 第二象限
                    theta_lower = 180 - atan(
                        (lower_horn_y - leading_edge_point_y) / (leading_edge_point_x - lower_horn_x)) / pi * 180
            else:
                if lower_horn_x < leading_edge_point_x:  # 第三象限
                    theta_lower = 180 + atan(
                        (leading_edge_point_y - lower_horn_y) / (leading_edge_point_x - lower_horn_x)) / pi * 180
                else:  # 第四象限
                    theta_lower = 360 - atan(
                        (leading_edge_point_y - lower_horn_y) / (lower_horn_x - leading_edge_point_x)) / pi * 180

            list.append([number, upper_ice_horn, lower_ice_horn, stationary_ice_horn, ice_limit_upper, ice_limit_lower,
                         theta_upper, theta_lower, area])

            # plt.subplot(1, 2, 1)
            # plt.plot(ice[:, 0], ice[:, 1], 'red')
            # plt.plot(foil[:, 0], foil[:, 1], 'green')
            # plt.plot(staionary_x, stationary_y, 'o', color='brown')
            # plt.plot(foil_staionary_x, foil_staionary_y, 'o', color='brown')
            # plt.plot(upper_horn_x, upper_horn_y, 'o', color='c')
            # plt.plot(foil_upper_horn_x, foil_upper_horn_y, 'o', color='c')
            # plt.plot(lower_horn_x, lower_horn_y, 'o', color='orange')
            # plt.plot(foil_lower_horn_x, foil_lower_horn_y, 'o', color='orange')
            # plt.plot([staionary_x, foil_staionary_x], [stationary_y, foil_staionary_y], '--', color='brown')
            # plt.plot([upper_horn_x, foil_upper_horn_x], [upper_horn_y, foil_upper_horn_y], '--', color='c')
            # plt.plot([lower_horn_x, foil_lower_horn_x], [lower_horn_y, foil_lower_horn_y], '--', color='orange')
            #
            # # 将三个极值点画到 ξ-η 坐标系
            # plt.subplot(1, 2, 2)
            # plt.plot(ksi, eta)
            # plt.plot(point[0], point[1], 'o', color='brown')
            # plt.plot(ksi[upper_index], eta[upper_index], 'o', color='c')
            # plt.plot(ksi[lower_index], eta[lower_index], 'o', color='orange')
            # plt.show()

        path = './output/feature/' + types[index] + '.csv'
        columns = ['编号', '上冰角高度', '下冰角高度', '驻点高度', '上极限', '下极限', '上冰角', '下冰角', '面积']
        df = pd.DataFrame(list, columns=columns)
        df.to_csv(path, sep=',', index=False, encoding='utf_8_sig')


def generate_foil_images(path):
    """
    生成翼型数据
    """
    types = os.listdir(path)
    for type in types:
        foil = np.loadtxt('./data/body/' + type + '.csv', delimiter=',')
        x, y = foil[:, 0], foil[:, 1] * 10
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.gcf().set_size_inches(10, 10)
        plt.axis('off')
        plt.plot(x, y, color='w', linewidth=5)
        # plt.title(type)
        # plt.show()
        # plt.fill_between(x, y, color='w')
        plt.savefig('./data/img/foil/' + type + '.png', dpi=100, facecolor='black', bbox_inches='tight', pad_inches=0)
        img = cv2.imread('./data/img/foil/' + type + '.png')
        img = cv2.resize(img, (512, 512))
        cv2.imwrite('./data/img/foil/' + type + '.bmp', img)
        os.remove('./data/img/foil/' + type + '.png')
        plt.clf()

@jit
def generate_ice_img(raw_data_path, new_data_path):
    foils = os.listdir(new_data_path)
    for foil in foils:
        files = os.listdir(raw_data_path + '/' + foil)
        for file in files:
            ice = np.loadtxt(raw_data_path + '/' + foil + '/' + file, skiprows=405, max_rows=401)
            x, y = ice[:, 0], ice[:, 1]
            plt.gca().set_aspect('equal')
            plt.gcf().set_size_inches(6, 4)
            plt.xlim(-0.2, 0.4)
            plt.ylim(-0.2, 0.2)
            plt.axis('off')
            plt.plot(x, y, color='w', linewidth=3)
            # plt.show()
            save_path = './data/img/' + foil + '/'
            filename = file.split('.')[0]
            check_dir([save_path])
            plt.savefig(save_path + filename + '.png', dpi=100, facecolor='black', bbox_inches='tight', pad_inches=0)
            img = cv2.imread(save_path + filename + '.png')
            img = cv2.resize(img, (300, 200))
            cv2.imwrite(save_path + filename + '.bmp', img)
            os.remove(save_path + filename + '.png')
            plt.clf()
