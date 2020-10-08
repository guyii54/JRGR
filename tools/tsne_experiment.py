import numpy as np
from sklearn.manifold import TSNE
from sklearn import datasets
from time import time
import matplotlib.pyplot as plt
import os
import fnmatch
import cv2

ubuntu = 0
windows = 1
platform = windows

def get_data_example():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

def get_data(experiment_name, img_nums = 50):
    # all names
    # visualize_names = ['Os', 'Ot', 'Bs','Bt',
    #                    'pred_Bs', 'pred_Ot', 'pred_pred_Bs', 'pred_pred_Os',
    #                    'pred_Bt', 'pred_Os', 'pred_pred_Bt', 'pred_pred_Ot',
    #                    'pred_Rs', 'pred_Rst', 'pred_pred_Rt', 'pred_pred_Rts',
    #                    'pred_Rt', 'pred_Rts', 'pred_pred_Rs', 'pred_pred_Rst']
    visualize_names = ['Rs', 'Rt','pred_Rs','pred_Rt','pred_Rts','pred_Rst']
    print('getting datas......')
    if platform:
        result_path = r'G:\Projects\RainCycleGAN_cross\results'
        # result_path = r'D:\LowLevelforReal\RainCycleGAN_cross\results'
    else:
        # result_path = r'/media/solanliu/YYT2T/Projects/RainCycleGAN_cross/results'
        result_path = r'/home/solanliu/yeyuntong/RainCycleGAN_ubuntu/results'
    image_path = os.path.join(result_path, experiment_name, 'test_latest', 'images')
    name_list = os.listdir(image_path)
    name_list = fnmatch.filter(name_list, '*_pred_pred_Bs.png')
    count = 0
    img_example = cv2.imread(os.path.join(image_path, name_list[0]))
    feature_length = img_example.shape[0] * img_example.shape[1] * img_example.shape[2]
    datas = np.zeros((img_nums*len(visualize_names), feature_length))
    labels = []
    for visualize_name in visualize_names:
        for name in name_list[0:img_nums]:
            little_name = name.replace('pred_pred_Bs.png', '')
        # print(little_name)
            img = cv2.imread(os.path.join(image_path,little_name + visualize_name+'.png'))
            feature = np.reshape(img, feature_length)
            datas[count,:] = feature
            labels.append(visualize_name)
            count += 1
            # if count >= img_nums * len(visualize_names):
            #     break
    return datas, labels, count, feature_length, len(visualize_names)



def plot_embedding_example(data, label, title):
    color = {'Os': 1, 'Ot': 2, 'Bs': 3, 'Bt':4,
             'pred_Bt': 5, 'pred_Bs': 6, 'pred_Os': 7,'pred_Ot': 8}
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(color[label[i]]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show(fig)
    return fig


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def plot_embedding_2(data, n_target, label, title):
    color_map = {'Rs': 'b', 'Rt': 'b', 'pred_Rt': 'r', 'pred_Rs': 'r','pred_Rst':'g','pred_Rts':'g'}
    make_map = {'Rs': '*', 'Rt': 'D', 'pred_Rt': 'D', 'pred_Rs': '*','pred_Rst':'D','pred_Rts':'*'}
    label_map  = {'Rs': 'Syn rain', 'Rt': 'Real rain', 'pred_Rt': 'Pred real rain', 'pred_Rs': 'Decom syn rain','pred_Rst':'Gen real rain','pred_Rts':'Decom syn rain'}
    size = 25
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    n_point = int(data.shape[0]/n_target)
    for i_target in range(0,n_target):
        i_label = label[5 + i_target*n_point]
        data_x = data[i_target*n_point : (i_target+1) * n_point, 0]
        data_y = data[i_target*n_point : (i_target+1) * n_point, 1]
        color = color_map[i_label]
        m = make_map[i_label]
        label_m = label_map[i_label]
        plt.scatter(data_x, data_y, s=size, label=label_m, marker=m, c=color, cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.legend()
    plt.show()



def plot_embedding(data, label):
    color_map = {'Os': 1, 'Ot': 2, 'Bs': 3, 'Bt':4}
    m = {'Os': 'v', 'Ot': '^', 'Bs': 'D', 'Bt': 'p'}
    size_map = {'Os': 2, 'Ot': 2, 'Bs': 2, 'Bt': 2}
    cm = list(map(lambda x: m[x], label))
    # print(cm)
    color = list(map(lambda x: color_map[x], label))
    size = list(map(lambda x: size_map[x], label))
    size = 25
    # normalize
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    # fig = plt.figure()
    fig, ax = plt.subplots()
    mscatter(data[:,0], data[:,1], label= label, s=size, c=color, m=cm, ax=ax, cmap=plt.cm.RdYlBu)
    plt.legend()

    # plot
    # plt.xticks([])
    # plt.yticks([])
    plt.title('TSNE')

    plt.show()



def main():
    data, label, n_samples, n_features, n_target = get_data('lr1e-5', 150)
    print('Computing t-SNE embedding')
    print('num_images: %d\nnum_feature: %d' % (n_samples, n_features))
    print('data shape: ',data.shape)
    tsne = TSNE(n_components=2, init='pca', random_state=0, learning_rate=100, perplexity=10,n_iter=1000)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding_2(result, n_target, label, 'Visulization of rain layer feature')
    # plt.show(fig)


if __name__ == '__main__':
    main()
