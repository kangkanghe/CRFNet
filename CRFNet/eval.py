import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils import data
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from dataset_data import Dataset_test, Test_write
import argparse
from torch.utils.data import TensorDataset
from load_data import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams["font.family"] = "Times New Roman"
def plot_confusion_matrix(cm,labels, title='Confusion Matrix of PDNet'):
    # font1 = {'family': 'Times New Roman',
    #          'size':50}
    # font2 = {'family': 'Times New Roman',
    #          'size':35}
    # font3 = {'family': 'Times New Roman'}
    plt.imshow(cm)   #  , interpolation='nearest', cmap=plt.cm.binary
    # plt.title(title,fontsize=80) # ,fontfamily='Times New Roman')
    # plt.colorbar().ax.tick_params(labelsize=50)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=45,fontsize=50) # ) # ,fontfamily='Times New Roman')
    plt.yticks(xlocations, labels,fontsize=50) # ,fontfamily='Times New Roman')
    plt.ylabel('True label',fontsize=50) # ,fontfamily='Times New Roman')
    plt.xlabel('Predicted label',fontsize=50) # ,fontfamily='Times New Roman')

def draw(y_true,y_pred,labels):
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    # np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(60, 60), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=70, va='center', ha='center') # ) # ,fontfamily='Times New Roman') 50
        else:
            plt.text(x_val, y_val, 0, color='red', fontsize=70, va='center', ha='center') # ) # ,fontfamily='Times New Roman') 50
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15) # 0.15

    plot_confusion_matrix(cm_normalized, labels, title='Confusion matrix of the proposed FusionMixer')
    # show confusion matrix
    plt.savefig('Confusion_FusionMixer_Houston.png', format='png')
    plt.close()


class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def main(args):

    # input_landsat = r'D:\Users\kang_\Desktop\CNN_Traff\data_all\four\Landsat-8.tif'
    # input_traffic = r'D:\Users\kang_\Desktop\CNN_Traff\data_all\four\traffic_3.tif'
    # input_all = r'D:\Users\kang_\Desktop\CNN_Traff\data_all\four\all_files.tif'
    # dataset_stype = 'DB'
    # image_size = 17

    input_landsat = r'D:\Users\kang_\Desktop\CNN_Traff\data_all\first\Landsat-8.tif'
    input_traffic = r'D:\Users\kang_\Desktop\CNN_Traff\data_all\first\traffic_3.tif'
    input_all = r'D:\Users\kang_\Desktop\CNN_Traff\data_all\first\all_files.tif'
    dataset_stype = 'JBT'
    image_size = 19

    dataset = Dataset_test(image_size, input_landsat, input_traffic, input_all, dataset_stype)
    test_dataset = TensorDataset(dataset['total']['data'][0], dataset['total']['data'][1], dataset['total']['data'][2], torch.tensor(dataset['total']['points']))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # models_complate = [
    #     # {"name": "TwoBranchCNN", "model": '.\\model-pth-NEcomplete\\17_TwoBranchCNN-DB-2.pth'},
    #     {"name": "FusionMixer", "model": '.\\model-pth-NEcomplete\\17_FusionMixer-DB-Modules-1.pth'},
    #     # {"name": "MFT", "model": '.\\model-pth-NEcomplete\\17-0530-MFT-2.pth'},
    #     # {"name": "MViT", "model": '.\\model-pth-NEcomplete\\17-0530-MViT-DB-2.pth'},
    #     # {"name": "CRFNet", "model": '.\\model-pth-NEcomplete\\17-0530_Cross_Modal_Mixer-DB-1.pth'}
    # ]

    models_complate = [
        # {"name": "TwoBranchCNN", "model": '.\\model-pth-JBTcomplete\\19_TwoBranchCNN-JBT-1.pth'},
        {"name": "FusionMixer", "model": '.\\model-pth-JBTcomplete\\19_FusionMixer-JBT-Modules-1.pth'},
        # {"name": "MFT", "model": '.\\model-pth-JBTcomplete\\19_MFT-JBT-1.pth'},
        # {"name": "MViT", "model": '.\\model-pth-JBTcomplete\\19_MViT-JBT-1.pth'},
        # {"name": "CRFNet", "model": '.\\model-pth-JBTcomplete\\19_Cross-Modal Mixer-JBT-3.pth'}
    ]


    # models_missing1 = [
    #     {"name": "TwoBranchCNN", "model": '.\\model-pth-NEimcomplete\\17_TwoBranchCNN-DB-1.pth'},
    #     {"name": "FusionMixer", "model": '.\\model-pth-NEimcomplete\\17_FusionMixer-DB-1.pth'},
    #     {"name": "MFT", "model": '.\\model-pth-NEimcomplete\\17_MFT-DB-1.pth'},
    #     {"name": "MViT", "model": '.\\model-pth-NEimcomplete\\17_MViT-DB-1.pth'},
    #     {"name": "CRFNet", "model": '.\\model-pth-NEimcomplete\\17_Cross-Modal Mixer-DB-1.pth'}
    # ]
    # imcomplete_stype = 'im'
    # models_missing1 = [
    #     # {"name": "TwoBranchCNN", "model": '.\\model-pth-JBTimcomplete\\19_TwoBranchCNN-JBT-1.pth'},
    #     # {"name": "FusionMixer", "model": '.\\model-pth-JBTimcomplete\\19_FusionMixer-JBT-1.pth'},
    #     # {"name": "MFT", "model": '.\\model-pth-JBTimcomplete\\19_MFT-JBT-1.pth'},
    #     {"name": "MViT", "model": '.\\model-pth-JBTimcomplete\\19_MViT-JBT-1.pth'},
    #     {"name": "CRFNet", "model": '.\\model-pth-JBTimcomplete\\19_Cross-Modal Mixer-JBT-1.pth'}
    # ]
    # imcomplete_stype = 'im'



    for model_info in models_complate:
        model_name = model_info["name"]
        output_path = f"{os.path.splitext(input_traffic)[0]}_{model_name}_{os.path.splitext(input_traffic)[1]}"
        # output_path = f"{os.path.splitext(input_traffic)[0]}_{model_name}_{imcomplete_stype}_{os.path.splitext(input_traffic)[1]}"
        model = model_info["model"]
        model2 = torch.load(model)  # best_val_model_multi_4_10_POI
        print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))
        model2 = model2.to(device)
        print('start eval')
        print(model_name)
        model2.eval()
        # 获取坐标的最大值来确定矩阵大小
        all_points = np.vstack([data[3].numpy() for data in test_loader.dataset])
        max_x, max_y = np.max(all_points, axis=0)
        # 初始化二维矩阵
        result_matrix = np.zeros((max_x + 1, max_y + 1), dtype=int)  # +1因为索引是从0开始的

        true_label = []
        pred_label = []
        num_class = 3
        classes = ('1', '2', '3')
        class_correct2 = list(0. for i in range(num_class))
        class_total2 = list(0. for i in range(num_class))
        correct_prediction_2 = 0.
        total_2 = 0

        with torch.no_grad():
            for landsat, all_data, labels, points in test_loader:
                labels = labels.to(device).squeeze()
                all_data = all_data.to(device)
                landsat = landsat.to(device)
                points = points.cpu().numpy()
                if model_name == 'CRFNet':
                    scenario = 'val'
                    pred_result = model2(landsat, all_data, scenario)
                elif model_name == 'MFT':
                    pred_result = model2(all_data, landsat)
                else:
                    pred_result = model2(landsat, all_data)

                _2, pred = torch.max(pred_result, 1)

                pred_label.append(pred)
                true_label.append(labels)
                c2 = (pred == labels).squeeze()
                for label_idx in range(len(labels)):
                    label = labels[label_idx]
                    class_correct2[int(label.item())] += c2[label_idx].item()
                    class_total2[int(label)] += 1
                total_2 += labels.size(0)
                # add correct
                correct_prediction_2 += (pred == labels).sum().item()

                pred = pred.cpu().numpy()
                for point, prediction in zip(points, pred):
                    x, y = point
                    result_matrix[x, y] = prediction  # 填充预测值

            t_l = torch.cat(true_label, dim=0)
            p_l = torch.cat(pred_label, dim=0)
            t_l = t_l.cpu().numpy()
            p_l = p_l.cpu().numpy()
        # for i in range(args.num_class):
        #     print('Model ResNet50 - Accuracy of %5s : %2f%%: Correct Num: %d in Total Num: %d' % (
        #         classes[i], 100 * class_correct1[i] / class_total1[i], class_correct1[i], class_total1[i]))
        # acc_1 = correct_prediction_1 / total_1
        # print("Total Acc Model ResNet50: %.4f" % (correct_prediction_1 / total_1))
        print('----------------------------------------------------')
        for i in range(3):
            print('Model - Accuracy of %5s : %2f%%: Correct Num: %d in Total Num: %d' % (
                classes[i], 100 * class_correct2[i] / class_total2[i], class_correct2[i], class_total2[i]))
        acc_2 = correct_prediction_2 / total_2
        print("Total Acc Model: %.4f" % (correct_prediction_2 / total_2))
        print('----------------------------------------------------')
        print(t_l, p_l)
        Test_write(output_path, result_matrix, input_traffic)
        draw(t_l, p_l, classes)
        print(cohen_kappa_score(t_l, p_l))
        print(accuracy_score(t_l, p_l))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=2, type=int)
    # parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    args = parser.parse_args()
    main(args)

