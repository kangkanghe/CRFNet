import os
import dataset_data_first

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
from torch import nn, optim
from torch.utils import data
import argparse
from torch.utils.data import TensorDataset
from slideCut import slideCut, split_labels, datasets
from dataset_data_first import ClassifyDataset
# from dataset_data_second import ClassifyDataset
# from dataset_data_third import ClassifyDataset
# from models.model_fuse_2024_519 import MMFMixer_mission   # 预测包含模态缺失的
from models.Cross_Modal_Mixer import MMFMixer  # 预测是完整模态
from models.Blocks_Analysis import FRM_main, FMM_main, RestNet_3_main, RestNet_7_main, RestNet_3_7_main, FMM_FRM_main  # 预测是完整模态
from models.MFT import MFT
from models.MViT import MViT
from models.TwoBranchCNN import TwoBranchCNN, SingleBranchCNN, DualBranchResNet50, SingleBranchResNet50
from models.FusionMixer import MMF_MLPMixer
import datetime
import os
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score
import random
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis


def save_metrics(file, epoch, details, acc, kappa, precision, is_best=False):
    with open(file, "a") as f:
        f.write(f"Epoch {epoch}:\n")
        for detail in details:
            f.write(f"{detail}\n")
        f.write(f"Acc: {acc:.4f}, Kappa: {kappa:.4f}, Precision: {precision:.4f} sec\n")
        if is_best:
            f.write(f"New Best Acc: {acc:.4f}, Kappa: {kappa:.4f} at epoch {epoch}\n")
        f.write('----------------------------------------------------\n')


def main(args):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # 设置随机数种子
    setup_seed(100)  # 100
    # for image_size in range(5, 24, 2):  # 从3到27，间隔为2


    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

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


    # landsat是原始影像；7波段
    # all_files是地数据['DEM', 'slope', '地形湿度指数', '坡位指数', '水流力指数', '土地利用','地灾', '构造', '岩土体', 'RCI']；
    # train_ratios = [0.01, 0.03, 0.05, 0.07, 0.09]
    train_ratios = [0.01]
    for train_ratio in train_ratios:
        # dataset = ClassifyDataset(image_size, input_landsat, input_traffic, input_all, dataset_stype)
        dataset = ClassifyDataset(image_size, input_landsat, input_traffic, input_all, dataset_stype, train_ratio)

        # 三模态
        train_dataset = TensorDataset(dataset['train']['data'][0], dataset['train']['data'][1], dataset['train']['data'][2])
        test_dataset = TensorDataset(dataset['test']['data'][0], dataset['test']['data'][1], dataset['test']['data'][2])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

        ##########################################选择模型#########################################
        models_complate = [
            # {"name": "TwoBranchCNN", "model": TwoBranchCNN(num_classes=3)},
            # {"name": "SingleBranchCNN_7", "model": SingleBranchCNN(7, num_classes=3)},
            # {"name": "SingleBranchCNN_10", "model": SingleBranchCNN(10, num_classes=3)},
            {"name": "FusionMixer", "model": MMF_MLPMixer(args.num_class, image_size)},
            # {"name": "MFT", "model": MFT(16, 10, 7, 3, False)},
            # {"name": "MViT", "model": MViT(patch_size=image_size, num_patches=[7, 10], num_classes=3, dim=64, depth=6, heads=4, mlp_dim=32, dropout=0.1, emb_dropout=0.1, mode='MViT')},
            # {"name": "Cross-Modal Mixer", "model": MMFMixer(args.num_class, image_size)}
        ]

        # models_missing1 = [
        #     {"name": "Cross-Modal Mixer", "model": MMFMixer(args.num_class, image_size)},
        #     # {"name": "FusionMixer", "model": MMF_MLPMixer(args.num_class, image_size)},
        #     # {"name": "MViT", "model": MViT(patch_size=image_size, num_patches=[7, 10], num_classes=3, dim=64, depth=6, heads=4, mlp_dim=32,dropout=0.1, emb_dropout=0.1, mode='MViT')},
        #     # {"name": "MFT", "model": MFT(16, 10, 7, 3, False)},
        #     # {"name": "TwoBranchCNN", "model": TwoBranchCNN(num_classes=3)},
        # ]

        # # Block Analysis
        # Block_Analysis = [
        #     # {"name": "RestNet_landsat_3", "model": RestNet_3_main(args.num_class, image_size)},
        #     {"name": "RestNet_all_factor_10", "model": RestNet_7_main(args.num_class, image_size)},
        #     {"name": "RestNet_landsat_all_factor", "model": RestNet_3_7_main(args.num_class, image_size)},
        #     {"name": "FMM_FRM", "model": FMM_FRM_main(args.num_class, image_size)},
        #     {"name": "FRM", "model": FRM_main(args.num_class, image_size)},
        #     {"name": "FMM", "model": FMM_main(args.num_class, image_size)},
        # ]

        ##########################################开始训练#########################################
        def train_and_evaluate_model(model2, model_name):
            print(model2)
            print(model_name)
            # model3 = torch.load('.\\model-pth\\MMF-4.pth')
            # print(model3)

            print(f"Processing image_size: {image_size}")
            filename = f"{image_size}_{dataset_stype}_{model_name}_{train_ratio}ModulesACC-1.txt"

            print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))

            model2 = model2.to(device)
            # model3 = model3.to(device)
            cost2 = nn.CrossEntropyLoss().to(device)
            optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=5e-4)

            best_acc_2 = 0.
            best_epoch = 0
            best_kappa_2 = 0.
            epoch_durations = []

            for epoch in range(1, args.epochs + 1):
                model2.train()
                total_epoch_loss = 0

                start = time.time()
                index = 0

                # 超参数，需要你根据任务来调整
                lambda_var = [0.5, 0.5]

                for train_data in train_loader:  # 三模态
                    landsat, all_data, labels = train_data
                    landsat = landsat.to(device)  # 三模态
                    labels = labels.to(device).squeeze()
                    all_data = all_data.to(device)

                    if model_name == 'Cross-Modal Mixer':
                        # 调用模型进行前向传播
                        scenario = 'train'
                        outputs, outputs_mission = model2(landsat, all_data, scenario)
                        loss_complete = cost2(outputs, labels.long())

                        # 根据all_data中的实际情况来确定使用哪个损失函数
                        loss_missing = cost2(outputs_mission, labels.long())

                        total_loss = loss_complete * lambda_var[0] + loss_missing * lambda_var[1]
                        optimizer2.zero_grad()
                        total_loss.backward()
                        optimizer2.step()
                        index += 1
                    elif model_name == 'SingleBranchCNN_7' or model_name == 'SingleBranchResNet50_7' or model_name == 'RestNet_landsat_3':
                        # 调用模型进行前向传播
                        outputs = model2(landsat)
                        # 根据all_data中的实际情况来确定使用哪个损失函数
                        total_loss = cost2(outputs, labels.long())

                        optimizer2.zero_grad()
                        total_loss.backward()
                        optimizer2.step()
                        index += 1
                    elif model_name == 'SingleBranchCNN_10' or model_name == 'SingleBranchResNet50_10' or model_name == 'RestNet_all_factor_10':
                        # 调用模型进行前向传播
                        outputs = model2(all_data)
                        # 根据all_data中的实际情况来确定使用哪个损失函数
                        total_loss = cost2(outputs, labels.long())

                        optimizer2.zero_grad()
                        total_loss.backward()
                        optimizer2.step()
                        index += 1
                    elif model_name == 'MFT':
                        # 调用模型进行前向传播
                        outputs = model2(all_data, landsat)
                        # 根据all_data中的实际情况来确定使用哪个损失函数
                        total_loss = cost2(outputs, labels.long())

                        optimizer2.zero_grad()
                        total_loss.backward()
                        optimizer2.step()
                        index += 1
                    else:
                        # 调用模型进行前向传播
                        outputs = model2(landsat, all_data)
                        # 根据all_data中的实际情况来确定使用哪个损失函数
                        total_loss = cost2(outputs, labels.long())

                        optimizer2.zero_grad()
                        total_loss.backward()
                        optimizer2.step()
                        index += 1

                if epoch % 1 == 0:
                    end = time.time()
                    print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, total_loss.item(), (end - start) * 2))
                    epoch_durations.append(end - start)  # 将这个epoch的运行时间添加到列表中
                    model2.eval()
                    classes = ('1', '2', '3')  # {'住宅区': 0, '公共服务区域': 1, '商业区': 2, '工业区': 3}
                    class_correct2 = list(0. for i in range(args.num_class))
                    class_total2 = list(0. for i in range(args.num_class))
                    correct_prediction_2 = 0.
                    total_2 = 0
                    all_labels = []
                    all_predictions = []
                    with torch.no_grad():
                        for test_data in test_loader:
                            landsat, all_data, labels = test_data
                            landsat = landsat.to(device)
                            # print(images.shape)
                            labels = labels.to(device).squeeze()
                            all_data = all_data.to(device)
                            # all_data[:, 7:, :, :] = 0
                            if model_name == 'Cross-Modal Mixer':
                                scenario = 'val'
                                outputs = model2(landsat, all_data, scenario)
                            elif model_name == 'SingleBranchCNN_7' or model_name == 'SingleBranchResNet50_7' or model_name == 'RestNet_landsat_3':
                                outputs = model2(landsat)
                            elif model_name == 'SingleBranchCNN_10' or model_name == 'SingleBranchResNet50_10' or model_name == 'RestNet_all_factor_10':
                                outputs = model2(all_data)
                            elif model_name == 'MFT':
                                outputs = model2(all_data, landsat)
                            else:
                                outputs = model2(landsat, all_data)

                            _2, predicted2 = torch.max(outputs, 1)
                            c2 = (predicted2 == labels).squeeze()
                            # print(len(labels))
                            for label_idx in range(len(labels)):
                                # print(label_idx)
                                label = labels[label_idx]
                                # print(label)
                                class_correct2[label.int()] += c2[label_idx].item()
                                class_total2[label.int()] += 1
                            total_2 += labels.size(0)
                            # add correct
                            correct_prediction_2 += (predicted2 == labels).sum().item()
                            all_labels.extend(labels.cpu().numpy())
                            all_predictions.extend(predicted2.cpu().numpy())

                    details = []
                    for i in range(len(classes)):
                        if class_total2[i] != 0:
                            accuracy = 100 * class_correct2[i] / class_total2[i]
                        else:
                            accuracy = 0.0  # 或您可以选择一个合适的值来表示这个类没有出现在测试数据集中
                        print('Accuracy of %5s : %2d%% (%2d/%2d)' % (classes[i], accuracy, class_correct2[i], class_total2[i]))
                        details.append(
                            f'Accuracy of {classes[i]} : {accuracy:.4f}% ({class_correct2[i]:.0f}/{class_total2[i]:.0f})')
                    # 计算总OA
                    acc_2 = correct_prediction_2 / total_2
                    # 计算Kappa系数
                    kappa = cohen_kappa_score(all_labels, all_predictions)
                    # 计算Precision Score，注意指定平均类型，例如'macro', 'micro', 'weighted'
                    precision = precision_score(all_labels, all_predictions, average='macro')
                    print("Total Acc Model: %.4f" % (correct_prediction_2 / total_2))
                    print("Total kappa_score Model: %.4f" % (kappa))
                    print("Total precision_score Model: %.4f" % (precision))
                    print("Former Best Acc Model: %.4f" % (best_acc_2))
                    print('----------------------------------------------------')
                    details.append(f"Total Acc Model: {(correct_prediction_2 / total_2):.4f}")
                    details.append(f"Total kappa_score Model: {kappa:.4f}")
                    details.append(f"Total precision_score Model: {precision:.4f}")
                    details.append(f"Former Best Acc Model: {best_acc_2:.4f}")
                    save_metrics(filename, epoch, details, acc_2, kappa, precision, acc_2 > best_acc_2)
                    if best_acc_2 > 0.91:
                        break

                if acc_2 > best_acc_2:
                    print('save new best acc_2', acc_2)
                    # 获取当前日期
                    current_date = datetime.datetime.now().strftime("%m%d")  # 以月和日的形式获取日期
                    # 构建模型保存的文件名
                    # model_filename = f"{image_size}-{current_date}_{model_name}-{dataset_stype}-2.pth"
                    model_filename = f"{image_size}_{model_name}-{dataset_stype}-Modules-1.pth"
                    torch.save(model2, os.path.join(args.model_path, model_filename))
                    best_acc_2 = acc_2
                    best_kappa_2 = kappa
                    best_epoch = epoch

            # 在所有epochs都完成后，计算平均运行时间
            average_epoch_duration = sum(epoch_durations) / len(epoch_durations)
            # 在训练结束时，将平均epoch运行时间写入文件
            print('save new best acc_2', best_acc_2, best_epoch)
            print(f"Completed image_size: {image_size}")
            print(f'Average epoch duration: {average_epoch_duration:.2f} seconds\n')

        ##########################################循环遍历#########################################
        for model_info in models_complate:
            model_name = model_info["name"]
            model = model_info["model"]
            train_and_evaluate_model(model, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=3, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--model_path", default='./model-pth', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--pretrained_model", default='', type=str)
    args = parser.parse_args()
    main(args)