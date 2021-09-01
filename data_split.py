import os
import csv
import shutil

base_dir = os.getcwd()

gt = open('/home/pmi-minos/Documents/MinosNet/datasets/ISIC/ISIC_2019_Training_GroundTruth.csv')
ori = os.path.join(base_dir, "datasets/ISIC/ISIC_2019_Training_Input")
des = os.path.join(base_dir, "datasets/ISIC2019")


gt_csv = csv.reader(gt)
targetnames = ['mel', 'nv', 'bcc', 'ak', 'bkl', 'df', 'vasc', 'scc', 'unk']

for line in gt_csv:
    img_name = line[0]
    file_path = os.path.join(ori, img_name + ".jpg")

    for i in range(9):
        img_ann = line[i+1]
        if img_ann == "1.0":
            des_path = os.path.join(des, targetnames[i])
            shutil.copyfile(file_path, des_path + "/" + img_name + ".jpg")
            print(file_path)
            print(des_path)

            break

gt.close()
