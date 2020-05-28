import argparse
import pickle
from tqdm import tqdm
import sys
import json
sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

training_subjects = [3,4,6,7,9,12,13,15,17,19,25]
action_classes={
'Cook.Cleandishes':0,'Cook.Cleanup':1,'Cook.Cut':2, 'Cook.Stir':3, 'Cook.Usestove':4, 
'Cutbread':5, 'Drink.Frombottle':6, 'Drink.Fromcan':7, 'Drink.Fromcup':8, 'Drink.Fromglass':9,
'Eat.Attable':10, 'Eat.Snack':11, 'Enter':12,'Getup':13, 'Laydown':14, 'Leave':15, 
'Makecoffee.Pourgrains':16, 'Makecoffee.Pourwater':17, 'Maketea.Boilwater':18, 'Maketea.Insertteabag':19, 'Pour.Frombottle':20, 
'Pour.Fromcan':21, 'Pour.Fromkettle':22, 'Readbook':23, 'Sitdown':24, 'Takepills':25, 
'Uselaptop':26, 'Usetelephone':27, 'Usetablet':28, 'Walk':29, 'WatchTV':30
}

action_cv={
'Cutbread':0, 'Drink.Frombottle':1, 'Drink.Fromcan':2, 'Drink.Fromcup':3, 'Drink.Fromglass':4, 'Eat.Attable':5, 
'Eat.Snack':6, 'Enter':7, 'Getup':8, 'Leave':9, 'Pour.Frombottle':10, 'Pour.Fromcan':11, 
'Readbook':12, 'Sitdown':13, 'Takepills':14, 'Uselaptop':15, 'Usetablet':16,
'Usetelephone':17, 'Walk':18
}
 
training_cameras1 = [1]
training_cameras2 = [1,3,4,6,7]
val_cameras = [5]
testing_cameras=[2]

max_body_true = 1
max_body=2
num_joint = 15
max_frame = 4000

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as json_data:
        skeleton_sequence = json.load(json_data)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=2, num_joint=15):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, len(seq_info['frames']), num_joint, 3))
    for n, f in enumerate(seq_info['frames']):
        if len(f)!=0:
            for m, b in enumerate(f):
                for j in range(num_joint):
                    if m < max_body:
                        if j < 13:
                            k=13-j-1
                            data[m, n, j, :] = [b['pose3d'][k], b['pose3d'][k+13], b['pose3d'][k+13*2]]
                        elif j==13:
                            data[m, n, j, :] = [(b['pose3d'][10]+b['pose3d'][11])/2, (b['pose3d'][23]+b['pose3d'][24])/2, (b['pose3d'][36]+b['pose3d'][37])/2]
                        elif j==14:
                            data[m, n, j, :] = [(b['pose3d'][4]+b['pose3d'][5])/2, (b['pose3d'][17]+b['pose3d'][18])/2, (b['pose3d'][30]+b['pose3d'][31])/2]                  
                    else:
                        pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.json' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if (filename in ignored_samples) or ((filename.split('_')[0] not in action_cv.keys()) and (benchmark == 'xview1' or benchmark == 'xview2')) :
            continue
        if benchmark == 'xview1' or benchmark == 'xview2':
            action_class = int(
                action_cv[filename.split('_')[0]])
        else:
            action_class = int(
                action_classes[filename.split('_')[0]])
        subject_id = int(
            filename.split('_')[1][1:])
        camera_id = int(
            filename.split('_')[4][1:3])

        if benchmark == 'xview1':
            istraining = (camera_id in training_cameras1)
            istesting=(camera_id in testing_cameras)
            isval=(camera_id in val_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xview2':
            istraining = (subject_id in training_cameras2)
            istesting=(camera_id in testing_cameras)
            isval=(camera_id in val_cameras)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            if benchmark == 'xview1' or  benchmark == 'xview2':
                issample = isval
            else:
                issample = not (istraining)
        elif part == 'test':
            if benchmark == 'xview1' or  benchmark == 'xview2':
                issample = istesting
            else:
                issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data

    #fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smarthome Data Converter.')
    parser.add_argument('--data_path', default='../data/smarthome_raw/smarthome_skeletons/')
    parser.add_argument('--ignored_sample_path',
                        default=None)
    parser.add_argument('--out_folder', default='../data/smarthome/')
    benchmark = ['xsub']
    part = ['train', 'val']# 'test']
    arg = parser.parse_args()
    print('raw_path: ', arg.data_path)
    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)

            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
