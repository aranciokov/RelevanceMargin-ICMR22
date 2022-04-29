import torch as th
import numpy as np

def to_tensor(arr, gpu=False):
    tensor = th.tensor(arr).float()
    if gpu:
        return tensor.cuda()
    return tensor


def sample_triplets(anchor_idxs, x_to_class_dict, class_to_y_dict, num_triplets, sampling_method='random'):
    assert sampling_method in {'random'}
    if sampling_method == 'random':
        return sample_random_triplets(anchor_idxs, x_to_class_dict, class_to_y_dict, num_triplets)

def sample_random_triplets(anchor_idxs, x_to_class_dict, class_to_y_dict, num_triplets):
    classes = set(class_to_y_dict.keys())
    from tqdm import tqdm
    pos_idxs = []
    neg_idxs = []
    ff = 0
    for anchor in tqdm(anchor_idxs):
        pos_class = x_to_class_dict[anchor]
        pos_class_ = pos_class if isinstance(pos_class, list) else [pos_class]
        #print(pos_class_, len(pos_class_))
        if len(pos_class_) == 0 or \
                (len(pos_class_) == 1 and
                 (pos_class_[0] == -1 or (isinstance(pos_class_[0], str) and "-1" in pos_class_[0]))):
            # no class for this caption type was available when computing the relational dicts (can happen in YC2)
            ff += 1
            #print(pos_class_)
            if len(pos_class_) == 0 or pos_class_[0] == -1:
                compatible_classes = [k for k in x_to_class_dict.values() if (isinstance(k, int) and k != -1)
                                      or (isinstance(k, list) and len(k) > 0)]
            elif "-1" in pos_class_[0]:
                _a, _b = pos_class_[0].split("_")
                fn = (lambda x: x.startswith(_a)) if _b == "-1" else (lambda x: x.endswith(_b))
                compatible_classes = [k for k in x_to_class_dict.values() if "-1" not in k and fn(k)]
            else:
                print(pos_class_)
                assert False, "should never enter here"

            if len(compatible_classes) == 0:
                #print("no compatible classes found")
                compatible_classes = list(x_to_class_dict.values())
                #input()
            #print(compatible_classes, len(compatible_classes), np.random.randint(0, len(compatible_classes), 1))
            pos_class_ = compatible_classes[np.random.randint(0, len(compatible_classes), 1)[0]]
            pos_class_ = pos_class_ if isinstance(pos_class_, list) else [pos_class_]
            #pos_class_ = [rnd_cls_ix]
            #print(ff, pos_class_)  # happens around 45 times in YC2 train
        pos_idxs.append(sample_n(class_to_y_dict, pos_class_, num_triplets))
        neg_classes = list(classes - set(pos_class_))
        neg_idxs.append(sample_n(class_to_y_dict, neg_classes, num_triplets))
    pos_idxs = np.array(pos_idxs)
    neg_idxs = np.array(neg_idxs)
    return pos_idxs, neg_idxs

def sample_n(class_to_y_dict, classes, num_triplets):
    #print(len(classes))
    if len(classes) == 1:
        sample_classes = classes * num_triplets
    else:
        classes = np.array(classes)
        sample_classes = classes[np.random.randint(len(classes), size=num_triplets)]
    sampled_triplets = np.zeros(num_triplets, dtype=np.int)
    for i, class_ in enumerate(sample_classes):
        class_idxs = class_to_y_dict[class_]
        sampled_triplets[i] = class_idxs[np.random.randint(len(class_idxs))]
    return sampled_triplets

def convert_rel_dicts_to_uids(x_to_class, class_to_x, uid_to_idx):
    new_x_to_class = {}
    for uid in x_to_class:
        new_x = uid_to_idx[uid]
        new_x_to_class[new_x] = x_to_class[uid]
    new_class_to_x = {}
    for class_ in class_to_x:
        new_class_to_x[class_] = []
        for uid in class_to_x[class_]:
            new_class_to_x[class_].append(uid_to_idx[uid])
    return new_x_to_class, new_class_to_x
