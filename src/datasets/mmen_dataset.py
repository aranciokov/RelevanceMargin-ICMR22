import os
import sys
import numpy as np
import pandas as pd

from datasets import to_tensor, sample_triplets, convert_rel_dicts_to_uids
import defaults.EPIC_MMEN as EPIC_MMEN


def create_epic_mmen_dataset(caption_type, is_train=True,
        batch_size=EPIC_MMEN.batch_size, num_triplets=EPIC_MMEN.num_triplets,
        action_dataset=False, is_test=False, all_nouns=False, rgb_only=False, rgb_flow_only=False):
    """
    Creates a mmen dataset object for EPIC2020 using default locations of feature files and relational dicts files.
    """
    is_train_str = 'train' if is_train else 'validation'
    if is_test:
        is_train_str = 'test'
    #Find location of word features based on caption type and load
    if caption_type in ['caption', 'verb', 'noun']:
        word_features_path = 'EPIC_100_retrieval_{}_text_features_{}.pkl'.format(caption_type, is_train_str)
    else:
        raise NotImplementedError(caption_type)
    word_features_path = './data/text_features/{}'.format(word_features_path)
    text_features = pd.read_pickle(word_features_path)

    #load video features
    video_features = pd.read_pickle('./data/video_features/EPIC_100_retrieval_{}_features_mean.pkl'.format(is_train_str))
    video_features = np.array(video_features['features'])
    if rgb_only or rgb_flow_only:
        print(f"mmen_dataset.py: requested {'RGB' if rgb_only else ('RGB+Flow' if rgb_flow_only else '??')} only), "
              f"shape before:", video_features.shape)
        if rgb_only:
            video_features = video_features[:, :1024]
        elif rgb_flow_only:
            video_features = video_features[:, :2048]
        else:
            assert False
        print(f"mmen_dataset.py: requested {'RGB' if rgb_only else ('RGB+Flow' if rgb_flow_only else '??')} only), "
              f"shape after:", video_features.shape)
    #video_features = video_features[:, :2048]
    #print("src/datasets/mmen_dataset.py: caption_type {} video_features {}".format(caption_type, video_features.shape))

    #Load relational dictionaries
    if is_train:
        rel_dicts = pd.read_pickle('./data/relational/EPIC_100_retrieval_{}_relational_dict_{}.pkl'.format(caption_type, is_train_str))
        rel_dicts = [rel_dicts['vid2class'], rel_dicts['class2vid'],
                rel_dicts['sent2class'], rel_dicts['class2sent']]
        if all_nouns:
            rel_dicts = pd.read_pickle('./data/relational/{}_allnouns_relational_EPIC_100_retrieval_{}.pkl'.format(caption_type, is_train_str))
            if action_dataset:
                rel_dicts = [rel_dicts['vid2class'], rel_dicts['class2vid'],
                         rel_dicts['sent2class'], rel_dicts['class2sent'],
                         rel_dicts["vid2classes"], rel_dicts["sent2classes"]]
            else:
                rel_dicts = [rel_dicts['vid2class'], rel_dicts['class2vid'],
                        rel_dicts['sent2class'], rel_dicts['class2sent']]

    else:
        rel_dicts = None

    #Load relevancy matrix
    rel_matrix = None
    if True:  #not is_train and not is_test:
        rel_matrix = pd.read_pickle('./data/relevancy/{}_relevancy_EPIC_100_retrieval_{}.pkl'.format(caption_type,is_train_str))
        #print("reading", './data/relevancy/{}_relevancy_EPIC_100_retrieval_{}.pkl'.format(caption_type,is_train_str))
    if action_dataset:
        return MMEN_Dataset_Action(video_features, text_features, rel_dicts,
                batch_size=batch_size, num_triplets=num_triplets,
                relevancy_matrix=rel_matrix)
    else:
        return MMEN_Dataset(video_features, text_features, rel_dicts,
                batch_size=batch_size, num_triplets=num_triplets,
                relevancy_matrix=rel_matrix)


def create_youcook_mmen_dataset(caption_type, is_train=True,
        batch_size=EPIC_MMEN.batch_size, num_triplets=EPIC_MMEN.num_triplets,
        action_dataset=False, is_test=False, all_nouns=False, rgb_only=False, rgb_flow_only=False):
    """
    Creates a mmen dataset object for YouCook2 using default locations of feature files and relational dicts files.
    """
    is_train_str = 'train' if is_train else 'val'
    if is_test:
        is_train_str = 'val'
    #Find location of word features based on caption type and load
    if caption_type in ['caption', 'verb', 'noun']:
        word_features_path = '{}_features_parsed_youcookii_synsets_{}.pkl'.format(caption_type, is_train_str)
    else:
        raise NotImplementedError(caption_type)
    word_features_path = './data/text_features/{}'.format(word_features_path)
    text_features = pd.read_pickle(word_features_path)

    #load video features
    video_names_path = "./public_split_yc2/{}_names.npy".format('trn' if is_train else 'val')
    video_names = np.load(video_names_path)
    print("reading", len(video_names), "names from", video_names_path)
    if not rgb_flow_only and not rgb_only:
        #input("Check that RGB+Flow+Audio features are ready. Setting rgb_only=True")
        rgb_only = True

    if rgb_only or rgb_flow_only:
        coll_fn = lambda x: np.mean(x, 0, keepdims=True)  # ResNet-152 features have shape (N, 2048)
        app_fts = lambda n: coll_fn(np.load("./data/video_features/YC2_Resnet_feats/{}.npz".format(n))["features"])
        video_features = np.concatenate([app_fts(n) for n in video_names], 0)
        print(f"mmen_dataset.py: requested {'RGB' if rgb_only else ('RGB+Flow' if rgb_flow_only else '??')} only), "
              f"(ResNet152 feats) shape before:", video_features.shape)
    if rgb_flow_only:
        assert False, "RGB+Flow not implemented"

    #Load relational dictionaries
    if is_train:
        rel_dicts = pd.read_pickle('./data/relational/{}_allnouns_relational_parsed_youcookii_synsets_{}.pkl'.format(caption_type, is_train_str))
        """if caption_type == "caption":
            vid2classes = {k: [v] for k, v in rel_dicts["vid2classes"].items()}
            sent2classes = {k: [v] for k, v in rel_dicts["sent2classes"].items()}
        else:"""
        vid2classes = rel_dicts["vid2classes"]
        sent2classes = rel_dicts["sent2classes"]
        if action_dataset:
            rel_dicts = [rel_dicts['vid2class'], rel_dicts['class2vid'],
                         rel_dicts['sent2class'], rel_dicts['class2sent'],
                         vid2classes, sent2classes]
        else:
            if caption_type in ["verb", "noun"]:
                rel_dicts = [rel_dicts['vid2classes'], rel_dicts['class2vid'],
                             rel_dicts['sent2classes'], rel_dicts['class2sent']]
            else:
                rel_dicts = [rel_dicts['vid2class'], rel_dicts['class2vid'],
                             rel_dicts['sent2class'], rel_dicts['class2sent']]

    else:
        rel_dicts = None

    #Load relevancy matrix
    rel_matrix = None
    if True:  #not is_train and not is_test:
        rel_matrix = pd.read_pickle('./data/relevancy/{}_prova_relevancy_parsed_youcookii_synsets_{}.pkl'.format(caption_type,is_train_str))
        #print("reading", './data/relevancy/{}_relevancy_EPIC_100_retrieval_{}.pkl'.format(caption_type,is_train_str))
    if action_dataset:
        return MMEN_Dataset_Action(video_features, text_features, rel_dicts,
                batch_size=batch_size, num_triplets=num_triplets,
                relevancy_matrix=rel_matrix)
    else:
        return MMEN_Dataset(video_features, text_features, rel_dicts,
                batch_size=batch_size, num_triplets=num_triplets,
                relevancy_matrix=rel_matrix)


class MMEN_Dataset:
    """
    Dataset wrapper for a multi-modal embedding Network Dataset.
    """
    def __init__(self, x, y, relation_dicts, batch_size=64, num_triplets=10, x_name='v', y_name='t', relevancy_matrix=None):
        if relation_dicts is None:
            self.test_dataset = True
        else:
            self.test_dataset = False
            if len(relation_dicts) == 4:
                x_to_class, class_to_x, y_to_class, class_to_y = relation_dicts
                self.all_nouns = False
            else:
                x_to_class, class_to_x, y_to_class, class_to_y, x_to_classes, y_to_classes = relation_dicts
                self.all_nouns = True
        self._x = x
        self._y = y

        self.x_name = x_name
        self.y_name = y_name
        self.x_to_x_name = '{}{}'.format(x_name, x_name)
        self.x_to_y_name = '{}{}'.format(x_name, y_name)
        self.y_to_x_name = '{}{}'.format(y_name, x_name)
        self.y_to_y_name = '{}{}'.format(y_name, y_name)


        self.x_size = x.shape[-1]
        self.x_len = x.shape[0]
        self.y_size = y.shape[-1]
        self.y_len = y.shape[0]


        if not self.test_dataset:
            self.triplets = {self.x_to_x_name: [], self.x_to_y_name: [],
                    self.y_to_x_name: [], self.y_to_y_name: []}

            x_uids = list(x_to_class.keys())
            self._x_idxs = np.array(list(range(len(x_uids))))
            self.x_uid_to_idx = {uid: idx for idx, uid in enumerate(x_uids)}
            self.x_idx_to_uid = {idx: uid for idx, uid in enumerate(x_uids)}
            self._y_idxs = np.array(list(y_to_class.keys()))

            if len(relation_dicts) == 6:
                for k, v in y_to_classes.items():
                    pr = False
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], list) and isinstance(v[1], list):
                        #print(v)
                        if len(v[0]) == 0:
                            y_to_classes[k][0] = list(y_to_classes.values())[np.random.randint(0, len(y_to_classes), 1)[0]][0]
                            pr = True
                        if len(v[1]) == 0:
                            y_to_classes[k][1] = list(y_to_classes.values())[np.random.randint(0, len(y_to_classes), 1)[0]][1]
                            pr = True
                        #print("->", y_to_classes[k])
                        #if pr: input()
                for k, v in x_to_classes.items():
                    pr = False
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], list) and isinstance(v[1], list):
                        #print(v)
                        if len(v[0]) == 0:
                            x_to_classes[k][0] = list(x_to_classes.values())[np.random.randint(0, len(x_to_classes), 1)[0]][0]
                            pr = True
                        if len(v[1]) == 0:
                            x_to_classes[k][1] = list(x_to_classes.values())[np.random.randint(0, len(x_to_classes), 1)[0]][1]
                            pr = True
                        #print("->", x_to_classes[k])
                        #if pr: input()
                self.y_to_classes = y_to_classes
                x_to_classes, _ = convert_rel_dicts_to_uids(x_to_classes, class_to_x, self.x_uid_to_idx)
                self.x_to_classes = x_to_classes

                #print(list(x_to_class.keys())[0], list(x_to_classes.keys())[0], list(y_to_class.keys())[0], list(y_to_classes.keys())[0])

            x_to_class, class_to_x = convert_rel_dicts_to_uids(x_to_class, class_to_x, self.x_uid_to_idx)

            self.x_to_class = x_to_class
            self.class_to_x = class_to_x
            self.y_to_class = y_to_class
            self.class_to_y = class_to_y

        self.batch_size = batch_size
        self.num_triplets = num_triplets

        self.relevancy_matrix = relevancy_matrix

    def sample_triplets(self, cross_modal_pair, sampling_method='random'):
        if cross_modal_pair[0] == self.x_name:
            b1 = self._x
            to_class = self.x_to_class
            b_idxs = self._x_idxs
        elif cross_modal_pair[0] == self.y_name:
            b1 = self._y
            to_class = self.y_to_class
            b_idxs = self._y_idxs
        else:
            raise Exception('Unknown cross_modal_pair: {}'.format(cross_modal_pair))

        if cross_modal_pair[1] == self.x_name:
            b2 = self._x
            from_class = self.class_to_x
        elif cross_modal_pair[1] == self.y_name:
            b2 = self._y
            from_class = self.class_to_y
        else:
            raise Exception('Unknown cross_modal_pair: {}'.format(cross_modal_pair))

        self.triplets[cross_modal_pair] = []

        anchors = b_idxs
        pos_idxs, neg_idxs = sample_triplets(anchors, to_class,
                from_class, self.num_triplets, sampling_method)
        positives = pos_idxs
        negatives = neg_idxs
        self.triplets[cross_modal_pair] = (anchors, positives, negatives)
        self.sampling_method = sampling_method

    def _get_triplet_batch_start_end(self, i, modality_length):
        start = i * self.batch_size
        end = (i + 1) * self.batch_size

        if start > modality_length:
            start = start % modality_length
            end = end % modality_length
            if start > end:
                end = modality_length
        if end > modality_length:
            end = modality_length
        return start, end

    def get_triplet_batch(self, cross_modal_pairs, gpu=False):
        assert isinstance(cross_modal_pairs, list)
        modalities = set([pair[0] for pair in cross_modal_pairs])
        if self.x_to_y_name[0] in modalities and self.y_to_x_name[0] in modalities:
            longest_modality = self.x_len if self.x_len > self.y_len else self.y_len
        elif self.x_to_y_name[0] in modalities:
            longest_modality = self.x_len
        elif self.y_to_x_name[0] in modalities:
            longest_modality = self.y_len
        else:
            raise Exception('No modalities found in cross_modal_pairs: {}.'.format(cross_modal_pair))

        num_batches = int(np.ceil(longest_modality / self.batch_size))

        x_idxs = np.array(range(self.x_len))
        y_idxs = np.array(range(self.y_len))
        np.random.shuffle(x_idxs)
        np.random.shuffle(y_idxs)
        for i in range(num_batches):
            x_start, x_end = self._get_triplet_batch_start_end(i, self.x_len)
            y_start, y_end = self._get_triplet_batch_start_end(i, self.y_len)
            batch_dict = {}
            for cross_modal_pair in cross_modal_pairs:
                if cross_modal_pair[0] == self.x_to_y_name[0]:
                    anch_vals = self._x
                    start = x_start
                    end = x_end
                    idxs = x_idxs
                else:
                    anch_vals = self._y
                    start = y_start
                    end = y_end
                    idxs = y_idxs
                if cross_modal_pair[1] == self.x_to_y_name[1]:
                    other_vals = self._y
                else:
                    other_vals = self._x
                triplets = self.triplets[cross_modal_pair]
                batch_anchors_idxs = np.repeat(triplets[0][idxs[start:end]], self.num_triplets)
                batch_pos_idxs = triplets[1][idxs[start:end]].flatten()
                if self.sampling_method in ['random']:
                    batch_neg_idxs = triplets[2][idxs[start:end]].flatten()
                else:
                    neg = np.repeat(triplets[2][idxs[start:end]], self.num_triplets)
                anch = to_tensor(anch_vals[batch_anchors_idxs, :], gpu=gpu)
                pos = to_tensor(other_vals[batch_pos_idxs, :], gpu=gpu)
                if self.sampling_method in ['random']:
                    neg = to_tensor(other_vals[batch_neg_idxs, :], gpu=gpu)
                #print(cross_modal_pair, anch_vals.shape, other_vals.shape, ";", anch.shape, pos.shape, neg.shape)
                #print(self.relevancy_matrix.shape)
                batch_dict[cross_modal_pair] = (anch, pos, neg)
                return_classes = False
                if cross_modal_pair[0] == self.x_to_y_name[0] and cross_modal_pair[1] == self.x_to_y_name[1]:
                    if not self.all_nouns:
                        anc_classes = [self.x_to_class[k] for k in batch_anchors_idxs]
                        pos_classes = [self.y_to_class[k] for k in batch_pos_idxs]
                        neg_classes = [self.y_to_class[k] for k in batch_neg_idxs]
                    else:
                        anc_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_anchors_idxs]
                        pos_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_pos_idxs]
                        neg_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_neg_idxs]
                    return_classes = True
                elif cross_modal_pair[0] == self.x_to_y_name[1] and cross_modal_pair[1] == self.x_to_y_name[0]:
                    #print(cross_modal_pair, batch_anchors_idxs[0], list(self.x_to_classes.keys())[0], list(self.x_to_class.keys())[0], list(self.y_to_classes.keys())[0], list(self.y_to_class.keys())[0])
                    if not self.all_nouns:
                        anc_classes = [self.y_to_class[k] for k in batch_anchors_idxs]
                        pos_classes = [self.x_to_class[k] for k in batch_pos_idxs]
                        neg_classes = [self.x_to_class[k] for k in batch_neg_idxs]
                    else:
                        anc_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_anchors_idxs]
                        pos_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_pos_idxs]
                        neg_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_neg_idxs]
                    return_classes = True
                elif cross_modal_pair[0] == self.x_to_y_name[0] and cross_modal_pair[1] == self.x_to_y_name[0]:
                    #print(cross_modal_pair, batch_anchors_idxs[0], list(self.x_to_classes.keys())[0], list(self.x_to_class.keys())[0], list(self.y_to_classes.keys())[0], list(self.y_to_class.keys())[0])
                    if not self.all_nouns:
                        anc_classes = [self.x_to_class[k] for k in batch_anchors_idxs]
                        pos_classes = [self.x_to_class[k] for k in batch_pos_idxs]
                        neg_classes = [self.x_to_class[k] for k in batch_neg_idxs]
                    else:
                        anc_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_anchors_idxs]
                        pos_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_pos_idxs]
                        neg_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_neg_idxs]
                    return_classes = True
                elif cross_modal_pair[0] == self.x_to_y_name[1] and cross_modal_pair[1] == self.x_to_y_name[1]:
                    #print(cross_modal_pair, batch_anchors_idxs[0], list(self.x_to_classes.keys())[0], list(self.x_to_class.keys())[0], list(self.y_to_classes.keys())[0], list(self.y_to_class.keys())[0])
                    if not self.all_nouns:
                        anc_classes = [self.y_to_class[k] for k in batch_anchors_idxs]
                        pos_classes = [self.y_to_class[k] for k in batch_pos_idxs]
                        neg_classes = [self.y_to_class[k] for k in batch_neg_idxs]
                    else:
                        """for k in batch_anchors_idxs:
                            print(self.y_to_class[k], self.y_to_classes[k])
                            input()"""
                        anc_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_anchors_idxs]
                        pos_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_pos_idxs]
                        neg_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_neg_idxs]
                    return_classes = True
                batch_dict[cross_modal_pair] = (anch, pos, neg)
                # batch_*_ixs -> (640, ) indices [0-67218]
                if return_classes:  #cross_modal_pair[0] != cross_modal_pair[1]:
                    batch_dict[f"{cross_modal_pair[0]}_{cross_modal_pair[1]}_cls"] = (anc_classes, pos_classes, neg_classes)
            yield batch_dict

    def get_eval_batch(self, gpu=False):
        return to_tensor(self._x, gpu=gpu), to_tensor(self._y, gpu=gpu)


class MMEN_Dataset_Action(MMEN_Dataset):
    def __init__(self, x, y, relation_dicts, batch_size=64, num_triplets=10, x_name='v', y_name='t', relevancy_matrix=None):
        super(MMEN_Dataset_Action, self).__init__(x, y, relation_dicts,
                batch_size=batch_size, num_triplets=num_triplets,
                x_name=x_name, y_name=y_name, relevancy_matrix=relevancy_matrix)

    def get_triplet_batch(self, cross_modal_pairs, PoS_datasets, gpu=False):
        assert isinstance(cross_modal_pairs, list)
        modalities = set([pair[0] for pair in cross_modal_pairs])
        if self.x_to_y_name[0] in modalities and self.y_to_x_name[0] in modalities:
            longest_modality = self.x_len if self.x_len > self.y_len else self.y_len
        elif self.x_to_y_name[0] in modalities:
            longest_modality = self.x_len
        elif self.y_to_x_name[0] in modalities:
            longest_modality = self.y_len
        else:
            raise Exception('No modalities found in cross_modal_pairs: {}.'.format(cross_modal_pair))

        num_batches = int(np.ceil(longest_modality / self.batch_size))

        x_idxs = np.array(range(self.x_len))
        y_idxs = np.array(range(self.y_len))
        np.random.shuffle(x_idxs)
        np.random.shuffle(y_idxs)
        for i in range(num_batches):
            x_start, x_end = self._get_triplet_batch_start_end(i, self.x_len)
            y_start, y_end = self._get_triplet_batch_start_end(i, self.y_len)
            batch_dict = {}
            for cross_modal_pair in cross_modal_pairs:
                if cross_modal_pair[0] == self.x_to_y_name[0]:
                    anch_vals = {PoS: PoS_datasets[PoS]._x for PoS in PoS_datasets}
                    start = x_start
                    end = x_end
                    idxs = x_idxs
                else:
                    anch_vals = {PoS: PoS_datasets[PoS]._y for PoS in PoS_datasets}
                    start = y_start
                    end = y_end
                    idxs = y_idxs
                if cross_modal_pair[1] == self.y_name:
                    other_vals = {PoS: PoS_datasets[PoS]._y for PoS in PoS_datasets}
                else:
                    other_vals = {PoS: PoS_datasets[PoS]._x for PoS in PoS_datasets}
                triplets = self.triplets[cross_modal_pair]
                batch_anchors_idxs = np.repeat(triplets[0][idxs[start:end]], self.num_triplets)
                batch_pos_idxs = triplets[1][idxs[start:end]].flatten()
                if self.sampling_method in ['random']:
                    batch_neg_idxs = triplets[2][idxs[start:end]].flatten()
                else:
                    neg = np.repeat(triplets[2][idxs[start:end]], self.num_triplets)
                anch = {PoS: to_tensor(anch_vals[PoS][batch_anchors_idxs, :], gpu=gpu) for PoS in PoS_datasets}
                pos = {PoS: to_tensor(other_vals[PoS][batch_pos_idxs, :], gpu=gpu) for PoS in PoS_datasets}
                if self.sampling_method in ['random']:
                    neg = {PoS: to_tensor(other_vals[PoS][batch_neg_idxs, :], gpu=gpu) for PoS in PoS_datasets}
                batch_dict[cross_modal_pair] = (anch, pos, neg)
                #print(cross_modal_pair, anch_vals.shape, other_vals.shape, ";", anch.shape, pos.shape, neg.shape)
                #print(self.relevancy_matrix.shape)
                check_non_empty = lambda vs, x: x if len(x) > 0 else np.random.randint(0, max([v for v in vs if v != -1]), 1)
                return_classes = False
                if cross_modal_pair[0] == self.x_to_y_name[0] and cross_modal_pair[1] == self.x_to_y_name[1]:
                    if not self.all_nouns:
                        anc_classes = [self.x_to_class[k] for k in batch_anchors_idxs]
                        pos_classes = [self.y_to_class[k] for k in batch_pos_idxs]
                        neg_classes = [self.y_to_class[k] for k in batch_neg_idxs]
                    else:
                        _tmp = self.x_to_classes[batch_anchors_idxs[0]]
                        if isinstance(_tmp, list) and len(_tmp) == 2 and isinstance(_tmp[0], list) and isinstance(_tmp[1], list):
                            #YC
                            anc_classes = [f"{','.join(list(map(str, self.x_to_classes[k][0])))}_{','.join(list(map(str, self.x_to_classes[k][1])))}" for k in batch_anchors_idxs]
                            pos_classes = [f"{','.join(list(map(str, self.y_to_classes[k][0])))}_{','.join(list(map(str, self.y_to_classes[k][1])))}" for k in batch_pos_idxs]
                            neg_classes = [f"{','.join(list(map(str, self.y_to_classes[k][0])))}_{','.join(list(map(str, self.y_to_classes[k][1])))}" for k in batch_neg_idxs]
                        else:
                            anc_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_anchors_idxs]
                            pos_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_pos_idxs]
                            neg_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_neg_idxs]
                    return_classes = True
                elif cross_modal_pair[0] == self.x_to_y_name[1] and cross_modal_pair[1] == self.x_to_y_name[0]:
                    #print(cross_modal_pair, batch_anchors_idxs[0], list(self.x_to_classes.keys())[0], list(self.x_to_class.keys())[0], list(self.y_to_classes.keys())[0], list(self.y_to_class.keys())[0])
                    if not self.all_nouns:
                        anc_classes = [self.y_to_class[k] for k in batch_anchors_idxs]
                        pos_classes = [self.x_to_class[k] for k in batch_pos_idxs]
                        neg_classes = [self.x_to_class[k] for k in batch_neg_idxs]
                    else:
                        _tmp = self.y_to_classes[batch_anchors_idxs[0]]
                        if isinstance(_tmp, list) and len(_tmp) == 2 and isinstance(_tmp[0], list) and isinstance(
                                _tmp[1], list):
                            # YC
                            anc_classes = [
                                f"{','.join(list(map(str, self.y_to_classes[k][0])))}_{','.join(list(map(str, self.y_to_classes[k][1])))}"
                                for k in batch_anchors_idxs]
                            pos_classes = [
                                f"{','.join(list(map(str, self.x_to_classes[k][0])))}_{','.join(list(map(str, self.x_to_classes[k][1])))}"
                                for k in batch_pos_idxs]
                            neg_classes = [
                                f"{','.join(list(map(str, self.x_to_classes[k][0])))}_{','.join(list(map(str, self.x_to_classes[k][1])))}"
                                for k in batch_neg_idxs]
                        else:
                            anc_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_anchors_idxs]
                            pos_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_pos_idxs]
                            neg_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_neg_idxs]
                    return_classes = True
                elif cross_modal_pair[0] == self.x_to_y_name[0] and cross_modal_pair[1] == self.x_to_y_name[0]:
                    #print(cross_modal_pair, batch_anchors_idxs[0], list(self.x_to_classes.keys())[0], list(self.x_to_class.keys())[0], list(self.y_to_classes.keys())[0], list(self.y_to_class.keys())[0])
                    if not self.all_nouns:
                        anc_classes = [self.x_to_class[k] for k in batch_anchors_idxs]
                        pos_classes = [self.x_to_class[k] for k in batch_pos_idxs]
                        neg_classes = [self.x_to_class[k] for k in batch_neg_idxs]
                    else:
                        _tmp = self.x_to_classes[batch_anchors_idxs[0]]
                        if isinstance(_tmp, list) and len(_tmp) == 2 and isinstance(_tmp[0], list) and isinstance(
                                _tmp[1], list):
                            # YC
                            anc_classes = [
                                f"{','.join(list(map(str, self.x_to_classes[k][0])))}_{','.join(list(map(str, self.x_to_classes[k][1])))}"
                                for k in batch_anchors_idxs]
                            pos_classes = [
                                f"{','.join(list(map(str, self.x_to_classes[k][0])))}_{','.join(list(map(str, self.x_to_classes[k][1])))}"
                                for k in batch_pos_idxs]
                            neg_classes = [
                                f"{','.join(list(map(str, self.x_to_classes[k][0])))}_{','.join(list(map(str, self.x_to_classes[k][1])))}"
                                for k in batch_neg_idxs]
                        else:
                            anc_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_anchors_idxs]
                            pos_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_pos_idxs]
                            neg_classes = [f"{self.x_to_class[k].split('_')[0]}_{','.join(list(map(str, self.x_to_classes[k])))}" for k in batch_neg_idxs]
                    return_classes = True
                elif cross_modal_pair[0] == self.x_to_y_name[1] and cross_modal_pair[1] == self.x_to_y_name[1]:
                    #print(cross_modal_pair, batch_anchors_idxs[0], list(self.x_to_classes.keys())[0], list(self.x_to_class.keys())[0], list(self.y_to_classes.keys())[0], list(self.y_to_class.keys())[0])
                    if not self.all_nouns:
                        anc_classes = [self.y_to_class[k] for k in batch_anchors_idxs]
                        pos_classes = [self.y_to_class[k] for k in batch_pos_idxs]
                        neg_classes = [self.y_to_class[k] for k in batch_neg_idxs]
                    else:
                        _tmp = self.y_to_classes[batch_anchors_idxs[0]]
                        if isinstance(_tmp, list) and len(_tmp) == 2 and isinstance(_tmp[0], list) and isinstance(
                                _tmp[1], list):
                            # YC
                            anc_classes = [
                                f"{','.join(list(map(str, self.y_to_classes[k][0])))}_{','.join(list(map(str, self.y_to_classes[k][1])))}"
                                for k in batch_anchors_idxs]
                            pos_classes = [
                                f"{','.join(list(map(str, self.y_to_classes[k][0])))}_{','.join(list(map(str, self.y_to_classes[k][1])))}"
                                for k in batch_pos_idxs]
                            neg_classes = [
                                f"{','.join(list(map(str, self.y_to_classes[k][0])))}_{','.join(list(map(str, self.y_to_classes[k][1])))}"
                                for k in batch_neg_idxs]
                        else:
                            anc_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_anchors_idxs]
                            pos_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_pos_idxs]
                            neg_classes = [f"{self.y_to_class[k].split('_')[0]}_{','.join(list(map(str, self.y_to_classes[k])))}" for k in batch_neg_idxs]
                    return_classes = True
                batch_dict[cross_modal_pair] = (anch, pos, neg)
                # batch_*_ixs -> (640, ) indices [0-67218]
                if return_classes:  #cross_modal_pair[0] != cross_modal_pair[1]:
                    batch_dict[f"{cross_modal_pair[0]}_{cross_modal_pair[1]}_cls"] = (anc_classes, pos_classes, neg_classes)
            yield batch_dict


if __name__ == '__main__':
    for caption_type in ['caption', 'verb', 'noun']:
        print('LOADING: {}'.format(caption_type))
        mm_ds = create_epic_mmen_dataset(caption_type)
        mm_ds.sample_triplets('vt')
        mm_ds.sample_triplets('tv')
        for batch in mm_ds.get_triplet_batch(['vt', 'tv']):
            vt_batch = batch['vt']
            tv_batch = batch['tv']
            anch, pos, neg = vt_batch
            assert pos.shape == neg.shape
            assert anch.shape[0] == pos.shape[0]
            anch, pos, neg = tv_batch
            assert pos.shape == neg.shape
            assert anch.shape[0] == pos.shape[0]
        eval_batch = mm_ds.get_eval_batch()
