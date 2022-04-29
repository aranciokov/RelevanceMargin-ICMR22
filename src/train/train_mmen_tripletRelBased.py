import os
import sys

import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
import parsing
import defaults.EPIC_MMEN as EPIC_MMEN

from models.mmen import MMEN
from datasets.mmen_dataset import create_epic_mmen_dataset, create_youcook_mmen_dataset
from losses.triplet import TripletLoss
from evaluation import nDCG


from .train_jpose_tripletRelBased import get_relevances, parse_classes

def sample_triplets_for_dataset(dataset, cross_modal_weights_dict):
    for cross_modal_pair in cross_modal_weights_dict:
        if cross_modal_weights_dict[cross_modal_pair] > 0:
            dataset.sample_triplets(cross_modal_pair)

def train_epoch(model, dataset, cross_modal_weights_dict, loss_dict, optimiser, gpu=False,
                rel_margin=False):
    model.train()
    accum_loss = 0.
    avail_triplets = [pair for pair in cross_modal_weights_dict if cross_modal_weights_dict[pair] > 0]
    num_batches = int(np.ceil(dataset.x_len / dataset.batch_size))
    for i, batch_dict in enumerate(tqdm(dataset.get_triplet_batch(avail_triplets, gpu=gpu), total=num_batches)):
        optimiser.zero_grad()

        if "v_t_cls" in batch_dict and "t_v_cls" in batch_dict:
            tv_anc_cls, tv_pos_cls, tv_neg_cls = batch_dict["t_v_cls"]
            vt_anc_cls, vt_pos_cls, vt_neg_cls = batch_dict["v_t_cls"]
            batch_dict.pop("t_v_cls")
            batch_dict.pop("v_t_cls")
            #print("t_v", [(tv_anc_cls[i], tv_pos_cls[i], tv_neg_cls[i]) for i in range(5)],
            #      "v_t", [(vt_anc_cls[i], vt_pos_cls[i], vt_neg_cls[i]) for i in range(5)])
            assert "t_v_cls" not in batch_dict and "v_t_cls" not in batch_dict

        if "t_t_cls" in batch_dict and "v_v_cls" in batch_dict:
            tt_anc_cls, tt_pos_cls, tt_neg_cls = batch_dict["t_t_cls"]
            vv_anc_cls, vv_pos_cls, vv_neg_cls = batch_dict["v_v_cls"]
            batch_dict.pop("t_t_cls")
            batch_dict.pop("v_v_cls")
            #print("t_t", [(tt_anc_cls[i], tt_pos_cls[i], tt_neg_cls[i]) for i in range(5)],
            #      "v_v", [(vv_anc_cls[i], vv_pos_cls[i], vv_neg_cls[i]) for i in range(5)])
            assert "t_t_cls" not in batch_dict and "v_v_cls" not in batch_dict

        for cross_modal_pair in batch_dict:
            anc, pos, neg = batch_dict[cross_modal_pair]
            anc_id = cross_modal_pair[0]
            other_id = cross_modal_pair[1]
            forward_dict = [{anc_id: anc},
                    {other_id: pos},
                    {other_id: neg}
            ]
            out_dict = model.forward(forward_dict)
            anc_loss = out_dict[0][anc_id]
            pos_loss = out_dict[1][other_id]
            neg_loss = out_dict[2][other_id]
            if rel_margin:
                if cross_modal_pair == "tv":
                    _a, _p, _n = tv_anc_cls, tv_pos_cls, tv_neg_cls
                elif cross_modal_pair == "vt":
                    _a, _p, _n = vt_anc_cls, vt_pos_cls, vt_neg_cls
                elif cross_modal_pair == "tt":
                    _a, _p, _n = tt_anc_cls, tt_pos_cls, tt_neg_cls
                elif cross_modal_pair == "vv":
                    _a, _p, _n = vv_anc_cls, vv_pos_cls, vv_neg_cls

                #delta = get_relevances(parse_classes(_a), parse_classes(_p)) - \
                #        get_relevances(parse_classes(_a), parse_classes(_n))
                delta = 1 - \
                        get_relevances(parse_classes(_a), parse_classes(_n))

                if gpu:
                    delta = delta.cuda()
                loss = loss_dict[cross_modal_pair](anc_loss, pos_loss, neg_loss, margin=delta)
            else:
                loss = loss_dict[cross_modal_pair](anc_loss, pos_loss, neg_loss)
            loss = loss / len(batch_dict)
            accum_loss  += loss.data.item()

            """import torchviz
            print("Computing and rendering the computational graph")
            render_params = dict(model.named_parameters())
            dot = torchviz.make_dot(loss, params=render_params)
            dot.render()
            exit(0)"""

            loss.backward()
        optimiser.step()
    print('...{}'.format(accum_loss))
    return None


def initialise_nDCG_values(dataset):
    relevancy_matrix = dataset.relevancy_matrix
    vis_k_counts = nDCG.calculate_k_counts(relevancy_matrix)
    txt_k_counts = nDCG.calculate_k_counts(relevancy_matrix.T)

    vis_IDCG = nDCG.calculate_IDCG(relevancy_matrix, vis_k_counts)
    txt_IDCG = nDCG.calculate_IDCG(relevancy_matrix.T, txt_k_counts)

    k_counts_dict = {'v': vis_k_counts, 't': txt_k_counts}
    IDCG_dict = {'v': vis_IDCG, 't': txt_IDCG}

    return IDCG_dict, k_counts_dict


def test_epoch(model, dataset, writer, epoch_num, out_dir, gpu=False, final_run=False):
    model.eval()

    vis_feat, txt_feat = dataset.get_eval_batch(gpu=gpu)
    out_dict = model.forward([{'v': vis_feat}, {'t': txt_feat}])

    vis_feat = out_dict[0]['v']
    txt_feat = out_dict[1]['t']

    if gpu:
        vis_feat = vis_feat.cpu()
        txt_feat = txt_feat.cpu()
    vis_feat = vis_feat.detach().numpy()
    txt_feat = txt_feat.detach().numpy()

    vis_sim_matrix = vis_feat.dot(txt_feat.T)
    txt_sim_matrix = vis_sim_matrix.T

    vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix, dataset.relevancy_matrix,
            dataset.k_counts['v'], IDCG=dataset.IDCG_values['v'])
    txt_nDCG = nDCG.calculate_nDCG(txt_sim_matrix, dataset.relevancy_matrix.T,
            dataset.k_counts['t'], IDCG=dataset.IDCG_values['t'])

    print('{:.3f} {:.3f}'.format(vis_nDCG, txt_nDCG))
    writer.add_scalars('nDCG/', {'action/vid2txt': vis_nDCG,
                                 'action/txt2vid': txt_nDCG}, epoch_num)
    if final_run:
        vis_txt_identifier = np.concatenate((np.ones(vis_feat.shape[0]), np.zeros(txt_feat.shape[0])))
        writer.add_embedding(np.concatenate((vis_feat, txt_feat)), metadata=vis_txt_identifier, global_step=epoch_num)
        all_vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix,
                dataset['action'].relevancy_matrix, dataset['action'].k_values['v'],
                IDCG=dataset['action'].IDCG['v'], reduction=None)
        all_txt_nDCG = nDCG.calculate_nDCG(txt_sim_matrix,
                dataset['action'].relevancy_matrix.T, dataset['action'].k_values['t'],
                IDCG=dataset['action'].IDCG['t'], reduction=None)
        x_idxs = [dataset.action.x_idx_to_uid[idx] for idx in dataset.action._x_idxs]
        y_idxs = dataset.action._y_idxs
        all_dict = {'vis2txt': all_vis_nDCG, 'txt2vis': all_txt_nDCG, 
                'x_idxs': x_idxs, 'y_idxs': y_idxs}
        utils.output.save_results(out_dir, all_dict, 'pre_mean_nDCG')


def main(args):
    print(args)

    if args.dataset == "epic100":
        dataset_fn = create_epic_mmen_dataset
    elif args.dataset == "youcook2":
        dataset_fn = create_youcook_mmen_dataset
    else:
        exit()
    train_ds = dataset_fn(args.caption_type, is_train=True, batch_size=args.batch_size, num_triplets=args.num_triplets,
                                        all_nouns=args.all_noun_classes)
    test_ds = dataset_fn(args.caption_type, is_train=False, batch_size=args.batch_size, num_triplets=args.num_triplets)

    full_out_dir = utils.output.get_out_dir(args)
    print('#Saving models and results in {}'.format(full_out_dir))

    writer = SummaryWriter(log_dir=os.path.join(full_out_dir, 'results'))

    modality_dict = {
            't': {
                'num_layers': args.num_layers,
                'layer_sizes': [train_ds.y_size, args.embedding_size]
            },
            'v': {
                'num_layers': args.num_layers,
                'layer_sizes': [train_ds.x_size, args.embedding_size]
            }
    }
    mmen = MMEN(modality_dict)

    tt_loss = TripletLoss(args.margin, args.tt_weight)
    tv_loss = TripletLoss(args.margin, args.tv_weight)
    vt_loss = TripletLoss(args.margin, args.vt_weight)
    vv_loss = TripletLoss(args.margin, args.vv_weight)
    loss_dict = {'tt': tt_loss, 'tv': tv_loss, 'vt': vt_loss, 'vv': vv_loss}

    if args.gpu:
        mmen.cuda()
        for modality_pair in loss_dict:
            loss_dict[modality_pair].cuda()

    optimiser = optim.SGD(mmen.parameters(), lr=args.learning_rate, momentum=args.momentum)

    cross_modal_weights_dict = {'tt': args.tt_weight,
                                'tv': args.tv_weight,
                                'vt': args.vt_weight,
                                'vv': args.vv_weight
    }

    test_IDCG_values, test_k_counts = initialise_nDCG_values(test_ds)
    test_ds.IDCG_values = test_IDCG_values
    test_ds.k_counts = test_k_counts

    for epoch_num in range(args.num_epochs):
        print('Beginning Epoch {}'.format(epoch_num + 1))
        if (epoch_num + 1) % args.triplet_sampling_rate == 1:
            sample_triplets_for_dataset(train_ds, cross_modal_weights_dict)

        #Train
        train_epoch(mmen, train_ds, cross_modal_weights_dict, loss_dict, optimiser, gpu=args.gpu,
                    rel_margin=args.rel_margin)

        if (epoch_num + 1) % args.checkpoint_rate == 1:
            utils.output.save_model(full_out_dir, mmen, epoch_num)
        #Test
        test_epoch(mmen, test_ds, writer, epoch_num, full_out_dir, gpu=args.gpu)

    utils.output.save_model(full_out_dir, mmen, epoch_num)
    print('#Saved models and results in {}'.format(full_out_dir))

if __name__ == '__main__':
    import argparse
    parser = parsing.get_MMEN_parser("Multi-Modal Embedding Network (MMEN) using Triplets")

    parser.add_argument('--num-triplets', type=int, help='How many triplets to sample per anchor. [{}]'.format(EPIC_MMEN.num_triplets))
    parser.add_argument('--triplet-sampling-rate', type=int, help='Number of epochs in between triplet sampling. [{}]'.format(EPIC_MMEN.triplet_sampling_rate))
    parser.add_argument('--rel-margin', action='store_true', help='Number of epochs in between triplet sampling. [{}]'.format(EPIC_MMEN.triplet_sampling_rate))
    parser.add_argument('--all-noun-classes', action='store_true')
    parser.add_argument('--dataset', default='epic100', choices=['epic100', 'youcook2'])

    parser.set_defaults(
            num_triplets=EPIC_MMEN.num_triplets,
            triplet_sampling_rate=EPIC_MMEN.triplet_sampling_rate,
    )
    main(parser.parse_args())
