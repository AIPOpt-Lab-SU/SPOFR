import numpy as np
import math
import random
import copy
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models import convert_vars_to_gpu
from utils import logsumexp, shuffle_combined, exp_lr_scheduler, get_optimizer, serialize, transform_dataset
from evaluation import compute_dcg_rankings, evaluate_model, multiple_sample_and_log_probability, sample_double_stoch, compute_dcg_max, evaluate_soft_model #JK

from fairness_loss import GroupFairnessLoss, BaselineAshudeepGroupFairnessLoss, get_group_merits, get_group_identities

# JK
from networksJK import PolicyLP, PolicyLP_Plus, PolicyLP_PlusNeq, PolicyLP_PlusSP
from birkhoff import birkhoff_von_neumann_decomposition
import time
from models import LinearModel, init_weights # JK
import pickle
import pandas as pd
from gurobi_rank import *
from fairness_loss import test_fairness
import matplotlib.pyplot as plt


# JK delete
#torch.autograd.set_detect_anomaly(True)

def log_and_print(model,
                  data_reader,
                  writer: SummaryWriter,
                  step,
                  name="val",
                  experiment_name=None,
                  gpu=True,
                  fairness_evaluation=False,
                  exposure_relevance_plot=False,
                  deterministic=True,
                  group_fairness_evaluation=False,
                  args=None):
    position_bias_vector = 1. / torch.arange(1.,
                                             100.) ** args.position_bias_power
    if gpu:
        position_bias_vector = position_bias_vector.cuda()
    results = evaluate_model(
        model,
        data_reader,
        deterministic=deterministic,
        fairness_evaluation=fairness_evaluation,
        num_sample_per_query=args.sample_size,
        # position_bias_vector=1. / np.log2(2 + np.arange(200)),
        position_bias_vector=position_bias_vector,
        group_fairness_evaluation=group_fairness_evaluation,
        track_other_disparities=args.track_other_disparities,
        args=args)
    """
    Evaluate
    """
    if group_fairness_evaluation:
        avg_group_exposure_disparity, avg_group_asym_disparity = results[
                                                                     "avg_group_disparity"], results[
                                                                     "avg_group_asym_disparity"]
        if args.track_other_disparities:
            other_disparities = results["other_disparities"]

    avg_ndcg, avg_dcg, average_rank = results["ndcg"], results["dcg"], results["avg_rank"]
    """
    Return
    """
    returned = args.lambda_reward * avg_dcg
    if args.lambda_group_fairness > 0:
        returned -= args.lambda_group_fairness * avg_group_asym_disparity
    """
    Print
    """
    curve_pre_text = "{}_{}".format(name, args.fullinfo)
    print("Step {}, Average {}: NDCG: {}, DCG {}, Average Rank {}".
          format(step, curve_pre_text, avg_ndcg, avg_dcg, average_rank))
    if group_fairness_evaluation:
        print(
            "Average {} Group Exposure disparity: {}, Group Asymmetric disparity: {}".
                format(curve_pre_text, avg_group_exposure_disparity,
                       avg_group_asym_disparity, avg_group_asym_disparity))

    """
    Log
    """
    if experiment_name is None:
        experiment_name = "/"
    else:
        experiment_name += "/"
    if writer is not None:
        writer.add_scalars(experiment_name + "ndcg",
                           {curve_pre_text: avg_ndcg}, step)
        writer.add_scalars(experiment_name + "rank",
                           {curve_pre_text: average_rank}, step)
        writer.add_scalars(experiment_name + "dcg",
                           {curve_pre_text: avg_dcg}, step)
        writer.add_scalars(experiment_name + "metric",
                           {curve_pre_text: returned}, step)
        if group_fairness_evaluation:
            writer.add_scalars(experiment_name + "avg_group_disparity", {
                curve_pre_text:
                    avg_group_exposure_disparity
            }, step)
            writer.add_scalars(experiment_name + "avg_group_asym_disparity", {
                curve_pre_text:
                    avg_group_asym_disparity
            }, step)
            if args.track_other_disparities:
                for k, v in other_disparities.items():
                    writer.add_scalars(
                        experiment_name + "avg_group_asym_disparity",
                        {curve_pre_text + "_" + k: v[0]},
                        step)
                    writer.add_scalars(
                        experiment_name + "avg_group_disparity",
                        {curve_pre_text + "_" + k: v[1]},
                        step)

        # log on the train_dcg graph if evaluating on other training set
        if "_train--TRAIN" in name:
            writer.add_scalars(experiment_name + "train_dcg",
                               {curve_pre_text: avg_dcg}, step)
            writer.add_scalars(experiment_name + "train_ndcg",
                               {curve_pre_text: avg_ndcg}, step)

    return returned


def on_policy_training(data_reader,
                       validation_data_reader,
                       test_data_reader,
                       model,
                       writer=None,
                       experiment_name=None,
                       args=None):
    other_str = "full" if args.fullinfo == "partial" else "partial"
    position_bias_vector = 1. / torch.arange(1.,
                                             100.) ** args.position_bias_power
    lr = args.lr
    num_epochs = args.epochs
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    entropy_regularizer = args.entropy_regularizer

    relu = nn.ReLU()

    print("Starting training with the following config")
    print(
        "Batch size {}, Learning rate {}, Weight decay {}, Entropy Regularizer {}, Entreg Decay {} Sample size {}\n"
        "Lambda_reward: {}, lambda_ind_fairness:{}, lambda_group_fairness:{}".
            format(args.batch_size, lr, weight_decay, args.entropy_regularizer,
                   args.entreg_decay, sample_size,
                   args.lambda_reward, args.lambda_ind_fairness,
                   args.lambda_group_fairness))

    if args.gpu:
        print("Use GPU")
        model = model.cuda()
        position_bias_vector = position_bias_vector.cuda()

    optimizer = get_optimizer(model.parameters(), lr, args.optimizer,
                              weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_decay, min_lr=1e-6, verbose=True,
        patience=6)

    train_feats, train_rels = data_reader
    train_dataset = torch.utils.data.TensorDataset(train_feats, train_rels, shuffle=True)
    valid_feats, valid_rels = validation_data_reader
    len_train_set = len(train_feats) // args.batch_size + 1
    fairness_evaluation = True if args.lambda_ind_fairness > 0.0 else False
    group_fairness_evaluation = True

    #JK remove these triple quotes
    if group_fairness_evaluation and args.disparity_type != 'ashudeep':
        with torch.no_grad():
            group0_merit, group1_merit = get_group_merits(
                train_feats, train_rels, args.group_feat_id, args.group_feat_threshold, mean=False)
            print("Group 0 mean merit: {}, Group1 mean merit: {}".format(
                group0_merit, group1_merit))
            sign = 1.0 if group0_merit >= group1_merit else -1.0
            if args.disparity_type != 'ashudeep_mod':
                # random starting estimate for group_disparity indicator
                group_disparity_indicator_batch_size = args.group_disparity_indicator_batch_size * args.batch_size
                if group_disparity_indicator_batch_size > 4000:
                    group_disparity_indicator_batch_size = 4000
                if group_disparity_indicator_batch_size < 1000:
                    group_disparity_indicator_batch_size = 1000
                rand_ids = random.choices(
                    range(len(train_rels)), k=group_disparity_indicator_batch_size)
                group_disp_feats = train_feats[rand_ids]
                group_disp_rels = train_rels[rand_ids]
                if args.gpu:
                    group_disp_feats, group_disp_rels = group_disp_feats.cuda(), group_disp_rels.cuda()
                indicator_dataset = torch.utils.data.TensorDataset(group_disp_feats, group_disp_rels)
                indicator_dataloader = torch.utils.data.DataLoader(indicator_dataset, batch_size=args.batch_size,
                                                                   shuffle=True)
                indicator_disparities = []
                for data in indicator_dataloader:
                    feats, rel = data # JK what is rel?
                    scores = model(feats).squeeze(-1)

                    rankings = multiple_sample_and_log_probability(
                        scores, sample_size, return_prob=False, batch=True)

                    group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
                    indicator_disparity = GroupFairnessLoss.compute_multiple_group_disparity(rankings, rel,
                                                                                             group_identities,
                                                                                             group0_merit,
                                                                                             group1_merit,
                                                                                             position_bias_vector,
                                                                                             args.disparity_type,
                                                                                             noise=args.noise,
                                                                                             en=args.en).mean(dim=-1)
                    indicator_disparities.append(indicator_disparity)
                indicator_disparities = torch.cat(indicator_disparities, dim=0)
                print("Disparities indicator: {}".format(indicator_disparities.mean().item()))
    #### JK put back the triple quotes
    if args.early_stopping:
        time_since_best = 0
        best_metric = -1e6
        best_model = None
        best_epoch = None

    entropy_list = []
    sum_loss_list = []
    rewards_list = []
    fairness_loss_list = []
    reward_variance_list = []
    train_ndcg_list = []
    train_dcg_list = []
    weight_list = []


    # JK new temporary lists for new output format
    entropy_list_JK = []
    sum_loss_list_JK = []
    rewards_list_JK = []
    fairness_loss_list_JK = []
    max_fairness_loss_list_JK = []
    reward_variance_list_JK = []
    train_ndcg_list_JK = []
    train_dcg_list_JK = []
    weight_list_JK = []
    # JK save metrics in a list to print at end of training
    entropy_writelist_JK = []
    sum_loss_writelist_JK = []
    rewards_writelist_JK = []
    fairness_loss_writelist_JK = []
    max_fairness_loss_writelist_JK = []
    reward_variance_writelist_JK = []
    train_ndcg_writelist_JK = []
    train_dcg_writelist_JK = []
    weight_writelist_JK = []
    # JK test metric lists
    test_ndcg_list_JK = []
    test_dcg_list_JK = []
    test_rank_list_JK = []
    test_group_expos_disp_list_JK = []
    test_group_asym_disp_list_JK = []



    epoch_iterator = range(num_epochs)

    for epoch in epoch_iterator:
        print("Entering training Epoch {}".format(epoch))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        if args.progressbar:
            train_dataloader = tqdm(train_dataloader)

        for batch_id, data in enumerate(train_dataloader):
            feats, rel = data
            scores = model(feats).squeeze(-1)
            probs = nn.functional.softmax(scores, dim=-1)
            #JK = True     #JK remove
            #if not JK:    #JK remove

            rankings, log_model_prob = multiple_sample_and_log_probability(
                scores, sample_size, return_prob=True, batch=True)

            with torch.no_grad():
                ndcgs, dcgs = compute_dcg_rankings(rankings, rel)
                utility_list = ndcgs if args.reward_type == "ndcg" else dcgs
                # FAIRNESS constraints
                if args.lambda_group_fairness > 0.0:

                    group_identities = get_group_identities(
                        feats, args.group_feat_id, args.group_feat_threshold)

                    indicator_disparities, group_fairness_coeffs = GroupFairnessLoss.compute_group_fairness_coeffs_generic(
                        rankings, rel, group_identities,
                        position_bias_vector,
                        group0_merit,
                        group1_merit,
                        indicator_disparities,
                        args.disparity_type,
                        indicator_type=args.indicator_type,
                        noise=args.noise,
                        en=args.en)

            optimizer.zero_grad()


            if args.lambda_group_fairness != 0.0:
                rewards = args.lambda_reward * utility_list - \
                          args.lambda_group_fairness * group_fairness_coeffs
            else:
                rewards = args.lambda_reward * utility_list
            rewards = rewards / (args.lambda_reward + args.lambda_group_fairness)
            baseline = 0.0
            if args.use_baseline:
                if args.baseline_type == "value":
                    baseline = rewards.mean(dim=-1, keepdim=True)
                else:
                    raise NotImplementedError
            reinforce_loss = ((rewards - baseline) * (-log_model_prob)).mean()
            #reinforce_spo((rewards - baseline) * (-log_model_prob)).mean()     JK wtf?

            entropy_loss = 0.0
            entropy = get_entropy(probs).mean()
            if args.entropy_regularizer > 0.0:
                entropy_loss = entropy_regularizer * (-entropy)

            sum_loss = reinforce_loss + entropy_loss
            sum_loss.backward()

            optimizer.step()
            # log the reward/dcg variance
            sum_loss_list.append(sum_loss.item())
            if args.lambda_group_fairness != 0.0:
                fairness_loss_list.append(group_fairness_coeffs.mean().item())
            reward_variance_list.append(utility_list.var(dim=1).mean().item())
            rewards_list.append(utility_list.mean().item())
            entropy_list.append(entropy.item())
            """
            print("ndcgs = ")
            print( ndcgs    )
            print("ndcgs.shape = ")
            print( ndcgs.shape    )
            print("ndcgs.mean(dim=1) = ")
            print( ndcgs.mean(dim=1)    )
            print("ndcgs.mean(dim=1).mean() = ")
            print( ndcgs.mean(dim=1).mean()    )
            print("ndcgs.mean(dim=1).sum() = ")
            print( ndcgs.mean(dim=1).sum()    )
            input("waiting")
            """
            train_ndcg_list.append(ndcgs.mean(dim=1).sum().item())
            train_dcg_list.append(dcgs.mean(dim=1).sum().item())
            weight_list.append(rel.sum().item())

            # JK save for my own ouputs with some changes to these metrics
            """
            print("Before saving violations: ")
            print("group_identities = ")
            print( group_identities )
            #print("group_fairness_coeffs = ")
            #print( group_fairness_coeffs  )
            #print("group_fairness_coeffs.mean() = ")
            #print( group_fairness_coeffs.mean()  )
            #print("group_fairness_coeffs.shape = ")
            #print( group_fairness_coeffs.shape  )
            input("waiting")
            """



            # JK note - right here, group_fairness_coeffs has shape ( batch_size * sample_size )

            if args.lambda_group_fairness != 0.0:
                fairness_loss_list_JK.append(group_fairness_coeffs.mean().item())
                max_fairness_loss_list_JK.append(group_fairness_coeffs.mean(1).max().item())    # the max expected violation among policies in the batch
            reward_variance_list_JK.append(utility_list.var(dim=1).mean().item())
            rewards_list_JK.append(utility_list.mean().item())            # This is ndcg or dcg depending on the setting
            entropy_list_JK.append(entropy.item())
            train_ndcg_list_JK.append(ndcgs.mean(dim=1).mean().item())    # JK changed sum to mean
            train_dcg_list_JK.append(dcgs.mean(dim=1).mean().item())      # JK changed sum to mean
            weight_list_JK.append(rel.sum().item())


            step = epoch * len_train_set + batch_id

        #if step % args.write_losses_interval == 0 and step > 0:    # just do every epoch instead
        """
            LOGGING
        """
        weight_sum = np.sum(weight_list)
        """  JK commenting writer
        log_output = "\nAverages of last 1000 rewards: {}, ndcgs: {}, dcgs: {}".format(
            np.mean(rewards_list),
            np.mean(train_ndcg_list),
            np.sum(train_dcg_list) / weight_sum)
        if args.lambda_group_fairness > 0.0:
            log_output += " disparity: {}".format(
                np.mean(fairness_loss_list))
        print(log_output)
        if writer is not None:
            writer.add_scalars(experiment_name + "/{}_sum_train_loss".format(
                args.fullinfo), {"sum_loss": np.mean(sum_loss_list)}, step)
            writer.add_scalars(
                experiment_name + "/{}_var_reward".format(args.fullinfo),
                {"var_reward": np.mean(reward_variance_list)}, step)
            writer.add_scalars(
                experiment_name + "/{}_entropy".format(args.fullinfo),
                {"entropy": np.mean(entropy_list)}, step)
            if args.lambda_group_fairness != 0.0:
                writer.add_scalars(experiment_name + "/{}_fairness_loss".format(
                    args.fullinfo), {"fairness_loss": np.mean(fairness_loss_list)}, step)
            writer.add_scalars(
                experiment_name + "/{}_train_ndcg".format(args.fullinfo),
                {"train_ndcg": np.mean(train_ndcg_list)}, step)
            writer.add_scalars(
                experiment_name + "/{}_train_dcg".format(args.fullinfo),
                {"train_dcg": np.sum(train_dcg_list) / np.sum(weight_list)}, step)
        """
        # JK save metrics for my output
        if args.lambda_group_fairness != 0.0:
            fairness_loss_writelist_JK.append(np.mean(fairness_loss_list_JK))     # each element the mean over a batch; this is mean over 'epoch' or writing interval
            max_fairness_loss_writelist_JK.append(np.max(max_fairness_loss_list_JK))     # for now, this carries the maximum expected fairness violation of any policy in the entire epoch.
        rewards_writelist_JK.append(np.mean(rewards_list_JK))
        entropy_writelist_JK.append(np.mean(entropy_list_JK))
        train_ndcg_writelist_JK.append(np.mean(train_ndcg_list_JK))
        train_dcg_writelist_JK.append(np.mean(train_dcg_list_JK))
        weight_writelist_JK.append(np.mean(weight_list_JK))
        # JK reset the temporary lists
        fairness_loss_list_JK = []
        max_fairness_loss_list_JK = []
        reward_variance_list_JK = []
        sum_loss_list_JK = []
        entropy_list_JK = []
        weight_list_JK = []
        train_ndcg_list_JK = []
        train_dcg_list_JK = []
        # End JK new metrics


        fairness_loss_list = []
        reward_variance_list = []
        sum_loss_list = []
        entropy_list = []
        weight_list = []
        train_ndcg_list = []
        train_dcg_list = []

        #if step % args.evaluate_interval == 0 and step > 0:   # just do every epoch instead

        """   JK don't need validation for now
        print(
            "Evaluating on validation set: iteration {}/{} of epoch {}".
                format(batch_id, len_train_set, epoch))
        curr_metric = log_and_print(
            model,
            (valid_feats, valid_rels),
            writer,
            step,
            "TEST_full--TRAIN",
            experiment_name,
            args.gpu,
            fairness_evaluation=fairness_evaluation,
            # exposure_relevance_plot=exposure_relevance_plot,
            deterministic=args.validation_deterministic,
            group_fairness_evaluation=group_fairness_evaluation,
            args=args)

        # LR and Entropy decay
        scheduler.step(curr_metric)
        """







        # JK do the test evaluation again
        # note - this function is called from within log_and_print so the work is done twice
        print("Entering test data evaluation")
        results = evaluate_model(
            model,
            #data_reader,
            test_data_reader,   # JK switch from eval on train to test data
            group0_merit = group0_merit,   # JK   new arg
            group1_merit = group1_merit,   # JK   new arg
            deterministic=args.validation_deterministic,
            fairness_evaluation=fairness_evaluation,
            num_sample_per_query=args.sample_size,
            # position_bias_vector=1. / np.log2(2 + np.arange(200)),
            position_bias_vector=position_bias_vector,
            group_fairness_evaluation=group_fairness_evaluation,
            track_other_disparities=args.track_other_disparities,
            args=args)

        test_ndcg_list_JK.append(results["ndcg"])      # JK evaluation.py line 504 for origin of these
        test_dcg_list_JK.append(results["dcg"])
        test_rank_list_JK.append(results["avg_rank"])
        if group_fairness_evaluation:
            test_group_expos_disp_list_JK.append(results["avg_group_disparity"])
            test_group_asym_disp_list_JK.append(results["avg_group_asym_disparity"])
        fair_viols_quantiles_test = results["fair_viols_quantiles"]
        # JK end test metric collection


        # """
        # Early stopping
        # """
        #if args.early_stopping:
        #    if best_model is None or curr_metric > best_metric + abs(best_metric) * 0.0001:
        #        best_metric = curr_metric
        #        best_model = copy.deepcopy(model)
        #        best_epoch = epoch
        #        time_since_best = 0
        #    else:
        #        time_since_best += 1
        #    if time_since_best >= 3:
        #        entropy_regularizer = args.entreg_decay * entropy_regularizer
        #        print("Decay entropy regularizer to {}".format(entropy_regularizer))
        #    if time_since_best >= args.stop_patience:
        #        print(
        #            "Validation set metric hasn't increased in 10 steps. Exiting")
        #        return best_model, best_metric


        # JK end this epoch



    # Final eval on the training set (need fairness viol quantiles)
    print("Entering train data evaluation")
    results = evaluate_model(
                model,
                data_reader,
                #test_data_reader,   # JK switch from eval on train to test data
                group0_merit = group0_merit,   # JK   new arg
                group1_merit = group1_merit,   # JK   new arg
                deterministic=args.validation_deterministic,
                fairness_evaluation=fairness_evaluation,
                num_sample_per_query=args.sample_size,
                # position_bias_vector=1. / np.log2(2 + np.arange(200)),
                position_bias_vector=position_bias_vector,
                group_fairness_evaluation=group_fairness_evaluation,
                track_other_disparities=args.track_other_disparities,
                args=args)
    fair_viols_quantiles = results['fair_viols_quantiles']

    print("Entering valid data evaluation")
    results_valid = evaluate_model(
        model,
        #data_reader,
        validation_data_reader,   # JK switch from eval on train to test data
        group0_merit = group0_merit,   # JK   new arg
        group1_merit = group1_merit,   # JK   new arg
        deterministic=args.validation_deterministic,
        fairness_evaluation=fairness_evaluation,
        num_sample_per_query=args.sample_size,
        # position_bias_vector=1. / np.log2(2 + np.arange(200)),
        position_bias_vector=position_bias_vector,
        group_fairness_evaluation=group_fairness_evaluation,
        track_other_disparities=args.track_other_disparities,
        args=args)
    valid_ndcg_final = results["ndcg"]      # JK evaluation.py line 504 for origin of these
    valid_dcg_final  = results["dcg"]
    valid_rank_final = results["avg_rank"]
    valid_group_expos_final = results["avg_group_disparity"]
    valid_group_asym_final  = results["avg_group_asym_disparity"]
    fair_viols_quantiles_valid = results["fair_viols_quantiles"]


    outs = {}
    outs['entropy_writelist_JK']  =  entropy_writelist_JK
    #outs["sum_loss_writelist_JK"] =  sum_loss_writelist_JK
    outs["rewards_writelist_JK"]  =  rewards_writelist_JK
    outs["fairness_loss_writelist_JK"] =  fairness_loss_writelist_JK
    #outs["reward_variance_writelist_JK"] = reward_variance_writelist_JK
    outs["train_ndcg_writelist_JK"] = train_ndcg_writelist_JK
    outs["train_dcg_writelist_JK"] = train_dcg_writelist_JK
    outs["weight_writelist_JK"] = weight_writelist_JK
    outs["test_ndcg_list_JK"] = test_ndcg_list_JK
    outs["test_dcg_list_JK"] = test_dcg_list_JK
    outs["test_rank_list_JK"] = test_rank_list_JK
    outs["test_group_expos_disp_list_JK"] = test_group_expos_disp_list_JK
    outs["test_group_asym_disp_list_JK"] = test_group_asym_disp_list_JK



    pickle.dump( outs, open('./plots_out/'+ "FULTR_" + args.output_tag + '_' + str(args.index) + '.p', 'wb') )



    plt.plot( range(len(train_ndcg_writelist_JK)),       train_ndcg_writelist_JK,  label = 'NDCG' )
    plt.plot( range(len(fairness_loss_writelist_JK)), fairness_loss_writelist_JK, label = 'Violation' )
    plt.legend()
    plt.savefig(  './png/'+ "FULTR_training_" +args.output_tag+'_'+str(args.index)+'.png'  )
    plt.close()

    plt.plot( range(len(test_ndcg_list_JK)),       test_ndcg_list_JK,  label = 'NDCG' )
    plt.plot( range(len(test_group_expos_disp_list_JK)), test_group_expos_disp_list_JK, label = 'Violation' )
    plt.legend()
    plt.savefig(  './png/'+ "FULTR_testing_" +args.output_tag+'_'+str(args.index)+'.png'  )


    csv_outs = {}
    csv_outs['entropy_final']  =  entropy_writelist_JK[-1]
    #csv_outs["sum_loss_final"] =  sum_loss_writelist_JK[-1]
    csv_outs["rewards_final"]  =  rewards_writelist_JK[-1]
    if args.lambda_group_fairness != 0.0:
        csv_outs["fairness_loss_final"] =  fairness_loss_writelist_JK[-1]
        csv_outs["max_fairness_loss_final"] =  max_fairness_loss_writelist_JK[-1]
    #csv_outs["reward_variance_final"] = reward_variance_writelist_JK[-1]
    csv_outs["train_ndcg_final"] = train_ndcg_writelist_JK[-1]
    csv_outs["train_dcg_final"] = train_dcg_writelist_JK[-1]
    csv_outs["weight_final"] = weight_writelist_JK[-1]
    csv_outs["test_ndcg_final"] = test_ndcg_list_JK[-1]
    csv_outs["test_dcg_final"] = test_dcg_list_JK[-1]
    csv_outs["test_rank_final"] = test_rank_list_JK[-1]
    csv_outs["test_group_expos_disp_final"] = test_group_expos_disp_list_JK[-1]
    csv_outs["test_group_asym_disp_final"] = test_group_asym_disp_list_JK[-1]
    csv_outs["valid_ndcg_final"] = valid_ndcg_final
    csv_outs["valid_dcg_final"] = valid_dcg_final
    csv_outs["valid_rank_final"] = valid_rank_final
    csv_outs["valid_group_expos_final"] = valid_group_expos_final
    csv_outs["valid_group_asym_final"] = valid_group_asym_final
    csv_outs["fair_viol_q_100"] = fair_viols_quantiles['1.00']
    csv_outs["fair_viol_q_95"]  = fair_viols_quantiles['0.95']
    csv_outs["fair_viol_q_90"]  = fair_viols_quantiles['0.90']
    csv_outs["fair_viol_q_85"]  = fair_viols_quantiles['0.85']
    csv_outs["fair_viol_q_80"]  = fair_viols_quantiles['0.80']
    csv_outs["fair_viol_q_75"]  = fair_viols_quantiles['0.75']
    csv_outs["fair_viol_q_70"]  = fair_viols_quantiles['0.70']
    csv_outs["fair_viol_q_65"]  = fair_viols_quantiles['0.65']
    csv_outs["fair_viol_q_60"]  = fair_viols_quantiles['0.60']
    csv_outs["fair_viol_q_55"]  = fair_viols_quantiles['0.55']
    csv_outs["fair_viol_q_50"]  = fair_viols_quantiles['0.50']
    csv_outs["fair_viol_q_100_test"] = fair_viols_quantiles_test['1.00']
    csv_outs["fair_viol_q_95_test"]  = fair_viols_quantiles_test['0.95']
    csv_outs["fair_viol_q_90_test"]  = fair_viols_quantiles_test['0.90']
    csv_outs["fair_viol_q_85_test"]  = fair_viols_quantiles_test['0.85']
    csv_outs["fair_viol_q_80_test"]  = fair_viols_quantiles_test['0.80']
    csv_outs["fair_viol_q_75_test"]  = fair_viols_quantiles_test['0.75']
    csv_outs["fair_viol_q_70_test"]  = fair_viols_quantiles_test['0.70']
    csv_outs["fair_viol_q_65_test"]  = fair_viols_quantiles_test['0.65']
    csv_outs["fair_viol_q_60_test"]  = fair_viols_quantiles_test['0.60']
    csv_outs["fair_viol_q_55_test"]  = fair_viols_quantiles_test['0.55']
    csv_outs["fair_viol_q_50_test"]  = fair_viols_quantiles_test['0.50']
    csv_outs["fair_viol_q_100_valid"] = fair_viols_quantiles_valid['1.00']
    csv_outs["fair_viol_q_95_valid"]  = fair_viols_quantiles_valid['0.95']
    csv_outs["fair_viol_q_90_valid"]  = fair_viols_quantiles_valid['0.90']
    csv_outs["fair_viol_q_85_valid"]  = fair_viols_quantiles_valid['0.85']
    csv_outs["fair_viol_q_80_valid"]  = fair_viols_quantiles_valid['0.80']
    csv_outs["fair_viol_q_75_valid"]  = fair_viols_quantiles_valid['0.75']
    csv_outs["fair_viol_q_70_valid"]  = fair_viols_quantiles_valid['0.70']
    csv_outs["fair_viol_q_65_valid"]  = fair_viols_quantiles_valid['0.65']
    csv_outs["fair_viol_q_60_valid"]  = fair_viols_quantiles_valid['0.60']
    csv_outs["fair_viol_q_55_valid"]  = fair_viols_quantiles_valid['0.55']
    csv_outs["fair_viol_q_50_valid"]  = fair_viols_quantiles_valid['0.50']





    csv_outs["index"] = args.index
    csv_outs["epochs"] = args.epochs
    csv_outs["lr"] = args.lr
    csv_outs["hidden_layer"] = args.hidden_layer
    csv_outs["optimizer"] = args.optimizer
    csv_outs["quad_reg"] = args.quad_reg
    csv_outs["partial_train_data"] = args.partial_train_data
    csv_outs["partial_val_data"] = args.partial_val_data
    csv_outs["full_test_data"] = args.full_test_data
    csv_outs["log_dir"] = args.log_dir
    csv_outs["sample_size"] = args.sample_size
    csv_outs["batch_size"] = args.batch_size
    csv_outs["soft_train"] = args.soft_train
    csv_outs["disparity_type"] = args.disparity_type
    csv_outs["lambda_group_fairness"] = args.lambda_group_fairness
    csv_outs["index"] = args.index

    csv_outs = {k:[v] for (k,v) in csv_outs.items()   }
    df_outs = pd.DataFrame.from_dict(csv_outs)
    outPathCsv = './csv/'+ "FULTR_" + args.output_tag + '_' + str(args.index)  + ".csv"

    df_outs.to_csv(outPathCsv)



    for (k,v) in csv_outs.items():
        print("{}:  {}".format(k,v))

    print("Outputs saved")
    quit()
    return model #, curr_metric











# JK 0922
# training soft policy distribution with SPO+
test_DSM_ndcg_list_JK = []
test_DSM_mean_viol_list_JK = []
test_DSM_max_viol_list_JK = []
test_ndcg_list_JK = []
test_dcg_list_JK = []
test_rank_list_JK = []
test_group_expos_disp_list_JK = []
test_group_asym_disp_list_JK = []
def soft_policy_training_spo(data_reader,
                             validation_data_reader,
                             test_data_reader,
                             model,
                             writer=None,
                             experiment_name=None,
                             args=None):
    other_str = "full" if args.fullinfo == "partial" else "partial"
    position_bias_vector = 1. / torch.arange(1.,
                                             100.) ** args.position_bias_power
    lr = args.lr
    num_epochs = args.epochs
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    entropy_regularizer = args.entropy_regularizer

    relu = nn.ReLU()

    print("Starting training with the following config")
    print(
        "Batch size {}, Learning rate {}, Weight decay {}, Entropy Regularizer {}, Entreg Decay {} Sample size {}\n"
        "Lambda_reward: {}, lambda_ind_fairness:{}, lambda_group_fairness:{}".
            format(args.batch_size, lr, weight_decay, args.entropy_regularizer,
                   args.entreg_decay, sample_size,
                   args.lambda_reward, args.lambda_ind_fairness,
                   args.lambda_group_fairness))

    if args.gpu:
        print("Use GPU")
        model = model.cuda()
        position_bias_vector = position_bias_vector.cuda()

    optimizer = get_optimizer(model.parameters(), lr, args.optimizer,
                              weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_decay, min_lr=1e-6, verbose=True,
        patience=6)

    train_feats, train_rels = data_reader
    train_dataset = torch.utils.data.TensorDataset(train_feats, train_rels, shuffle=True)
    valid_feats, valid_rels = validation_data_reader
    len_train_set = len(train_feats) // args.batch_size + 1
    fairness_evaluation = True if args.lambda_ind_fairness > 0.0 else False
    group_fairness_evaluation = True

    #JK remove these triple quotes
    if group_fairness_evaluation and args.disparity_type != 'ashudeep':
        with torch.no_grad():
            group0_merit, group1_merit = get_group_merits(
                train_feats, train_rels, args.group_feat_id, args.group_feat_threshold, mean=False)
            print("Group 0 mean merit: {}, Group1 mean merit: {}".format(
                group0_merit, group1_merit))
            sign = 1.0 if group0_merit >= group1_merit else -1.0
            if args.disparity_type != 'ashudeep_mod':
                # random starting estimate for group_disparity indicator
                group_disparity_indicator_batch_size = args.group_disparity_indicator_batch_size * args.batch_size
                if group_disparity_indicator_batch_size > 4000:
                    group_disparity_indicator_batch_size = 4000
                if group_disparity_indicator_batch_size < 1000:
                    group_disparity_indicator_batch_size = 1000
                rand_ids = random.choices(
                    range(len(train_rels)), k=group_disparity_indicator_batch_size)
                group_disp_feats = train_feats[rand_ids]
                group_disp_rels = train_rels[rand_ids]
                if args.gpu:
                    group_disp_feats, group_disp_rels = group_disp_feats.cuda(), group_disp_rels.cuda()
                indicator_dataset = torch.utils.data.TensorDataset(group_disp_feats, group_disp_rels)
                indicator_dataloader = torch.utils.data.DataLoader(indicator_dataset, batch_size=args.batch_size,
                                                                   shuffle=True)
                indicator_disparities = []
                # JK make a placeholder model for this sampling
                # Q: why do they use scores from the untrained model?
                model_kwargs = {'clamp': args.clamp}
                dummy_model = LinearModel(
                    input_dim=args.input_dim, **model_kwargs)
                if args.gpu:
                    dummy_model = dummy_model.cuda()
                for data in indicator_dataloader:
                    feats, rel = data
                    scores = dummy_model(feats).squeeze(-1)  # replace the model with dummy_model

                    rankings = multiple_sample_and_log_probability(
                        scores, sample_size, return_prob=False, batch=True)

                    group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
                    indicator_disparity = GroupFairnessLoss.compute_multiple_group_disparity(rankings, rel,
                                                                                             group_identities,
                                                                                             group0_merit,
                                                                                             group1_merit,
                                                                                             position_bias_vector,
                                                                                             args.disparity_type,
                                                                                             noise=args.noise,
                                                                                             en=args.en).mean(dim=-1)
                    indicator_disparities.append(indicator_disparity)
                indicator_disparities = torch.cat(indicator_disparities, dim=0)
                print("Disparities indicator: {}".format(indicator_disparities.mean().item()))
    #### JK put back the triple quotes

    if args.early_stopping:
        time_since_best = 0
        best_metric = -1e6
        best_model = None
        best_epoch = None

    entropy_list = []
    sum_loss_list = []
    rewards_list = []
    fairness_loss_list = []
    reward_variance_list = []
    train_ndcg_list = []
    train_dcg_list = []
    weight_list = []

    training_losses = []
    training_regrets = []
    training_viols = []

    epoch_iterator = range(num_epochs)


    # set up solvers for hot starts
    solver_dict = {}
    for i in range(1,args.list_len):


        if args.allow_unfairness:
            # Delta Fairness
            # Google solver only
            gids = torch.zeros(args.list_len).long()
            gids[:i] = 1
            s,x = ort_setup_Neq(args.list_len, gids, args.disparity_type, group0_merit, group1_merit, args.fairness_gap)
            key = int(gids.sum().item())      # JK check this key - not used?
            solver_dict[i] = ort_policyLP(s,x)
        else:
            # Perfect Fairness
            gids = torch.zeros(args.list_len).long()
            gids[:i] = 1
            s,x = ort_setup(args.list_len, gids, args.disparity_type, group0_merit, group1_merit)
            key = int(gids.sum().item())      # JK check this key - not used?
            solver_dict[i] = ort_policyLP(s,x)

    for epoch in epoch_iterator:

        epoch_losses = []  # this takes each batch loss and save mean after the epoch, then resets
        epoch_regrets = []
        epoch_viols = []  # each element is the average (mean) violation of policies in the batch

        print("Entering training Epoch {}".format(epoch))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        if args.progressbar:
            train_dataloader = tqdm(train_dataloader)

        for batch_id, data in enumerate(train_dataloader):

            feats, rel = data
            batsize = feats.shape[0]

            group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
            if group_identities.bool().all(1).any().item() or (1-group_identities).bool().all(1).any().item():
                continue
                # skip the iteration if only one group appears

            # Form the cross product between group a ID embedding and the document scores

            if args.embed_groups:
                scores, group_embed = model(feats, group_identities)
                scores= scores.squeeze(-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), group_embed.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            # Concatenate the document scores with group ID and predict N**2 independent QP coefficients using a MLP
            elif args.embed_quadscore:
                score_cross = model(feats, group_identities).squeeze(-1)
                #scores = score_cross.detach()  # doing this to avoid crash when scores.shape[0] is used to check batch size
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            else:
                scores = model(feats).squeeze(-1)
                test_dscts = ( 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2) ).repeat(scores.shape[0],1,1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), test_dscts.view(scores.shape[0],1,-1)  ).reshape(scores.shape[0],-1)



            # score_cross is used as the linear objective coefficients for the QP layer
            """
            if not args.allow_unfairness:
                policy_lp = PolicyLP_Plus(N=scores.shape[1], eps = args.quad_reg, position_bias_vector = position_bias_vector)
            else:
                policy_lp = PolicyLP_PlusNeq(N=scores.shape[1], eps = args.quad_reg, position_bias_vector = position_bias_vector, delta=args.fairness_gap)

            if args.gpu:
                policy_lp = policy_lp.cuda()
            p_mat = policy_lp(score_cross, group_identities)
            p_mat = relu( p_mat )
            """



            ####################
            ########
            ### NEW SPO work JK

            # Note this is for one sample
            test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)

            #true_costs = torch.bmm( rel.view(scores.shape[0],-1,1), test_dscts.view(scores.shape[0],1,-1)).view(scores.shape[0],-1)
            true_costs = torch.bmm( rel.view(batsize,-1,1), test_dscts.view(batsize,1,-1)).view(batsize,1,-1)

            """
            print('true_costs.size() = ')
            print( true_costs.size()    )
            print('score_cross.size() = ')
            print( score_cross.size()    )
            print('group_identities.size() = ')
            print( group_identities.size()    )
            """


            grad = []
            p_mat = []
            regrets = []
            with torch.no_grad():
                dcg_max = compute_dcg_max(rel)  # redundant, defined again below

                for i in range(batsize):


                    spo_group_ids = group_identities[i].detach().numpy()
                    sorting_ind = np.argsort(spo_group_ids)[::-1]
                    reverse_ind = np.argsort(sorting_ind)

                    solver = solver_dict[ int(spo_group_ids.sum().item()) ]

                    V_true  = true_costs[i].squeeze().detach().double().numpy() #compute 'true' cost coefficients here
                    V_true1 = true_costs[i].squeeze().detach().double().numpy()                    #delete
                    V_true  = (V_true.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()

                    #sol_true = solver.solve(V_true.detach().double().numpy())
                    sol_true = solver.solve(V_true)
                    #sol_true = sol_true.view(args.list_len,args.list_len)[reverse_ind].flatten()
                    sol_true = sol_true.reshape((args.list_len,args.list_len))[reverse_ind].flatten()


                    V_pred   = score_cross[i].squeeze().detach().double().numpy() #compute 'pred' cost coefficients here
                    V_pred   = (V_pred.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                    sol_pred = solver.solve(V_pred)
                    sol_pred = sol_pred.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                    p_mat.append(torch.Tensor(sol_pred).view(args.list_len,args.list_len))

                    V_spo    = (2*V_pred - V_true)
                    V_spo    = (V_spo.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                    sol_spo  = solver.solve(V_spo)
                    sol_spo  = sol_spo.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                    #reg = torch.dot(V_true1,(sol_true - sol_pred))
                    reg = torch.Tensor(  [np.dot(V_true1,(sol_true - sol_pred))]  )
                    regrets.append(reg)
                    use_reg = True
                    if use_reg:
                        grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                    else:
                        grad.append( torch.Tensor(sol_spo - sol_true)  )


                    """
                    #   might need to convert this to numpy ops if using again
                    # No hot start
                    if args.solver_software == 'gurobi':
                        spo_group_ids = group_identities[i]

                        V_true = true_costs[i].squeeze() #compute 'true' cost coefficients here
                        V_true1 = true_costs[i].squeeze()
                        solver = solver_dict[ spo_group_ids.sum().item() ]

                        sol_true = grb_solve(args.list_len, V_true.detach().numpy(), spo_group_ids)

                        V_pred = score_cross[i].squeeze()
                        sol_pred = grb_solve(args.list_len, V_pred.detach().numpy(), spo_group_ids)
                        p_mat.append(torch.Tensor(sol_pred.view(args.list_len,args.list_len)))

                        #print("NO_HS")
                        #print("obj (true)")
                        #print(   torch.dot(V_true1, sol_true)   )
                        #print("obj (pred)")
                        #print(   torch.dot(V_true1, sol_pred)   )


                        V_spo  = (2*V_pred - V_true)
                        sol_spo  = grb_solve(args.list_len, V_spo.detach().numpy(), spo_group_ids)

                        reg = torch.dot(V_true1,(sol_true - sol_pred))
                        regrets.append(reg)
                        use_reg = True
                        if use_reg:
                            grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                        else:
                            grad.append( torch.Tensor(sol_spo - sol_true)  )

                    if args.solver_software == 'google':
                        spo_group_ids = group_identities[i]

                        V_true = true_costs[i].squeeze() #compute 'true' cost coefficients here
                        solver = solver_dict[ spo_group_ids.sum().item() ]
                        sol_true = ort_solve(args.list_len, V_true.detach().numpy(), spo_group_ids)

                        # DELETE
                        #tff = test_fairness(torch.Tensor(sol_true.view(args.list_len,args.list_len)), spo_group_ids, position_bias_vector)
                        #input('test_fairness = {}'.format(tff))
                        # END

                        V_pred = score_cross[i].squeeze()
                        sol_pred = ort_solve(args.list_len, V_pred.detach().numpy(), spo_group_ids)
                        p_mat.append(torch.Tensor(sol_pred.view(args.list_len,args.list_len)))

                        V_spo  = (2*V_pred - V_true)
                        sol_spo  = ort_solve(args.list_len, V_spo.detach().numpy(), spo_group_ids)

                        reg = torch.dot(V_true,(sol_true - sol_pred))
                        regrets.append(reg)
                        use_reg = True
                        if use_reg:
                            grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                        else:
                            grad.append( torch.Tensor(sol_spo - sol_true)  )

                    """

                p_mat = torch.stack(p_mat)
                regrets = torch.stack(regrets)

            optimizer.zero_grad()

            score_cross.backward(gradient=torch.stack(grad))

            #loss = score_cross
            #loss.backward(gradient=torch.stack(grad))   # JK look up - does this take incoming or outgoing
            optimizer.step()



            with torch.no_grad():
                dcg_max = compute_dcg_max(rel)
                test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
                if args.gpu:
                    test_dscts = test_dscts.cuda()
                #v_unsq = v.unsqueeze(1)
                #f_unsq = f.unsqueeze(1).permute(0,2,1)
                #vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
                loss_a = torch.bmm( p_mat, test_dscts.view(batsize,-1,1) )
                loss_b = torch.bmm( rel.view(batsize,1,-1), loss_a ).squeeze()
                loss_norm = loss_b.squeeze() / dcg_max
                loss = loss_norm.mean()


                # Find average violation

                fair_viol_mean_batch = 0
                for kk in range(len(p_mat)):
                    fair_viol_mean_batch += test_fairness( p_mat[kk], group_identities[kk], position_bias_vector, args.disparity_type, group0_merit, group1_merit )
                fair_viol_mean_batch /= len(p_mat)

                epoch_viols.append(fair_viol_mean_batch)



                """ use this to check the 'true costs' against evalutation reward function
                true_costs = torch.bmm( rel.view(scores.shape[0],-1,1),  test_dscts.view(scores.shape[0],1,-1)   ).view(scores.shape[0],1,-1)
                loss2_b = torch.bmm(true_costs, p_mat.view(scores.shape[0],-1).unsqueeze(1).permute(0,2,1))
                loss2_norm = loss2_b.squeeze() / dcg_max
                loss2 = -loss2_norm.mean()
                """






            #### END NEW SPO work
            ########
            ####################



            # log the reward/dcg variance
            sum_loss_list.append(loss.item())
            #print("loss = {}".format(loss.item()))
            #print("regret = {}".format(regrets.mean().item()))
            epoch_losses.append( loss.item() )
            epoch_regrets.append( regrets.mean().item() )



            #step = epoch * len_train_set + batch_id  # JK added here
            #if step % args.evaluate_interval == 0 and step > 0:
        #print(
        #    "Evaluating on validation set: iteration {}/{} of epoch {}".
        #        format(batch_id, len_train_set, epoch))



        # JK do the custom test routine for this policy type
        print("Entering evalutation of test data")
        results = evaluate_soft_model(
                    model,
                    #data_reader,
                    test_data_reader,   # JK switch from eval on train to test data
                    group0_merit = group0_merit,   # JK   new arg
                    group1_merit = group1_merit,   # JK   new arg
                    deterministic=args.validation_deterministic,
                    fairness_evaluation=fairness_evaluation,
                    num_sample_per_query=args.sample_size,
                    # position_bias_vector=1. / np.log2(2 + np.arange(200)),
                    position_bias_vector=position_bias_vector,
                    group_fairness_evaluation=group_fairness_evaluation,
                    track_other_disparities=args.track_other_disparities,
                    args=args)
        fair_viols_quantiles_test = results["fair_viols_quantiles"]


        test_DSM_ndcg_list_JK.append(results["DSM_ndcg"])
        test_DSM_mean_viol_list_JK.append(results["DSM_mean_viol"])
        test_DSM_max_viol_list_JK.append(results["DSM_max_viol"])

        test_ndcg_list_JK.append(results["ndcg"])      # JK evaluation.py line 504 for origin of these
        test_dcg_list_JK.append(results["ndcg"])
        test_rank_list_JK.append(results["avg_rank"])

        if group_fairness_evaluation:
            test_group_expos_disp_list_JK.append(results["avg_group_disparity"])
            test_group_asym_disp_list_JK.append(results["avg_group_asym_disparity"])
        # JK end test metric collection


        if args.early_stopping:
            if best_model is None or curr_metric > best_metric + abs(best_metric) * 0.0001:
                best_metric = curr_metric
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                time_since_best = 0
            else:
                time_since_best += 1
            if time_since_best >= 3:
                entropy_regularizer = args.entreg_decay * entropy_regularizer
                print("Decay entropy regularizer to {}".format(entropy_regularizer))
            if time_since_best >= args.stop_patience:
                print(
                    "Validation set metric hasn't increased in 10 steps. Exiting")
                return best_model, best_metric

        # epoch loop end
        training_losses.append(  np.mean(epoch_losses).item()   )
        training_regrets.append( np.mean(epoch_regrets).item() )
        training_viols.append(   np.mean(epoch_viols).item()     )
        epoch_losses = []
        epoch_regrets = []
        epoch_viols = []


    print("Entering evalutation of valid data")
    results = evaluate_soft_model(
                model,
                validation_data_reader,
                #test_data_reader,   # JK switch from eval on train to test data
                group0_merit = group0_merit,   # JK   new arg
                group1_merit = group1_merit,   # JK   new arg
                deterministic=args.validation_deterministic,
                fairness_evaluation=fairness_evaluation,
                num_sample_per_query=args.sample_size,
                # position_bias_vector=1. / np.log2(2 + np.arange(200)),
                position_bias_vector=position_bias_vector,
                group_fairness_evaluation=group_fairness_evaluation,
                track_other_disparities=args.track_other_disparities,
                args=args)
    fair_viols_quantiles_valid = results['fair_viols_quantiles']

    valid_ndcg_final = results["ndcg"]
    valid_dcg_final  = results["dcg"]
    valid_rank_final = results["avg_rank"]
    valid_group_expos_disp_final = results["avg_group_disparity"]
    valid_group_asym_disp_final  = results["avg_group_asym_disparity"]

    if epoch > 2:
        if valid_ndcg_final > valid_ndcg_current:
            model_current = copy.deepcopy(model)
            fails = 0
        else:
            fails = fails + 1

        if fails > patience:
            break




    # Do a final evaluation on the training set (need fairness quantiles)
    print("Entering evalutation of train data")
    results = evaluate_soft_model(
                model,
                data_reader,
                #test_data_reader,   # JK switch from eval on train to test data
                group0_merit = group0_merit,   # JK   new arg
                group1_merit = group1_merit,   # JK   new arg
                deterministic=args.validation_deterministic,
                fairness_evaluation=fairness_evaluation,
                num_sample_per_query=args.sample_size,
                # position_bias_vector=1. / np.log2(2 + np.arange(200)),
                position_bias_vector=position_bias_vector,
                group_fairness_evaluation=group_fairness_evaluation,
                track_other_disparities=args.track_other_disparities,
                args=args)
    fair_viols_quantiles = results['fair_viols_quantiles']


    plt.plot( range(len(training_losses)),  training_losses,  label = 'NDCG' )
    plt.plot( range(len(training_regrets)), training_regrets, label = 'Regret' )
    plt.plot( range(len(training_viols)),   training_viols, label = 'Violation' )
    plt.legend()
    plt.savefig(  './png/'+ "LP_" +args.output_tag+'_'+str(args.index)+'.png'  )
    print('Outputs saved')




    # JK are all these metric defined yet?
    outs = {}
    outs["test_DSM_ndcg_list_JK"] = test_DSM_ndcg_list_JK
    outs["test_DSM_mean_viol_list_JK"] = test_DSM_mean_viol_list_JK
    outs["test_DSM_max_viol_list_JK"]  = test_DSM_max_viol_list_JK
    outs["test_ndcg_list_JK"] = test_ndcg_list_JK
    outs["test_dcg_list_JK"] = test_dcg_list_JK
    outs["test_rank_list_JK"] = test_rank_list_JK
    outs["test_group_expos_disp_list_JK"] = test_group_expos_disp_list_JK
    outs["test_group_asym_disp_list_JK"] = test_group_asym_disp_list_JK
    outs["training_loss"]   = training_losses    # this is just avg NDCG by epoch
    outs["training_regret"] = training_regrets



    pickle.dump( outs, open('./plots_out/'+ "LP_" + args.output_tag + '_plots_out_' + str(args.index) + '.p', 'wb') )


    csv_outs = {}

    csv_outs['test_DSM_ndcg_final']  =  test_DSM_ndcg_list_JK[-1]  if  test_DSM_ndcg_list_JK!=[] else 0
    csv_outs['test_DSM_ndcg_max']  =  np.max(test_DSM_ndcg_list_JK).item()  if  test_DSM_ndcg_list_JK!=[] else 0
    test_DSM_ndcg_max_ind =  np.argmax(test_DSM_ndcg_list_JK)
    csv_outs['test_DSM_mean_viol_argmax_JK'] = test_DSM_mean_viol_list_JK[test_DSM_ndcg_max_ind]
    csv_outs['test_DSM_max_viol_argmax_JK']  = test_DSM_max_viol_list_JK[test_DSM_ndcg_max_ind]
    csv_outs['test_ndcg_final']  =  test_ndcg_list_JK[-1]  if  test_ndcg_list_JK!=[] else 0
    csv_outs["test_dcg_final"]   =  test_dcg_list_JK[-1]   if  test_dcg_list_JK!=[] else 0
    csv_outs["test_rank_final"]  =  test_rank_list_JK[-1]  if  test_rank_list_JK!=[] else 0
    csv_outs["test_group_expos_disp_final"] =  test_group_expos_disp_list_JK[-1] if test_group_expos_disp_list_JK!=[] else 0
    csv_outs["test_group_asym_disp_final"] = test_group_asym_disp_list_JK[-1] if test_group_asym_disp_list_JK!=[] else 0

    csv_outs["valid_ndcg_final"] =  valid_ndcg_final
    csv_outs["valid_dcg_final"]  =  valid_dcg_final
    csv_outs["valid_rank_final"] =  valid_rank_final
    csv_outs["valid_group_expos_disp_final"] =  valid_group_expos_disp_final
    csv_outs["valid_group_asym_disp_final"]  =  valid_group_asym_disp_final

    csv_outs["max_training_loss"] = np.max(training_losses).item()
    csv_outs["min_training_regret"] = np.min(training_regrets).item()
    csv_outs["fair_viol_q_100"] = fair_viols_quantiles['1.00']
    csv_outs["fair_viol_q_95"]  = fair_viols_quantiles['0.95']
    csv_outs["fair_viol_q_90"]  = fair_viols_quantiles['0.90']
    csv_outs["fair_viol_q_85"]  = fair_viols_quantiles['0.85']
    csv_outs["fair_viol_q_80"]  = fair_viols_quantiles['0.80']
    csv_outs["fair_viol_q_75"]  = fair_viols_quantiles['0.75']
    csv_outs["fair_viol_q_70"]  = fair_viols_quantiles['0.70']
    csv_outs["fair_viol_q_65"]  = fair_viols_quantiles['0.65']
    csv_outs["fair_viol_q_60"]  = fair_viols_quantiles['0.60']
    csv_outs["fair_viol_q_55"]  = fair_viols_quantiles['0.55']
    csv_outs["fair_viol_q_50"]  = fair_viols_quantiles['0.50']

    csv_outs["fair_viol_q_100_test"] = fair_viols_quantiles_test['1.00']
    csv_outs["fair_viol_q_95_test"]  = fair_viols_quantiles_test['0.95']
    csv_outs["fair_viol_q_90_test"]  = fair_viols_quantiles_test['0.90']
    csv_outs["fair_viol_q_85_test"]  = fair_viols_quantiles_test['0.85']
    csv_outs["fair_viol_q_80_test"]  = fair_viols_quantiles_test['0.80']
    csv_outs["fair_viol_q_75_test"]  = fair_viols_quantiles_test['0.75']
    csv_outs["fair_viol_q_70_test"]  = fair_viols_quantiles_test['0.70']
    csv_outs["fair_viol_q_65_test"]  = fair_viols_quantiles_test['0.65']
    csv_outs["fair_viol_q_60_test"]  = fair_viols_quantiles_test['0.60']
    csv_outs["fair_viol_q_55_test"]  = fair_viols_quantiles_test['0.55']
    csv_outs["fair_viol_q_50_test"]  = fair_viols_quantiles_test['0.50']

    csv_outs["fair_viol_q_100_valid"] = fair_viols_quantiles_valid['1.00']
    csv_outs["fair_viol_q_95_valid"]  = fair_viols_quantiles_valid['0.95']
    csv_outs["fair_viol_q_90_valid"]  = fair_viols_quantiles_valid['0.90']
    csv_outs["fair_viol_q_85_valid"]  = fair_viols_quantiles_valid['0.85']
    csv_outs["fair_viol_q_80_valid"]  = fair_viols_quantiles_valid['0.80']
    csv_outs["fair_viol_q_75_valid"]  = fair_viols_quantiles_valid['0.75']
    csv_outs["fair_viol_q_70_valid"]  = fair_viols_quantiles_valid['0.70']
    csv_outs["fair_viol_q_65_valid"]  = fair_viols_quantiles_valid['0.65']
    csv_outs["fair_viol_q_60_valid"]  = fair_viols_quantiles_valid['0.60']
    csv_outs["fair_viol_q_55_valid"]  = fair_viols_quantiles_valid['0.55']
    csv_outs["fair_viol_q_50_valid"]  = fair_viols_quantiles_valid['0.50']




    csv_outs["index"] = args.index
    csv_outs["epochs"] = args.epochs
    csv_outs["lr"] = args.lr
    csv_outs["hidden_layer"] = args.hidden_layer
    csv_outs["optimizer"] = args.optimizer
    csv_outs["quad_reg"] = args.quad_reg
    csv_outs["partial_train_data"] = args.partial_train_data
    csv_outs["partial_val_data"] = args.partial_val_data
    csv_outs["full_test_data"] = args.full_test_data
    csv_outs["log_dir"] = args.log_dir
    csv_outs["sample_size"] = args.sample_size
    csv_outs["batch_size"] = args.batch_size
    csv_outs["soft_train"] = args.soft_train
    csv_outs["disparity_type"] = args.disparity_type
    csv_outs["embed_groups"] = args.embed_groups
    csv_outs["embed_quadscore"] = args.embed_quadscore
    csv_outs["allow_unfairness"] = args.allow_unfairness
    csv_outs["fairness_gap"] = args.fairness_gap
    csv_outs["index"] = args.index
    csv_outs["seed"]  = args.seed


    csv_outs = {k:[v] for (k,v) in csv_outs.items()   }
    df_outs = pd.DataFrame.from_dict(csv_outs)
    outPathCsv = './csv/'+ "LP_" + args.output_tag + '_' + str(args.index)  + ".csv"


    df_outs.to_csv(outPathCsv)



    print("test_ndcg_list_JK = ")
    print( test_ndcg_list_JK )
    print("test_dcg_list_JK = ")
    print( test_dcg_list_JK )
    print("test_rank_list_JK = ")
    print( test_rank_list_JK )
    print("test_group_expos_disp_list_JK = ")
    print( test_group_expos_disp_list_JK )
    print("test_group_asym_disp_list_JK = ")
    print( test_group_asym_disp_list_JK )
    quit()  # JK this is a hack to escape without crashing; curr_metric below is undefined. We have to return something to the main routine.
    return model, curr_metric





def get_entropy(probs):
    return -torch.sum(torch.log(probs + 1e-10) * probs, dim=-1)


def compute_baseline(state, type="max"):
    if type == "max":
        print("Depracated: Doesn't work anymore")
        rel = state
        max_dcg = 0.0
        for i in range(sum(rel)):
            max_dcg += 1.0 / math.log(2 + i)
        return max_dcg
    elif type == "value":
        rankings, rewards_list = state
        # state is sent as a set of rankings sampled using the policy and
        # the set of relevant documents
        return np.mean(rewards_list)
    else:
        print("-----No valid reward type selected-------")


def compute_multiple_log_model_probability(scores, rankings, gpu=None):
    subtracts = torch.zeros_like(rankings, dtype=torch.float)
    log_probs = torch.zeros_like(rankings, dtype=torch.float)
    batch_index = torch.arange(rankings.size()[0])
    scores = scores.squeeze(-1)
    if gpu:
        subtracts, log_probs = convert_vars_to_gpu([subtracts, log_probs])
        batch_index = convert_vars_to_gpu(batch_index)
    for j in range(rankings.size()[1]):
        posj = rankings[:, j]
        log_probs[:, j] = scores[posj] - logsumexp(scores - subtracts, dim=1)
        subtracts[batch_index, posj] = scores[posj] + 1e6
    return torch.sum(log_probs, dim=1)


def compute_log_model_probability(scores, ranking, gpu=None):
    """
    more stable version
    if rel is provided, use it to calculate probability only till
    all the relevant documents are found in the ranking
    """
    subtracts = torch.zeros_like(scores)
    log_probs = torch.zeros_like(scores)
    if gpu:
        subtracts, log_probs = convert_vars_to_gpu([subtracts, log_probs])
    for j in range(scores.size()[0]):
        posj = ranking[j]
        log_probs[j] = scores[posj] - logsumexp(scores - subtracts, dim=0)
        subtracts[posj] = scores[posj] + 1e6
    return torch.sum(log_probs)



































# Deprecated
# JK 0813
# training expected reward over soft policy distribution
test_ndcg_list_JK = []
test_dcg_list_JK = []
test_rank_list_JK = []
test_group_expos_disp_list_JK = []
test_group_asym_disp_list_JK = []
def soft_policy_training(data_reader,
                       validation_data_reader,
                       test_data_reader,
                       model,
                       writer=None,
                       experiment_name=None,
                       args=None):
    other_str = "full" if args.fullinfo == "partial" else "partial"
    position_bias_vector = 1. / torch.arange(1.,
                                             100.) ** args.position_bias_power
    lr = args.lr
    num_epochs = args.epochs
    save_state_dict = args.save_state_dict
    sample_size = args.sample_size
    entropy_regularizer = args.entropy_regularizer

    relu = nn.ReLU()

    print("Starting training with the following config")
    print(
        "Batch size {}, Learning rate {}, Weight decay {}, Entropy Regularizer {}, Entreg Decay {} Sample size {}\n"
        "Lambda_reward: {}, lambda_ind_fairness:{}, lambda_group_fairness:{}".
            format(args.batch_size, lr, save_state_dict, args.entropy_regularizer,
                   args.entreg_decay, sample_size,
                   args.lambda_reward, args.lambda_ind_fairness,
                   args.lambda_group_fairness))

    if args.gpu:
        print("Use GPU")
        model = model.cuda()
        position_bias_vector = position_bias_vector.cuda()

    optimizer = get_optimizer(model.parameters(), lr, args.optimizer,
                              weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_decay, min_lr=1e-6, verbose=True,
        patience=6)

    train_feats, train_rels = data_reader
    train_dataset = torch.utils.data.TensorDataset(train_feats, train_rels)
    valid_feats, valid_rels = validation_data_reader
    len_train_set = len(train_feats) // args.batch_size + 1
    fairness_evaluation = True if args.lambda_ind_fairness > 0.0 else False
    group_fairness_evaluation = True

    #JK remove these triple quotes
    if group_fairness_evaluation and args.disparity_type != 'ashudeep':
        with torch.no_grad():
            group0_merit, group1_merit = get_group_merits(
                train_feats, train_rels, args.group_feat_id, args.group_feat_threshold, mean=False)
            print("Group 0 mean merit: {}, Group1 mean merit: {}".format(
                group0_merit, group1_merit))
            sign = 1.0 if group0_merit >= group1_merit else -1.0
            if args.disparity_type != 'ashudeep_mod':
                # random starting estimate for group_disparity indicator
                group_disparity_indicator_batch_size = args.group_disparity_indicator_batch_size * args.batch_size
                if group_disparity_indicator_batch_size > 4000:
                    group_disparity_indicator_batch_size = 4000
                if group_disparity_indicator_batch_size < 1000:
                    group_disparity_indicator_batch_size = 1000
                rand_ids = random.choices(
                    range(len(train_rels)), k=group_disparity_indicator_batch_size)
                group_disp_feats = train_feats[rand_ids]
                group_disp_rels = train_rels[rand_ids]
                if args.gpu:
                    group_disp_feats, group_disp_rels = group_disp_feats.cuda(), group_disp_rels.cuda()
                indicator_dataset = torch.utils.data.TensorDataset(group_disp_feats, group_disp_rels)
                indicator_dataloader = torch.utils.data.DataLoader(indicator_dataset, batch_size=args.batch_size,
                                                                   shuffle=True)
                indicator_disparities = []
                # JK make a placeholder model for this sampling
                # Q: why do they use scores from the untrained model?
                model_kwargs = {'clamp': args.clamp}
                dummy_model = LinearModel(
                    input_dim=args.input_dim, **model_kwargs)
                if args.gpu:
                    dummy_model = dummy_model.cuda()
                for data in indicator_dataloader:
                    feats, rel = data
                    scores = dummy_model(feats).squeeze(-1)  # replace the model with dummy_model

                    rankings = multiple_sample_and_log_probability(
                        scores, sample_size, return_prob=False, batch=True)

                    group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
                    indicator_disparity = GroupFairnessLoss.compute_multiple_group_disparity(rankings, rel,
                                                                                             group_identities,
                                                                                             group0_merit,
                                                                                             group1_merit,
                                                                                             position_bias_vector,
                                                                                             args.disparity_type,
                                                                                             noise=args.noise,
                                                                                             en=args.en).mean(dim=-1)
                    indicator_disparities.append(indicator_disparity)
                indicator_disparities = torch.cat(indicator_disparities, dim=0)
                print("Disparities indicator: {}".format(indicator_disparities.mean().item()))
    #### JK put back the triple quotes

    if args.early_stopping:
        time_since_best = 0
        best_metric = -1e6
        best_model = None
        best_epoch = None

    entropy_list = []
    sum_loss_list = []
    rewards_list = []
    fairness_loss_list = []
    reward_variance_list = []
    train_ndcg_list = []
    train_dcg_list = []
    weight_list = []

    epoch_iterator = range(num_epochs)


    for epoch in epoch_iterator:
        print("Entering training Epoch {}".format(epoch))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        if args.progressbar:
            train_dataloader = tqdm(train_dataloader)

        for batch_id, data in enumerate(train_dataloader):

            feats, rel = data

            group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
            if group_identities.bool().all(1).any().item() or (1-group_identities).bool().all(1).any().item():
                continue
                # skip the iteration if only one group appears

            # Form the cross product between group a ID embedding and the document scores

            if args.embed_groups:
                scores, group_embed = model(feats, group_identities)
                scores= scores.squeeze(-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), group_embed.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            # Concatenate the document scores with group ID and predict N**2 independent QP coefficients using a MLP
            elif args.embed_quadscore:
                score_cross = model(feats, group_identities).squeeze(-1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            else:
                scores = model(feats).squeeze(-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)

            # score_cross is used as the linear objective coefficients for the QP layer
            if not args.allow_unfairness:
                sparse = True
                if not sparse:
                    policy_lp = PolicyLP_Plus(N=scores.shape[1], eps = args.quad_reg, position_bias_vector = position_bias_vector)
                else:
                    policy_lp = PolicyLP_PlusSP(N=scores.shape[1], eps = args.quad_reg, position_bias_vector = position_bias_vector)

            else:
                policy_lp = PolicyLP_PlusNeq(N=scores.shape[1], eps = args.quad_reg, position_bias_vector = position_bias_vector, delta=args.fairness_gap)

            if args.gpu:
                policy_lp = policy_lp.cuda()
            p_mat = policy_lp(score_cross, group_identities)
            p_mat = relu( p_mat )
            #rankings, log_model_prob = sample_double_stoch(                  # JK no sampling required
            #        p_mat, sample_size, return_prob=True, batch=True)

            #ndcgs, dcgs = compute_dcg_rankings(rankings, rel)


            optimizer.zero_grad()

            dcg_max = compute_dcg_max(rel)
            test_dscts = ( 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2) ).repeat(scores.shape[0],1,1)
            if args.gpu:
                test_dscts = test_dscts.cuda()
            loss_a = torch.bmm( p_mat, test_dscts.view(scores.shape[0],-1,1) )
            loss_b = torch.bmm( rel.view(scores.shape[0],1,-1), loss_a ).squeeze()
            loss_norm = loss_b / dcg_max
            loss = -loss_norm.mean()

            loss.backward()

            optimizer.step()
            # log the reward/dcg variance
            sum_loss_list.append(loss.item())
            print("loss = {}".format(loss.item()))

            """ No need to evaluate the training data policies
            sampling_interval = 100    # JK
            step = epoch * len_train_set + batch_id # JK
            if step % sampling_interval == 0 and step > 0:   # JK
                ######## JK sampling 0816  #########
                with torch.no_grad():
                    P = p_mat.cpu().detach().numpy()
                    R = []
                    for it, policy in enumerate(P):
                        decomp = birkhoff_von_neumann_decomposition(policy)
                        convex_coeffs, permutations = zip(*decomp)
                        permutations = np.array(permutations)
                        rolls = torch.multinomial(torch.Tensor(convex_coeffs),sample_size,replacement=True).numpy()
                        #rolls = np.random.multinomial(sample_size, np.array(convex_coeffs))  # sample the permutations based on convex_coeffs
                        p_sample = permutations[rolls]       # access the permutations
                        r_sample = p_sample.argmax(2)        # convert to rankings
                        r_sample = torch.tensor( r_sample )  # convert to same datatype as FULTR implementation
                        R.append(r_sample)
                        #print("Finished policy sampling iteration {}".format(it))
                    rankings = torch.stack(R)
                    if args.gpu:
                        rankings = rankings.cuda()   # JK testing
                    print("Sampling Complete")

                    #with torch.no_grad():
                    ndcgs, dcgs = compute_dcg_rankings(rankings, rel)
                    utility_list = ndcgs if args.reward_type == "ndcg" else dcgs
                    # FAIRNESS constraints
                    if args.lambda_group_fairness > 0.0:
                        if args.unweighted_fairness:
                            rel = (rel > 0.0).float()
                        group_identities = get_group_identities(
                            feats, args.group_feat_id, args.group_feat_threshold)
                        if args.disparity_type == "ashudeep_mod":
                            group_fairness_coeffs = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                                rankings, rel, group_identities, position_bias_vector, sign=sign)
                        elif args.disparity_type == "ashudeep":
                            group_fairness_coeffs = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                                rankings, rel, group_identities, position_bias_vector)
                        else:
                            indicator_disparities, group_fairness_coeffs = GroupFairnessLoss.compute_group_fairness_coeffs_generic(
                                rankings, rel, group_identities,
                                position_bias_vector,
                                group0_merit,
                                group1_merit,
                                indicator_disparities,
                                args.disparity_type,
                                indicator_type=args.indicator_type,
                                noise=args.noise,
                                en=args.en)
                ####################################
                # END with no_grad



                if args.lambda_group_fairness != 0.0:
                    fairness_loss_list.append(group_fairness_coeffs.mean().item())
                reward_variance_list.append(utility_list.var(dim=1).mean().item())
                rewards_list.append(utility_list.mean().item())
                #entropy_list.append(entropy.item())   JK  no entropy
                train_ndcg_list.append(ndcgs.mean(dim=1).sum().item())
                train_dcg_list.append(dcgs.mean(dim=1).sum().item())
                weight_list.append(rel.sum().item())

            step = epoch * len_train_set + batch_id
            if step % args.write_losses_interval == 0 and step > 0:

                #    LOGGING

                weight_sum = np.sum(weight_list)
                log_output = "\nAverages of last 1000 rewards: {}, ndcgs: {}, dcgs: {}".format(
                    np.mean(rewards_list),
                    np.mean(train_ndcg_list),
                    np.sum(train_dcg_list) / weight_sum)
                if args.lambda_group_fairness > 0.0:
                    log_output += " disparity: {}".format(
                        np.mean(fairness_loss_list))
                print(log_output)
                if writer is not None:
                    writer.add_scalars(experiment_name + "/{}_sum_train_loss".format(
                        args.fullinfo), {"sum_loss": np.mean(sum_loss_list)}, step)
                    writer.add_scalars(
                        experiment_name + "/{}_var_reward".format(args.fullinfo),
                        {"var_reward": np.mean(reward_variance_list)}, step)
                    writer.add_scalars(
                        experiment_name + "/{}_entropy".format(args.fullinfo),
                        {"entropy": 0}, step)   # JK 0 because no entropy
                        #{"entropy": np.mean(entropy_list)}, step)
                    if args.lambda_group_fairness != 0.0:
                        writer.add_scalars(experiment_name + "/{}_fairness_loss".format(
                            args.fullinfo), {"fairness_loss": np.mean(fairness_loss_list)}, step)
                    writer.add_scalars(
                        experiment_name + "/{}_train_ndcg".format(args.fullinfo),
                        {"train_ndcg": np.mean(train_ndcg_list)}, step)
                    writer.add_scalars(
                        experiment_name + "/{}_train_dcg".format(args.fullinfo),
                        {"train_dcg": np.sum(train_dcg_list) / np.sum(weight_list)}, step)
                fairness_loss_list = []
                reward_variance_list = []
                sum_loss_list = []
                entropy_list = []
                weight_list = []
                train_ndcg_list = []
                train_dcg_list = []
            """
            step = epoch * len_train_set + batch_id  # JK added here
            if step % args.evaluate_interval == 0 and step > 0:
                print(
                    "Evaluating on validation set: iteration {}/{} of epoch {}".
                        format(batch_id, len_train_set, epoch))
                """ JK skip this
                curr_metric = log_and_print(
                    model,
                    (valid_feats, valid_rels),
                    writer,
                    step,
                    "TEST_full--TRAIN",
                    experiment_name,
                    args.gpu,
                    fairness_evaluation=fairness_evaluation,
                    # exposure_relevance_plot=exposure_relevance_plot,
                    deterministic=args.validation_deterministic,
                    group_fairness_evaluation=group_fairness_evaluation,
                    args=args)

                # LR and Entropy decay
                scheduler.step(curr_metric)
                #
                # Early stopping
                #
                """


                # JK do the custom test routine for this policy type
                results = evaluate_soft_model(
                            model,
                            data_reader,
                            deterministic=args.validation_deterministic,
                            fairness_evaluation=fairness_evaluation,
                            num_sample_per_query=args.sample_size,
                            # position_bias_vector=1. / np.log2(2 + np.arange(200)),
                            position_bias_vector=position_bias_vector,
                            group_fairness_evaluation=group_fairness_evaluation,
                            track_other_disparities=args.track_other_disparities,
                            args=args)

                test_ndcg_list_JK.append(results["ndcg"])      # JK evaluation.py line 504 for origin of these
                test_dcg_list_JK.append(results["ndcg"])
                test_rank_list_JK.append(results["avg_rank"])
                if group_fairness_evaluation:
                    test_group_expos_disp_list_JK.append(results["avg_group_disparity"])
                    test_group_asym_disp_list_JK.append(results["avg_group_asym_disparity"])
                # JK end test metric collection



                if args.early_stopping:
                    if best_model is None or curr_metric > best_metric + abs(best_metric) * 0.0001:
                        best_metric = curr_metric
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch
                        time_since_best = 0
                    else:
                        time_since_best += 1
                    if time_since_best >= 3:
                        entropy_regularizer = args.entreg_decay * entropy_regularizer
                        print("Decay entropy regularizer to {}".format(entropy_regularizer))
                    if time_since_best >= args.stop_patience:
                        print(
                            "Validation set metric hasn't increased in 10 steps. Exiting")
                        return best_model, best_metric


    # JK are all these metric defined yet?
    outs = {}
    outs["test_ndcg_list_JK"] = test_ndcg_list_JK
    outs["test_dcg_list_JK"] = test_dcg_list_JK
    outs["test_rank_list_JK"] = test_rank_list_JK
    outs["test_group_expos_disp_list_JK"] = test_group_expos_disp_list_JK
    outs["test_group_asym_disp_list_JK"] = test_group_asym_disp_list_JK
    pickle.dump( outs, open('./plots_out/'+ "LP_" + "fultr_benchmarks1_" + 'plots_out' + "_run_" + str(args.index) + '.p', 'wb') )


    csv_outs = {}
    csv_outs['test_ndcg_final']  =  test_ndcg_list_JK[:-1]
    csv_outs["test_dcg_final"] =  test_dcg_list_JK[:-1]
    csv_outs["test_rank_final"]  =  test_rank_list_JK[:-1]
    csv_outs["test_group_expos_disp_final"] =  test_group_expos_disp_list_JK[:-1]
    csv_outs["test_group_asym_disp_final"] = test_group_asym_disp_list_JK[:-1]

    csv_outs["index"] = args.index
    csv_outs["epochs"] = args.epochs
    csv_outs["lr"] = args.lr
    csv_outs["hidden_layer"] = args.hidden_layer
    csv_outs["optimizer"] = args.optimizer
    csv_outs["quad_reg"] = args.quad_reg
    csv_outs["partial_train_data"] = args.partial_train_data
    csv_outs["partial_val_data"] = args.partial_val_data
    csv_outs["full_test_data"] = args.full_test_data
    csv_outs["log_dir"] = args.log_dir
    csv_outs["sample_size"] = args.sample_size
    csv_outs["batch_size"] = args.batch_size
    csv_outs["soft_train"] = args.soft_train
    csv_outs["disparity_type"] = args.disparity_type
    csv_outs["embed_groups"] = args.embed_groups
    csv_outs["embed_quadscore"] = args.embed_quadscore
    csv_outs["allow_unfairness"] = args.allow_unfairness
    csv_outs["fairness_gap"] = args.fairness_gap
    csv_outs["index"] = args.index

    csv_outs = {k:[v] for (k,v) in csv_outs.items()   }
    df_outs = pd.DataFrame.from_dict(csv_outs)
    outPathCsv = './csv/'+ "LP_" + "fultr_benchmarks2_" + 'csv_out' + "_run_" + str(args.index) + ".csv"
    df_outs.to_csv(outPathCsv)


    # Epoch Complete
    print("test_ndcg_list_JK = ")
    print( test_ndcg_list_JK )
    print("test_dcg_list_JK = ")
    print( test_dcg_list_JK )
    print("test_rank_list_JK = ")
    print( test_rank_list_JK )
    print("test_group_expos_disp_list_JK = ")
    print( test_group_expos_disp_list_JK )
    print("test_group_asym_disp_list_JK = ")
    print( test_group_asym_disp_list_JK )
    quit()  # JK this is a hack to escape without crashing; curr_metric below is undefined. We have to return something to the main routine.
    return model, curr_metric







# JK 0922
# training soft policy distribution with SPO+
test_ndcg_list_JK = []
test_dcg_list_JK = []
test_rank_list_JK = []
test_group_expos_disp_list_JK = []
test_group_asym_disp_list_JK = []
def soft_policy_training_spo_saved(data_reader,
                             validation_data_reader,
                             model,
                             writer=None,
                             experiment_name=None,
                             args=None):
    other_str = "full" if args.fullinfo == "partial" else "partial"
    position_bias_vector = 1. / torch.arange(1.,
                                             100.) ** args.position_bias_power
    lr = args.lr
    num_epochs = args.epochs
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    entropy_regularizer = args.entropy_regularizer

    relu = nn.ReLU()

    print("Starting training with the following config")
    print(
        "Batch size {}, Learning rate {}, Weight decay {}, Entropy Regularizer {}, Entreg Decay {} Sample size {}\n"
        "Lambda_reward: {}, lambda_ind_fairness:{}, lambda_group_fairness:{}".
            format(args.batch_size, lr, weight_decay, args.entropy_regularizer,
                   args.entreg_decay, sample_size,
                   args.lambda_reward, args.lambda_ind_fairness,
                   args.lambda_group_fairness))

    if args.gpu:
        print("Use GPU")
        model = model.cuda()
        position_bias_vector = position_bias_vector.cuda()

    optimizer = get_optimizer(model.parameters(), lr, args.optimizer,
                              weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_decay, min_lr=1e-6, verbose=True,
        patience=6)

    train_feats, train_rels = data_reader
    train_dataset = torch.utils.data.TensorDataset(train_feats, train_rels)
    valid_feats, valid_rels = validation_data_reader
    len_train_set = len(train_feats) // args.batch_size + 1
    fairness_evaluation = True if args.lambda_ind_fairness > 0.0 else False
    group_fairness_evaluation = True

    #JK remove these triple quotes
    if group_fairness_evaluation and args.disparity_type != 'ashudeep':
        with torch.no_grad():
            group0_merit, group1_merit = get_group_merits(
                train_feats, train_rels, args.group_feat_id, args.group_feat_threshold, mean=False)
            print("Group 0 mean merit: {}, Group1 mean merit: {}".format(
                group0_merit, group1_merit))
            sign = 1.0 if group0_merit >= group1_merit else -1.0
            if args.disparity_type != 'ashudeep_mod':
                # random starting estimate for group_disparity indicator
                group_disparity_indicator_batch_size = args.group_disparity_indicator_batch_size * args.batch_size
                if group_disparity_indicator_batch_size > 4000:
                    group_disparity_indicator_batch_size = 4000
                if group_disparity_indicator_batch_size < 1000:
                    group_disparity_indicator_batch_size = 1000
                rand_ids = random.choices(
                    range(len(train_rels)), k=group_disparity_indicator_batch_size)
                group_disp_feats = train_feats[rand_ids]
                group_disp_rels = train_rels[rand_ids]
                if args.gpu:
                    group_disp_feats, group_disp_rels = group_disp_feats.cuda(), group_disp_rels.cuda()
                indicator_dataset = torch.utils.data.TensorDataset(group_disp_feats, group_disp_rels)
                indicator_dataloader = torch.utils.data.DataLoader(indicator_dataset, batch_size=args.batch_size,
                                                                   shuffle=True)
                indicator_disparities = []
                # JK make a placeholder model for this sampling
                # Q: why do they use scores from the untrained model?
                model_kwargs = {'clamp': args.clamp}
                dummy_model = LinearModel(
                    input_dim=args.input_dim, **model_kwargs)
                if args.gpu:
                    dummy_model = dummy_model.cuda()
                for data in indicator_dataloader:
                    feats, rel = data
                    scores = dummy_model(feats).squeeze(-1)  # replace the model with dummy_model

                    rankings = multiple_sample_and_log_probability(
                        scores, sample_size, return_prob=False, batch=True)

                    group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
                    indicator_disparity = GroupFairnessLoss.compute_multiple_group_disparity(rankings, rel,
                                                                                             group_identities,
                                                                                             group0_merit,
                                                                                             group1_merit,
                                                                                             position_bias_vector,
                                                                                             args.disparity_type,
                                                                                             noise=args.noise,
                                                                                             en=args.en).mean(dim=-1)
                    indicator_disparities.append(indicator_disparity)
                indicator_disparities = torch.cat(indicator_disparities, dim=0)
                print("Disparities indicator: {}".format(indicator_disparities.mean().item()))
    #### JK put back the triple quotes

    if args.early_stopping:
        time_since_best = 0
        best_metric = -1e6
        best_model = None
        best_epoch = None

    entropy_list = []
    sum_loss_list = []
    rewards_list = []
    fairness_loss_list = []
    reward_variance_list = []
    train_ndcg_list = []
    train_dcg_list = []
    weight_list = []

    epoch_iterator = range(num_epochs)


    for epoch in epoch_iterator:
        print("Entering training Epoch {}".format(epoch))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        if args.progressbar:
            train_dataloader = tqdm(train_dataloader)

        for batch_id, data in enumerate(train_dataloader):

            feats, rel = data

            group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
            if group_identities.bool().all(1).any().item() or (1-group_identities).bool().all(1).any().item():
                continue
                # skip the iteration if only one group appears

            # Form the cross product between group a ID embedding and the document scores

            if args.embed_groups:
                scores, group_embed = model(feats, group_identities)
                scores= scores.squeeze(-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), group_embed.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            # Concatenate the document scores with group ID and predict N**2 independent QP coefficients using a MLP
            elif args.embed_quadscore:
                score_cross = model(feats, group_identities).squeeze(-1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            else:
                scores = model(feats).squeeze(-1)
                test_dscts = ( 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2) ).repeat(scores.shape[0],1,1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), test_dscts.view(scores.shape[0],1,-1)  ).reshape(scores.shape[0],-1)



            # score_cross is used as the linear objective coefficients for the QP layer
            """
            if not args.allow_unfairness:
                policy_lp = PolicyLP_Plus(N=scores.shape[1], eps = args.quad_reg, position_bias_vector = position_bias_vector)
            else:
                policy_lp = PolicyLP_PlusNeq(N=scores.shape[1], eps = args.quad_reg, position_bias_vector = position_bias_vector, delta=args.fairness_gap)

            if args.gpu:
                policy_lp = policy_lp.cuda()
            p_mat = policy_lp(score_cross, group_identities)
            p_mat = relu( p_mat )
            """





            ####################
            ########
            ### NEW SPO work JK

            # Note this is for one sample
            test_dscts = ( 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2) ).repeat(scores.shape[0],1,1)
            #true_costs = torch.bmm( rel.view(scores.shape[0],-1,1), test_dscts.view(scores.shape[0],1,-1)).view(scores.shape[0],-1)
            true_costs = torch.bmm( rel.view(scores.shape[0],-1,1), test_dscts.view(scores.shape[0],1,-1)).view(scores.shape[0],1,-1)

            """
            print('true_costs.size() = ')
            print( true_costs.size()    )
            print('score_cross.size() = ')
            print( score_cross.size()    )
            print('group_identities.size() = ')
            print( group_identities.size()    )
            """

            print("scores.max() = ")
            print( scores.max() )
            print("scores.min() = ")
            print( scores.min() )


            grad = []
            p_mat = []
            regrets = []
            with torch.no_grad():
                dcg_max = compute_dcg_max(rel)  # redundant, defined again below
                for i in range(scores.shape[0]):
                    spo_group_ids = group_identities[i]
                    V_true = true_costs[i].squeeze() #compute 'true' cost coefficients here

                    sol_true = grb_solve(args.list_len, V_true.detach().numpy(), spo_group_ids)

                    #P_spo = grb_solve(args.list_len, spo_coeffs, spo_group_ids)
                    # the true-shifted predictions

                    #V_pred = torch.nn.Sigmoid()( score_cross[i] )
                    #V_pred = score_cross[i] / torch.abs(score_cross[i]).sum() * V_true.sum()
                    V_pred = score_cross[i].squeeze()

                    sol_pred = grb_solve(args.list_len, V_pred.detach().numpy(), spo_group_ids)
                    p_mat.append(torch.Tensor(sol_pred.view(args.list_len,args.list_len)))

                    V_spo  = (2*V_pred - V_true)
                    sol_spo  = grb_solve(args.list_len, V_spo.detach().numpy(), spo_group_ids)

                    #reg = sum((sol_true - sol_pred)*V_true)

                    reg = torch.dot(V_true,(sol_true - sol_pred))
                    regrets.append(reg)

                    use_reg = True  #False
                    if use_reg:

                        #print(torch.Tensor(sol_spo - sol_true))
                        #input("torch.Tensor(sol_spo - sol_true) ^")
                        grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                        #grad.append( torch.Tensor(sol_spo - sol_true)*reg  )
                    else:
                        grad.append( torch.Tensor(sol_spo - sol_true)  )

                    #reg = sum((sol_true - sol_pred)*V_true)
                    #grad = reg*grad




                #grad /= float(scores.shape[0])
                p_mat = torch.stack(p_mat)
                regrets = torch.stack(regrets)

            optimizer.zero_grad()

            #loss.backward(gradient=grad)
            score_cross.backward(gradient=torch.stack(grad))
            #score_cross.backward(gradient=-torch.stack(grad)) # JK NOTE! - negative regret



            #loss = score_cross
            #loss.backward(gradient=torch.stack(grad))   # JK look up - does this take incoming or outgoing


            optimizer.step()




            with torch.no_grad():
                dcg_max = compute_dcg_max(rel)
                test_dscts = ( 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2) ).repeat(scores.shape[0],1,1)
                if args.gpu:
                    test_dscts = test_dscts.cuda()
                #v_unsq = v.unsqueeze(1)
                #f_unsq = f.unsqueeze(1).permute(0,2,1)
                #vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
                loss_a = torch.bmm( p_mat, test_dscts.view(scores.shape[0],-1,1) )
                loss_b = torch.bmm( rel.view(scores.shape[0],1,-1), loss_a ).squeeze()
                loss_norm = loss_b.squeeze() / dcg_max
                loss = loss_norm.mean()

                """ use this to check the 'true costs' against evalutation reward function
                true_costs = torch.bmm( rel.view(scores.shape[0],-1,1),  test_dscts.view(scores.shape[0],1,-1)   ).view(scores.shape[0],1,-1)
                loss2_b = torch.bmm(true_costs, p_mat.view(scores.shape[0],-1).unsqueeze(1).permute(0,2,1))
                loss2_norm = loss2_b.squeeze() / dcg_max
                loss2 = -loss2_norm.mean()
                """






            #### END NEW SPO work
            ########
            ####################



            # log the reward/dcg variance
            sum_loss_list.append(loss.item())
            print("loss = {}".format(loss.item()))
            print("regret = {}".format(regrets.mean().item()))

            step = epoch * len_train_set + batch_id  # JK added here
            if step % args.evaluate_interval == 0 and step > 0:
                print(
                    "Evaluating on validation set: iteration {}/{} of epoch {}".
                        format(batch_id, len_train_set, epoch))


                # JK do the custom test routine for this policy type
                results = evaluate_soft_model(
                            model,
                            #data_reader,
                            test_data_reader,   # JK switch from eval on train to test data
                            deterministic=args.validation_deterministic,
                            fairness_evaluation=fairness_evaluation,
                            num_sample_per_query=args.sample_size,
                            # position_bias_vector=1. / np.log2(2 + np.arange(200)),
                            position_bias_vector=position_bias_vector,
                            group_fairness_evaluation=group_fairness_evaluation,
                            track_other_disparities=args.track_other_disparities,
                            args=args)

                test_DSM_ndcg_list_JK.append(results["DSM_ndcg"])
                test_ndcg_list_JK.append(results["ndcg"])      # JK evaluation.py line 504 for origin of these
                test_dcg_list_JK.append(results["ndcg"])
                test_rank_list_JK.append(results["avg_rank"])
                if group_fairness_evaluation:
                    test_group_expos_disp_list_JK.append(results["avg_group_disparity"])
                    test_group_asym_disp_list_JK.append(results["avg_group_asym_disparity"])
                # JK end test metric collection



                if args.early_stopping:
                    if best_model is None or curr_metric > best_metric + abs(best_metric) * 0.0001:
                        best_metric = curr_metric
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch
                        time_since_best = 0
                    else:
                        time_since_best += 1
                    if time_since_best >= 3:
                        entropy_regularizer = args.entreg_decay * entropy_regularizer
                        print("Decay entropy regularizer to {}".format(entropy_regularizer))
                    if time_since_best >= args.stop_patience:
                        print(
                            "Validation set metric hasn't increased in 10 steps. Exiting")
                        return best_model, best_metric


    # JK are all these metric defined yet?
    outs = {}
    outs["test_ndcg_list_JK"] = test_ndcg_list_JK
    outs["test_dcg_list_JK"] = test_dcg_list_JK
    outs["test_rank_list_JK"] = test_rank_list_JK
    outs["test_group_expos_disp_list_JK"] = test_group_expos_disp_list_JK
    outs["test_group_asym_disp_list_JK"] = test_group_asym_disp_list_JK
    pickle.dump( outs, open('./plots_out/'+ "LP_" + "fultr_benchmarks1_" + 'plots_out' + "_run_" + str(args.index) + '.p', 'wb') )


    csv_outs = {}
    csv_outs['test_ndcg_final']  =  test_ndcg_list_JK[:-1]
    csv_outs["test_dcg_final"] =  test_dcg_list_JK[:-1]
    csv_outs["test_rank_final"]  =  test_rank_list_JK[:-1]
    csv_outs["test_group_expos_disp_final"] =  test_group_expos_disp_list_JK[:-1]
    csv_outs["test_group_asym_disp_final"] = test_group_asym_disp_list_JK[:-1]

    csv_outs["index"] = args.index
    csv_outs["epochs"] = args.epochs
    csv_outs["lr"] = args.lr
    csv_outs["hidden_layer"] = args.hidden_layer
    csv_outs["optimizer"] = args.optimizer
    csv_outs["quad_reg"] = args.quad_reg
    csv_outs["partial_train_data"] = args.partial_train_data
    csv_outs["partial_val_data"] = args.partial_val_data
    csv_outs["full_test_data"] = args.full_test_data
    csv_outs["log_dir"] = args.log_dir
    csv_outs["sample_size"] = args.sample_size
    csv_outs["batch_size"] = args.batch_size
    csv_outs["soft_train"] = args.soft_train
    csv_outs["disparity_type"] = args.disparity_type
    csv_outs["embed_groups"] = args.embed_groups
    csv_outs["embed_quadscore"] = args.embed_quadscore
    csv_outs["allow_unfairness"] = args.allow_unfairness
    csv_outs["fairness_gap"] = args.fairness_gap
    csv_outs["index"] = args.index

    csv_outs = {k:[v] for (k,v) in csv_outs.items()   }
    df_outs = pd.DataFrame.from_dict(csv_outs)
    outPathCsv = './csv/'+ "LP_" + "fultr_benchmarks2_" + 'csv_out' + "_run_" + str(args.index) + ".csv"
    df_outs.to_csv(outPathCsv)


    # Epoch Complete
    print("test_ndcg_list_JK = ")
    print( test_ndcg_list_JK )
    print("test_dcg_list_JK = ")
    print( test_dcg_list_JK )
    print("test_rank_list_JK = ")
    print( test_rank_list_JK )
    print("test_group_expos_disp_list_JK = ")
    print( test_group_expos_disp_list_JK )
    print("test_group_asym_disp_list_JK = ")
    print( test_group_asym_disp_list_JK )
    quit()  # JK this is a hack to escape without crashing; curr_metric below is undefined. We have to return something to the main routine.
    return model, curr_metric
