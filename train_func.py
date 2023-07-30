# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 14:24:59 2023

@author: xusem
"""
import numpy as np
import torch
import random
import time

import SyncTwin

from util import io_utils, train_utils, eval_utils

def run_one(sim_id, use_lasso=True, itr_fine_tune=20000, itr_pretrain=500, use_helper=False, robustness=0, n_hidden=10, rand_seed=100):
    itr=1
    regular=True
    pretrain_Y=True
    linear_decoder=True
    print("Running simulation with seed {}".format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    
    base_path_data = "data/{}-seed-".format(sim_id) + '100'
    data_path = base_path_data + "/{}-{}.{}"
    
    # loading config and data
    print("loading data")
    n_units, n_treated, n_units_total, step, train_step, control_sample, noise, n_basis, n_cluster = io_utils.load_config(
        data_path, "train"
    )
    (
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        m,
        sd,
        treatment_effect,
        y_pre_full,
    ) = io_utils.load_tensor(data_path, "train")
    (
        x_full_val,
        t_full_val,
        mask_full_val,
        batch_ind_full_val,
        y_full_val,
        y_control_val,
        y_mask_full_val,
        _,
        _,
        _,
        y_pre_full_val,
    ) = io_utils.load_tensor(data_path, "val")
    
    training_time_list = []
    
    
    for i in range(itr):
        print("Iteration {}".format(i))
        start_time = time.time()
    
        if regular:
            enc = SyncTwin.RegularEncoder(input_dim=x_full.shape[-1], hidden_dim=n_hidden)
            dec = SyncTwin.RegularDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
        else:
            enc = SyncTwin.GRUDEncoder(input_dim=x_full.shape[-1], hidden_dim=n_hidden)
            dec = SyncTwin.LSTMTimeDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
    
        if pretrain_Y:
            if not linear_decoder:
                dec_Y = SyncTwin.RegularDecoder(
                    hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
                )
                pre_dec_Y = SyncTwin.RegularDecoder(
                    hidden_dim=enc.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
                )
            else:
                dec_Y = SyncTwin.LinearDecoder(
                    hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
                )
                pre_dec_Y = SyncTwin.LinearDecoder(
                    hidden_dim=enc.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
                )
        else:
            dec_Y = None
            pre_dec_Y = None
    
        if use_helper:
            if regular:
                linear_helper = SyncTwin.LinearHelper(x_full.shape[-1]*x_full.shape[0], 2*n_hidden)
            else:
                linear_helper = SyncTwin.LinearHelper(x_full.shape[-1]*x_full.shape[0], n_hidden)
    
        nsc = SyncTwin.SyncTwin(
            n_units,
            n_treated,
            reg_B=0.0,
            lam_express=1.0,
            lam_recon=50,
            lam_prognostic=1,
            tau=1,
            encoder=enc,
            decoder=dec,
            decoder_Y=dec_Y,
            pre_decoder_Y=pre_dec_Y,
            use_lasso=use_lasso,
        )
    
        print("Pretrain")
        if not use_helper:
            train_utils.pre_train_reconstruction_prognostic_loss(
                nsc,
                x_full,
                t_full,
                mask_full,
                y_full,
                y_mask_full,
                x_full_val,
                t_full_val,
                mask_full_val,
                y_full_val,
                y_mask_full_val,
                y_pre_full,
                y_pre_full_val,
                niters=itr_pretrain,
                batch_size=100,
                robust=robustness,
            )
        else:
            
            train_utils.pre_train_reconstruction_prognostic_loss_linear_helper(
                nsc,
                x_full,
                t_full,
                mask_full,
                y_full,
                y_mask_full,
                x_full_val,
                t_full_val,
                mask_full_val,
                y_full_val,
                y_mask_full_val,
                y_pre_full,
                y_pre_full_val,
                niters=itr_pretrain,
                batch_size=100,
                robust=robustness,
                linear_helper=linear_helper,
                lam_helper=40
            )
    
    
        end_time = time.time()
        training_time = end_time - start_time
        training_time_list.append(training_time)
        print("--- Training done in %s seconds ---" % training_time)
    
        #train_utils.load_nsc(nsc, x_full_val, t_full_val, mask_full_val, batch_ind_full_val, model_path=model_path)
    
    print("Testing")
    
   
    
    (
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        m,
        sd,
        treatment_effect,
        y_pre_full
    ) = io_utils.load_tensor(data_path, "test")
    
    
    if not use_lasso:
        return_code = train_utils.train_B_self_expressive(
            nsc, x_full, t_full, mask_full, batch_ind_full, niters=itr_fine_tune,
        )
    else:
        # we use the lasso method with does not place restrictions on coefficients
        return_code = train_utils.train_B_self_expressive_lasso(
            nsc, x_full, t_full, mask_full, batch_ind_full, niters=itr_fine_tune,
        )
        nsc.set_synthetic_control(return_code[1])
         
          
    
    # evaluating best model
    effect_est, y_hat = eval_utils.get_treatment_effect(nsc, batch_ind_full, y_full, y_control)
    mae_effect = torch.mean(torch.abs(treatment_effect - effect_est.cpu())).item()
    mae_sd = torch.std(torch.abs(treatment_effect - effect_est.cpu())).item() / np.sqrt(n_treated)

    print("Treatment effect MAE: ({}, {})".format(mae_effect, mae_sd))
    return mae_effect

iterations = 50
synctwin_sc = []
ls_score = []
lss_score = []

for i in range(iterations):
    print('Round number', i+1)
    seed = random.randint(1,100000)
    orig = run_one(208, rand_seed=seed)
    ls = run_one(208, robustness=1, rand_seed=seed)
    lss = run_one(208, robustness=2, rand_seed=seed)
    
    synctwin_sc.append(orig)
    ls_score.append(ls)
    lss_score.append(lss)

sync_score = np.array(synctwin_sc)
ls_score = np.array(ls_score)
lss_score = np.array(lss_score)

print('sync vs ls', np.sum(sync_score < ls_score), '/', len(ls_score))
print('sync vs lss', np.sum(sync_score < lss_score), '/', len(ls_score))
print('lss vs ls', np.sum(lss_score < ls_score), '/', len(ls_score))