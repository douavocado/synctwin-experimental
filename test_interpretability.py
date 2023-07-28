# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:27:54 2023

@author: xusem
"""

# generating interpolated data, seeing whether coefficients agree/ are of same sign
# after passing through latent space

# first load data
import torch
import random
import numpy as np

import SyncTwin

from util import io_utils, train_utils

def jaccard_set(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def precision(true_l, estimated_l):
    if len(estimated_l) == 0:
        return float(len(true_l) == 0)
    return len([i for i in estimated_l if i in true_l])/len(estimated_l)

def recall(true_l, estimated_l):
    if len(true_l) == 0:
        return float(len(estimated_l) == 0)
    return len([i for i in true_l if i in estimated_l])/len(true_l)

def f1(true_l, estimated_l):
    p = precision(true_l,estimated_l)
    r = recall(true_l, estimated_l)
    if p+r == 0:
        return 0
    else:
        return p*r/(p+r)

seed = 100
pretrain_Y = True
itr = 1
itr_pretrain = 5000
itr_fine_tune = 20000
batch_size = 100
sim_id = 102
reduced_fine_tune = True
linear_decoder = True
lam_prognostic = 1.0
lam_recon = 50.0
tau = 1.0
regular = True
# if not regular:
#     n_hidden = n_hidden * 2
n_hidden=20

results_file = 'interpretability_results_low_sample.txt'

#loading config and data
base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
data_path = base_path_data + "/{}-{}.{}"

print("loading test data")
n_units, n_treated, n_units_total, step, train_step, control_sample, noise, n_basis, n_cluster = io_utils.load_config(
    data_path, "test"
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
    y_pre_full
) = io_utils.load_tensor(data_path, "test")




simulated_itr = 1000

# small_sample_size = 20
# extrapolate = True
# robustness = 0
# use_lasso = True
# helper = False
for extrapolate in [False, True]:
    for small_sample_size in [5,10,15]:
        for use_lasso in [True]:
            for helper in [True, False]:
                for robustness in [0,1,2]:                
                    parameter_str = 'sample_size: ' + str(small_sample_size)+ ', extrapolate: ' + str(extrapolate) + ', ' + 'robust: ' + str(robustness) + ', lasso: ' + str(use_lasso) + ', helper: ' + str(helper) + '\n'
                    print('Currently evaluating '+ parameter_str)    
                    # load model
                    if helper:
                        helper_str = '_helper'
                    else:
                        helper_str = ''
                    
                    if use_lasso:
                        lasso_str = '_lasso'
                    else:
                        lasso_str = ''
                        
                    base_path_model = "models/{}-seed-".format(sim_id) + str(seed) + '-robust_' + str(robustness) + lasso_str + helper_str
                    model_path = base_path_model + "/itr-" + str(0) + "-{}.pth"
                    if regular:
                        enc1 = SyncTwin.RegularEncoder(input_dim=2, hidden_dim=n_hidden)
                        dec1 = SyncTwin.RegularDecoder(hidden_dim=enc1.hidden_dim, output_dim=enc1.input_dim, max_seq_len=train_step)
                    else:
                        enc1 = SyncTwin.GRUDEncoder(input_dim=2, hidden_dim=n_hidden)
                        dec1 = SyncTwin.LSTMTimeDecoder(hidden_dim=enc1.hidden_dim, output_dim=enc1.input_dim, max_seq_len=train_step)
                    
                    if pretrain_Y:
                        if not linear_decoder:
                            dec_Y1 = SyncTwin.RegularDecoder(
                                hidden_dim=enc1.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
                            )
                            pre_dec_Y1 = SyncTwin.RegularDecoder(
                                hidden_dim=enc1.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
                            )
                        else:
                            dec_Y1 = SyncTwin.LinearDecoder(
                                hidden_dim=enc1.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
                            )
                            pre_dec_Y1 = SyncTwin.LinearDecoder(
                                hidden_dim=enc1.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
                            )
                    else:
                        dec_Y1 = None
                        pre_dec_Y1 = None
                    
                    
                    
                    nsc = SyncTwin.SyncTwin(
                        small_sample_size,
                        small_sample_size+1,
                        reg_B=0.0,
                        lam_express=1.0,
                        lam_recon=lam_recon,
                        lam_prognostic=lam_prognostic,
                        tau=tau,
                        encoder=enc1,
                        decoder=dec1,
                        decoder_Y=dec_Y1,
                        pre_decoder_Y=pre_dec_Y1,
                        use_lasso=use_lasso,
                    )
                    
                    scores = []
                    precisions = []
                    recalls = []
                    
                    
                    for j in range(simulated_itr):
                        print('iteration:', j+1)
                        # modifying to a smaller set randomly
                        
                        small_control_i = random.sample(range(n_units), small_sample_size)
                        small_treat_i = random.sample(range(n_units, n_units_total), small_sample_size)
                        
                        x_sub = torch.zeros((x_full.shape[0],small_sample_size*2, x_full.shape[-1]))
                        t_sub = torch.zeros((t_full.shape[0],small_sample_size*2, t_full.shape[-1]))
                        y_sub = torch.zeros((y_full.shape[0],small_sample_size*2, y_full.shape[-1]))
                        mask_sub = torch.ones((x_full.shape[0],small_sample_size*2, x_full.shape[-1]))
                        batch_ind_sub = torch.arange(small_sample_size*2)
                        y_control_sub = torch.zeros((y_full.shape[0],small_sample_size, y_full.shape[-1]))
                        y_mask_sub = torch.concat([torch.ones(n_units), torch.zeros(n_treated)])
                        y_pre_sub = torch.zeros((y_pre_full.shape[0],small_sample_size*2, y_pre_full.shape[-1]))
                        
                        # setting values
                        for i in range(small_sample_size):
                            x_sub[:,i,:] = x_full[:,small_control_i[i],:]
                            t_sub[:,i,:] = t_full[:,small_control_i[i],:]
                            y_sub[:,i,:] = y_full[:,small_control_i[i],:]
                            y_control_sub[:,i,:] = y_full[:,small_control_i[i],:]
                            y_pre_sub[:,i,:] = y_pre_full[:,small_control_i[i],:]
                        
                        for i in range(small_sample_size, small_sample_size*2):
                            x_sub[:,i,:] = x_full[:,small_treat_i[i-small_sample_size],:]
                            t_sub[:,i,:] = t_full[:,small_treat_i[i-small_sample_size],:]
                            y_sub[:,i,:] = y_full[:,small_treat_i[i-small_sample_size],:]
                            y_pre_sub[:,i,:] = y_pre_full[:,small_treat_i[i-small_sample_size],:]
                        
                        train_utils.load_nsc(nsc, x_sub, t_sub, mask_sub, batch_ind_sub, model_path=model_path, load_B=False)
                        train_utils.load_pre_train_and_init(
                            nsc, x_sub, t_sub, mask_sub, batch_ind_sub, model_path=model_path, init_decoder_Y=pretrain_Y, init_pre_dec_Y=robustness==2,
                        )
                        
                        # now generate random interpolated samples of the data and add them to the data
                        # we can also choose to test on extrapolated coefficients too
                        sample_size = random.randint(2,4)
                        sampled_i = random.sample(range(small_sample_size), sample_size)
                        if not extrapolate:
                            coefficients = torch.rand(sample_size)
                        else:
                            coefficients = torch.rand(sample_size)-0.3
                            #clipping so coefficients don't become too large
                            while torch.sum(coefficients) < 0.2:
                                coefficients = torch.rand(sample_size)-0.3
                        coefficients = coefficients/torch.sum(coefficients)
                        interpolated = torch.zeros((x_sub.shape[0],x_sub.shape[-1]))
                        for i in range(sample_size):
                            interpolated += coefficients[i]*x_sub[:,sampled_i[i],:]
                        
                        interpolated = interpolated.unsqueeze(1)
                        
                        new_x_sub = torch.concat([x_sub, interpolated], dim=1)
                        new_batch_ind_sub = torch.concat([batch_ind_sub, torch.tensor([batch_ind_sub.shape[0]])])
                        
                        # now calculate weighting vector from new data
                        if not use_lasso:
                            return_code = train_utils.train_B_self_expressive(
                                nsc, new_x_sub, t_sub, mask_sub, new_batch_ind_sub, niters=itr_fine_tune, model_path=model_path, verbose=False,
                            )
                            contributors = torch.where(nsc.get_B_reduced(new_batch_ind_sub)[-1] > 0.05)[0].tolist()
                        else:
                            # we use the lasso method with does not place restrictions on coefficients
                            return_code = train_utils.train_B_self_expressive_lasso(
                                nsc, new_x_sub, t_sub, mask_sub, new_batch_ind_sub, niters=itr_fine_tune, alpha=0.01, calculate_loss=False,
                            )
                            nsc.set_synthetic_control(return_code[1])
                            if not extrapolate:
                                contributors = torch.where(return_code[1][-1] > 0.01)[0].tolist()
                            else:
                                contributors = torch.where(torch.abs(return_code[1][-1]) > 0.02)[0].tolist()
                        # print('identified contributors', contributors)
                        # print('contributors', sampled_i)
                        #sim_score = jaccard_set(contributors, sampled_i)
                        p = precision(sampled_i, contributors)
                        r = recall(sampled_i, contributors)
                        sim_score = f1(sampled_i, contributors)
                        # print('similarity:', sim_score)
                        scores.append(sim_score)
                        precisions.append(p)
                        recalls.append(r)
                        
                        # test inconsistency
                        
                    
                    avg_score = sum(scores)/simulated_itr
                    avg_p = sum(precisions)/simulated_itr
                    avg_r = sum(recalls)/simulated_itr
                    stdev_score = np.std(scores) / np.sqrt(simulated_itr)
                    stdev_p = np.std(precisions) / np.sqrt(simulated_itr)
                    stdev_r = np.std(recalls) / np.sqrt(simulated_itr)
                    
                    # print('average score', sum(scores)/simulated_itr)
                    results_str = ''
                    results_str += 'f1 score: '+ str(avg_score) + ' (' + str(stdev_score) + ') \n'
                    results_str += 'precision: '+ str(avg_p) + ' (' + str(stdev_p) + ') \n'
                    results_str += 'recall: '+ str(avg_r) + ' (' + str(stdev_r) + ') \n'
                    
                    write_f = open(results_file, 'a')
                    write_f.write(parameter_str + results_str)
                    write_f.close()