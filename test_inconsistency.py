#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:25:09 2023

@author: james
"""

# Quantifying the discrepancies between differently trained on counterfactuals
# provided by the Synctwin paper

import numpy as np
import numpy.random
from scipy.spatial import procrustes

import SyncTwin

# from config import DEVICE
from util import io_utils, train_utils

seed = 100
pretrain_Y = True
itr = 1
itr_pretrain = 3000
itr_fine_tune = 20000
batch_size = 100
sim_id = 101
reduced_fine_tune = True
linear_decoder = True
lam_prognostic = 1.0
lam_recon = 100.0
tau = 1.0
regular = False
# if not regular:
#     n_hidden = n_hidden * 2
robustness = 0
use_lasso = False

#loading config and data
base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
data_path = base_path_data + "/{}-{}.{}"

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
        


# for n_hidden in [10,20,40,160]:
#     for use_treated in [True, False]:
#         if use_treated == True:
#             model_id = 'train_treated-' + str(n_hidden)
#         else:
#             model_id = 'train_control-' + str(n_hidden)
        
        
#         print("Running simulation with seed {}, for n_hidden {} and use treated {}".format(seed, n_hidden, use_treated))
#         numpy.random.seed(seed)
#         torch.manual_seed(seed)
        
#         base_path_model = "models/{}-seed-".format(sim_id) + str(seed) + model_id
#         base_path_plot = "plots/{}-seed-".format(sim_id) + str(seed) + model_id
#         
#         train_utils.create_paths(base_path_model, base_path_plot, base_path_data)
#         plot_path = base_path_plot + "/unit-{}-dim-{}-{}.png"
#         
        
#         
#         control_error_list = []
#         training_time_list = []
        
        
#         # TRAINING ######################################################################
        
#         for i in range(itr):
#             print("Iteration {}".format(i))
#             start_time = time.time()
        
#             model_path = base_path_model + "/itr-" + str(i) + "-{}.pth"
        
#             if regular:
#                 enc = SyncTwin.RegularEncoder(input_dim=3, hidden_dim=n_hidden)
#                 dec = SyncTwin.RegularDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
#             else:
#                 enc = SyncTwin.GRUDEncoder(input_dim=3, hidden_dim=n_hidden)
#                 dec = SyncTwin.LSTMTimeDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
        
#             if pretrain_Y:
#                 if not linear_decoder:
#                     dec_Y = SyncTwin.RegularDecoder(
#                         hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
#                     )
#                     pre_dec_Y = SyncTwin.RegularDecoder(
#                         hidden_dim=enc.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
#                     )
#                 else:
#                     dec_Y = SyncTwin.LinearDecoder(
#                         hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
#                     )
#                     pre_dec_Y = SyncTwin.LinearDecoder(
#                         hidden_dim=enc.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
#                     )
#             else:
#                 dec_Y = None
#                 pre_dec_Y = None
        
#             nsc = SyncTwin.SyncTwin(
#                 n_units,
#                 n_treated,
#                 reg_B=0.0,
#                 lam_express=1.0,
#                 lam_recon=lam_recon,
#                 lam_prognostic=lam_prognostic,
#                 tau=tau,
#                 encoder=enc,
#                 decoder=dec,
#                 decoder_Y=dec_Y,
#                 pre_decoder_Y=pre_dec_Y,
#                 use_lasso=use_lasso,
#             )
        
#             print("Pretrain")
#             if not pretrain_Y:
#                 train_utils.pre_train_reconstruction_loss(
#                     nsc,
#                     x_full,
#                     t_full,
#                     mask_full,
#                     x_full_val,
#                     t_full_val,
#                     mask_full_val,
#                     niters=itr_pretrain,
#                     model_path=model_path,
#                     batch_size=batch_size,
#                 )
#             else:
#                 train_utils.pre_train_reconstruction_prognostic_loss(
#                     nsc,
#                     x_full,
#                     t_full,
#                     mask_full,
#                     y_full,
#                     y_mask_full,
#                     x_full_val,
#                     t_full_val,
#                     mask_full_val,
#                     y_full_val,
#                     y_mask_full_val,
#                     y_pre_full,
#                     y_pre_full_val,
#                     niters=itr_pretrain,
#                     model_path=model_path,
#                     batch_size=batch_size,
#                     robust=robustness,
#                     use_treated=use_treated
#                 )
        
#             if not reduced_fine_tune:
#                 return_code = train_utils.train_all_losses(
#                     nsc,
#                     x_full,
#                     t_full,
#                     mask_full,
#                     batch_ind_full,
#                     y_full,
#                     y_control,
#                     y_mask_full,
#                     niters=itr_pretrain,
#                     model_path=model_path,
#                     batch_size=None,
#                 )
#             torch.save(nsc.state_dict(), model_path.format("nsc.pth"))

# Comparison of different model's C_i vectors ##################################

n_hidden=10

base_path_model1 = "models/{}-seed-".format(sim_id) + str(seed) + 'train_control-' + str(n_hidden)
base_path_model2 = "models/{}-seed-".format(sim_id) + str(seed) + 'train_treated-' + str(n_hidden)
model_path1 = base_path_model1 + "/itr-" + str(0) + "-{}.pth"
model_path2 = base_path_model2 + "/itr-" + str(0) + "-{}.pth"

# loading models
if regular:
    enc1 = SyncTwin.RegularEncoder(input_dim=3, hidden_dim=n_hidden)
    dec1 = SyncTwin.RegularDecoder(hidden_dim=enc1.hidden_dim, output_dim=enc1.input_dim, max_seq_len=train_step)
else:
    enc1 = SyncTwin.GRUDEncoder(input_dim=3, hidden_dim=n_hidden)
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
nsc1 = SyncTwin.SyncTwin(
    n_units,
    n_treated,
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

if regular:
    enc2 = SyncTwin.RegularEncoder(input_dim=3, hidden_dim=n_hidden)
    dec2 = SyncTwin.RegularDecoder(hidden_dim=enc2.hidden_dim, output_dim=enc2.input_dim, max_seq_len=train_step)
else:
    enc2 = SyncTwin.GRUDEncoder(input_dim=3, hidden_dim=n_hidden)
    dec2 = SyncTwin.LSTMTimeDecoder(hidden_dim=enc2.hidden_dim, output_dim=enc2.input_dim, max_seq_len=train_step)

if pretrain_Y:
    if not linear_decoder:
        dec_Y2 = SyncTwin.RegularDecoder(
            hidden_dim=enc2.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
        )
        pre_dec_Y2 = SyncTwin.RegularDecoder(
            hidden_dim=enc2.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
        )
    else:
        dec_Y2 = SyncTwin.LinearDecoder(
            hidden_dim=enc2.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
        )
        pre_dec_Y2 = SyncTwin.LinearDecoder(
            hidden_dim=enc2.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
        )
else:
    dec_Y1 = None
    pre_dec_Y1 = None

nsc2 = SyncTwin.SyncTwin(
    n_units,
    n_treated,
    reg_B=0.0,
    lam_express=1.0,
    lam_recon=lam_recon,
    lam_prognostic=lam_prognostic,
    tau=tau,
    encoder=enc2,
    decoder=dec2,
    decoder_Y=dec_Y2,
    pre_decoder_Y=pre_dec_Y2,
    use_lasso=use_lasso,
)
train_utils.load_nsc(nsc1, x_full, t_full, mask_full, batch_ind_full, model_path=model_path1)
train_utils.load_pre_train_and_init(
    nsc1, x_full, t_full, mask_full, batch_ind_full, model_path=model_path1, init_decoder_Y=pretrain_Y, init_pre_dec_Y=robustness==2,
)
train_utils.load_nsc(nsc2, x_full, t_full, mask_full, batch_ind_full, model_path=model_path2)
train_utils.load_pre_train_and_init(
    nsc2, x_full, t_full, mask_full, batch_ind_full, model_path=model_path2, init_decoder_Y=pretrain_Y, init_pre_dec_Y=robustness==2,
)

C1 = nsc1.get_representation(x_full, t_full, mask_full).cpu().detach().numpy()
C2 = nsc2.get_representation(x_full, t_full, mask_full).cpu().detach().numpy()

_,_,similarity = procrustes(C1,C2)

print('procrustes normalised distance:', similarity/np.sum(np.ones_like(C1)))