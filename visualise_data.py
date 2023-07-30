# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 13:01:09 2023

@author: xusem
"""
from util import io_utils, train_utils
import matplotlib.pyplot as plt
import numpy as np
import SyncTwin
import torch

sim_id = 103
seed = 100

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

#load model for inference
pretrain_Y = True
itr = 1
itr_pretrain = 5000
itr_fine_tune = 20000
batch_size = 100
reduced_fine_tune = True
linear_decoder = True
lam_prognostic = 1.0
lam_recon = 50.0
tau = 1.0
regular = True
use_lasso = True
robustness = 2
# if not regular:
#     n_hidden = n_hidden * 2
n_hidden=20
    
base_path_model1 = "models/{}-seed-".format(sim_id) + str(seed) + '-interptwin'
model_path1 = base_path_model1 + "/itr-" + str(0) + "-{}.pth"

base_path_model2 = "models/{}-seed-".format(sim_id) + str(seed) + '-synctwin'
model_path2 = base_path_model2 + "/itr-" + str(0) + "-{}.pth"


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


train_utils.load_nsc(nsc1, x_full, t_full, mask_full, batch_ind_full, model_path=model_path1)
train_utils.load_pre_train_and_init(
    nsc1, x_full, t_full, mask_full, batch_ind_full, model_path=model_path1, init_decoder_Y=pretrain_Y, init_pre_dec_Y=robustness==2,
)

if not use_lasso:
    return_code = train_utils.train_B_self_expressive(
        nsc1, x_full, t_full, mask_full, batch_ind_full, niters=itr_fine_tune, model_path=model_path1, verbose=False,
    )
    b_reduced = nsc1.get_B_reduced(batch_ind_full)
    y_hat1 = torch.matmul(b_reduced.to(y_control.device), y_control)
    y_pre_hat1 = torch.matmul(b_reduced.to(y_pre_full.device), y_pre_full[:,:n_units,:])
else:
    # we use the lasso method with does not place restrictions on coefficients
    return_code = train_utils.train_B_self_expressive_lasso(
        nsc1, x_full, t_full, mask_full, batch_ind_full, niters=itr_fine_tune, alpha=0.01, calculate_loss=False,
    )
    nsc1.set_synthetic_control(return_code[1])
    b_reduced = return_code[1]
    y_hat1 = torch.matmul(b_reduced.to(y_control.device), y_control)
    y_pre_hat1 = torch.matmul(b_reduced.to(y_pre_full.device), y_pre_full[:,:n_units,:])


#now get synctwin predictions #########################################
if regular:
    enc2 = SyncTwin.RegularEncoder(input_dim=2, hidden_dim=n_hidden)
    dec2 = SyncTwin.RegularDecoder(hidden_dim=enc1.hidden_dim, output_dim=enc1.input_dim, max_seq_len=train_step)
else:
    enc2 = SyncTwin.GRUDEncoder(input_dim=2, hidden_dim=n_hidden)
    dec2 = SyncTwin.LSTMTimeDecoder(hidden_dim=enc1.hidden_dim, output_dim=enc1.input_dim, max_seq_len=train_step)

if pretrain_Y:
    if not linear_decoder:
        dec_Y2 = SyncTwin.RegularDecoder(
            hidden_dim=enc1.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
        )
        pre_dec_Y2 = SyncTwin.RegularDecoder(
            hidden_dim=enc1.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
        )
    else:
        dec_Y2 = SyncTwin.LinearDecoder(
            hidden_dim=enc1.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
        )
        pre_dec_Y2 = SyncTwin.LinearDecoder(
            hidden_dim=enc1.hidden_dim, output_dim=y_pre_full.shape[-1], max_seq_len=train_step
        )
else:
    dec_Y2 = None
    pre_dec_Y2 = None



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
    use_lasso=False,
)
train_utils.load_nsc(nsc2, x_full, t_full, mask_full, batch_ind_full, model_path=model_path2)
train_utils.load_pre_train_and_init(
    nsc2, x_full, t_full, mask_full, batch_ind_full, model_path=model_path2, init_decoder_Y=pretrain_Y, init_pre_dec_Y=False,
)

return_code = train_utils.train_B_self_expressive(
    nsc2, x_full, t_full, mask_full, batch_ind_full, niters=itr_fine_tune, model_path=model_path2, verbose=False,
)
b_reduced2 = nsc2.get_B_reduced(batch_ind_full)
y_hat2 = torch.matmul(b_reduced2.to(y_control.device), y_control)
y_pre_hat2 = torch.matmul(b_reduced2.to(y_pre_full.device), y_pre_full[:,:n_units,:])



y_full_data = torch.concat([y_pre_full, y_full], dim=0)
y_full_hat1 = torch.concat([y_pre_hat1, y_hat1], dim=0)
y_full_hat2 = torch.concat([y_pre_hat2, y_hat2], dim=0).detach()

treatment_mask = torch.concat([torch.zeros_like(y_control), treatment_effect], dim=1)
#counter_fac = torch.concat([y_pre_full, y_full + treatment_mask], dim=0)
counter_fac = torch.concat([y_pre_full[-1:,:,:], y_full - treatment_mask], axis=0)

all_data = torch.concat([y_full_data, y_full_hat2, y_full_hat1, counter_fac], dim=0)

# plt.ylim(np.min(x_full[:,rand_i,:].numpy()), np.max(x_full[:,rand_i,:].numpy()))
# for j in range(x_full.shape[-1]):
#     plt.plot(np.arange(x_full[:,rand_i,j].shape[0]), x_full[:,rand_i,j])

rand_i = 212
# rand_i = 322
print('index', rand_i)


plt.ylim(-0.5, 2.5)

plt.plot(np.arange(y_full_data[:,rand_i,0].shape[0]), y_full_data[:,rand_i,0], 'b')

plt.plot(np.arange(y_full_hat1[:,rand_i,0].shape[0]), y_full_hat1[:,rand_i,0], 'y')
plt.plot(np.arange(y_full_hat2[:,rand_i,0].shape[0]), y_full_hat2[:,rand_i,0], 'g')

plt.plot(np.arange(train_step-1,step), counter_fac[:,rand_i,0], 'r')

plt.savefig('plot.png')