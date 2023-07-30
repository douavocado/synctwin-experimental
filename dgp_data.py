# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:03:03 2023

@author: xusem
"""

import numpy as np
import torch
import pickle

from util import train_utils


def generate_data(n_control=200, n_treated=200, latent_dim=10, covariate_dim=10, train_steps=25, steps=30, noise=0.1):
    q_pre_matrix = np.random.rand(latent_dim,latent_dim)
    q_post_matrix = np.random.rand(latent_dim,latent_dim)
    x_matrix = np.random.rand(train_steps,1)
    
    covariates = np.random.rand(n_control+n_treated, covariate_dim, train_steps)
    
    #create chebyshev polynomial matrix for the first latent dim number of chebyshev polys
    chebyshev_matrix = np.ones((latent_dim,steps))
    t1 = np.arange(steps)/steps
    chebyshev_matrix[1,:] = t1
    for i in range(2,latent_dim):
        chebyshev_matrix[i,:] = 2*t1*chebyshev_matrix[i-1,:] - chebyshev_matrix[i-2,:]
    
    q_pre = np.matmul(q_pre_matrix, chebyshev_matrix)
    q_post = np.matmul(q_post_matrix, chebyshev_matrix)
    
    q_control = q_pre.copy()
    q_treated = np.concatenate([q_pre[:,:train_steps], q_post[:,train_steps:]], axis=-1)

    
    c_matrix = np.tanh(np.matmul(covariates, x_matrix)[:,:,0])
    
    control_outcomes = torch.from_numpy(np.matmul(c_matrix[:n_control,:], q_control)).unsqueeze(-1)
    treated_outcomes = torch.from_numpy(np.matmul(c_matrix[n_control:,:], q_treated)).unsqueeze(-1)
    treated_cf = torch.from_numpy(np.matmul(c_matrix[n_control:,:], q_control)).unsqueeze(-1)
    
    outcomes_full = torch.concatenate([control_outcomes, treated_outcomes], dim=0)
    outcomes_full =outcomes_full.permute(1, 0,2)
    
    counter_factuals = treated_cf.permute(1,0,2)
    
    covariates = torch.from_numpy(covariates).permute(2,0,1)
    
    
    #add noise
    covariates = covariates + torch.randn_like(covariates) * noise
    
    n_units=n_control
    n_units_total = n_control + n_treated
    x_full = covariates.float()
    t_full = torch.ones_like(x_full).float()
    mask_full = torch.ones_like(x_full).float()
    batch_ind_full = torch.arange(n_units_total)
    y_mask_full = (batch_ind_full < n_units).float() * 1.0
    y_full = outcomes_full[train_steps:,:,:].float() + torch.randn_like(outcomes_full[train_steps:,:,:]).float() * noise
    y_control = y_full[:,:n_units,:].float()
    y_pre_full = outcomes_full[:train_steps,:,:].float() + torch.randn_like(outcomes_full[:train_steps,:,:]).float() * noise
    
    treatment_effect = outcomes_full[train_steps:,n_units:,:] - counter_factuals[train_steps:,:,:]
    treatment_effect = treatment_effect.float()
    return(
    (n_units, n_treated, n_units_total),
    x_full,
    t_full,
    mask_full,
    batch_ind_full,
    y_full,
    y_control,
    y_mask_full,
    treatment_effect,
    y_pre_full,)


n_units = 10
n_treated = 10
n_units_total = n_units + n_treated
step = 30
train_step=25
noise = 1
n_basis = 10
control_sample = n_units
n_cluster = 1
sim_id = 208
seed=100

base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
train_utils.create_paths(base_path_data)
data_path = base_path_data + "/{}-{}.{}"

for fold in ["test", "val", "train"]:
    (n_tuple,
     x_full,
     t_full,
     mask_full,
     batch_ind_full,
     y_full,
     y_control,
     y_mask_full,
     treatment_effect,
     y_pre_full
     ) = generate_data(n_control =n_units, n_treated=n_treated, train_steps=train_step, steps= step, noise=noise)
    # export data to csv
    X0 = x_full[:, :n_units, :]
    X0 = X0.permute((0, 2, 1)).reshape(X0.shape[0] * X0.shape[2], X0.shape[1]).cpu().numpy()
    
    X1 = x_full[:, n_units:, :]
    X1 = X1.permute((0, 2, 1)).reshape(X1.shape[0] * X1.shape[2], X1.shape[1]).cpu().numpy()
    
    Y_control = y_control[:, :, 0].cpu().numpy()
    Y_treated = y_full[:, n_units:, 0].cpu().numpy()
    Treatment_effect = treatment_effect[:, :, 0].cpu().numpy()
    
    np.savetxt(data_path.format(fold, "X0", "csv"), X0, delimiter=",")
    np.savetxt(data_path.format(fold, "X1", "csv"), X1, delimiter=",")
    np.savetxt(data_path.format(fold, "Y_control", "csv"), Y_control, delimiter=",")
    np.savetxt(data_path.format(fold, "Y_treated", "csv"), Y_treated, delimiter=",")
    np.savetxt(data_path.format(fold, "Treatment_effect", "csv"), Treatment_effect, delimiter=",")
    
    torch.save(x_full, data_path.format(fold, "x_full", "pth"))
    torch.save(t_full, data_path.format(fold, "t_full", "pth"))
    torch.save(mask_full, data_path.format(fold, "mask_full", "pth"))
    torch.save(batch_ind_full, data_path.format(fold, "batch_ind_full", "pth"))
    torch.save(y_pre_full, data_path.format(fold, "y_pre_full", "pth"))
    torch.save(y_full, data_path.format(fold, "y_full", "pth"))
    torch.save(y_control, data_path.format(fold, "y_control", "pth"))
    torch.save(y_mask_full, data_path.format(fold, "y_mask_full", "pth"))
    m = torch.rand(3)
    sd = torch.rand(3)
    torch.save(m, data_path.format(fold, "m", "pth"))
    torch.save(sd, data_path.format(fold, "sd", "pth"))
    torch.save(treatment_effect, data_path.format(fold, "treatment_effect", "pth"))
    
    config = {
        "n_units": n_units,
        "n_treated": n_treated,
        "n_units_total": n_units_total,
        "step": step,
        "train_step": train_step,
        "control_sample": control_sample,
        "noise": noise,
        "n_basis": n_basis,
        "n_cluster": n_cluster,
    }
    with open(data_path.format(fold, "config", "pkl"), "wb") as f:
        pickle.dump(config, file=f)