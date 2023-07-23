import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline

import GRUD
from config import D_TYPE, DEVICE

WRAP_INDEX_ON = True  # To run old code set WRAP_INDEX_ON to True, otherwise keep this as False.


# former NSC
class SyncTwin(nn.Module):
    def __init__(
        self,
        n_unit,
        n_treated,
        reg_B=0.0,
        lam_express=1.0,
        lam_recon=0.0,
        lam_prognostic=0.0,
        tau=1.0,
        encoder=None,
        decoder=None,
        decoder_Y=None,
        device=DEVICE,
        dtype=D_TYPE,
        reduce_gpu_memory=False,
        inference_only=False,
        use_lasso=False,
        pre_decoder_Y=None,
    ):
        super(SyncTwin, self).__init__()
        assert not (reduce_gpu_memory and inference_only)

        self.n_unit = n_unit
        self.n_treated = n_treated
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        if decoder_Y is not None:
            self.decoder_Y = decoder_Y.to(device)
        self.device = device
        if reduce_gpu_memory:
            init_B = (torch.ones(1, 1, dtype=dtype)).to(device) * 1.0e-4
        elif inference_only:
            init_B = (torch.ones(n_treated, n_unit, dtype=dtype)).to(device) * 1.0e-4
        else:
            init_B = (torch.ones(n_unit + n_treated, n_unit, dtype=dtype)).to(device) * 1.0e-4
        self.B = nn.Parameter(init_B)
        self.C0 = torch.zeros(n_unit, self.encoder.hidden_dim, dtype=dtype, requires_grad=False).to(device)
        # regularization strength of matrix B
        self.reg_B = reg_B
        #
        self.lam_recon = lam_recon
        self.lam_prognostic = lam_prognostic
        self.lam_express = lam_express
        self.tau = tau
        self.use_lasso = use_lasso
        self.lasso_classifier = None
        if pre_decoder_Y is not None:
            self.pre_decoder_Y = pre_decoder_Y.to(device)
    
    def set_synthetic_control(self, classifier):
        assert self.use_lasso # we are not using gumble softmax
        self.lasso_classifier = classifier
    
    def check_device(self, *args):
        a_list = []
        for a in args:
            if a.device != self.device:
                res = a.to(self.device)
            else:
                res = a
            a_list.append(res)
        return a_list

    def get_representation(self, x, t, mask):
        # get representation C: B(atch size), D(im hidden)
        x, t, mask = self.check_device(x, t, mask)  # pylint: disable=unbalanced-tuple-unpacking
        C = self.encoder(x, t, mask)
        return C

    def get_reconstruction(self, C, t, mask):
        C, t, mask = self.check_device(C, t, mask)  # pylint: disable=unbalanced-tuple-unpacking
        x_hat = self.decoder(C, t, mask)
        return x_hat

    def get_prognostics(self, C, t, mask, robust=0):
        C, t, mask = self.check_device(C, t, mask)  # pylint: disable=unbalanced-tuple-unpacking
        y_hat = self.decoder_Y(C, t, mask)
        if robust == 2:
            y_pre_hat = self.pre_decoder_Y(C, t, mask)
        else:
            y_pre_hat = None
        return y_hat, y_pre_hat

    def _wrap_index(self, ind_0, ind_1, tensor):
        assert ind_0.dim() == ind_1.dim() == 1
        assert tensor.dim() == 2
        if torch.any(ind_0 >= tensor.shape[0]).item() is True:
            ind_0 = ind_0 % (tensor.shape[0] + 1)
            selector = ind_0 < tensor.shape[0]
            ind_0, ind_1 = ind_0[selector], ind_1[selector]
        if torch.any(ind_1 >= tensor.shape[1]).item() is True:
            ind_1 = ind_1 % (tensor.shape[1] + 1)
            selector = ind_1 < tensor.shape[1]
            ind_0, ind_1 = ind_0[selector], ind_1[selector]
        return ind_0, ind_1

    def get_B_reduced(self, batch_ind):
        batch_ind = self.check_device(batch_ind)[0]

        # B * N0
        batch_index = torch.stack([batch_ind] * self.n_unit, dim=-1)
        #print('batch_index shape', batch_index.shape, self.n_unit, self.B.shape)
        B_reduced = torch.gather(self.B, 0, batch_index)

        # create mask for self
        # mask = torch.zeros_like(B_reduced)
        # mask[torch.arange(batch_index.shape[0]), batch_index] = 1.

        if WRAP_INDEX_ON:
            # Only enable this codepath to run old code (as at time of paper publication).
            mask_inf = torch.zeros_like(B_reduced)
            ind_0, ind_1 = torch.arange(batch_ind.shape[0]), batch_ind
            ind_0, ind_1 = self._wrap_index(ind_0, ind_1, mask_inf)
            mask_inf[ind_0, ind_1] = torch.Tensor([float("-inf")])
        else:
            # Keep WRAP_INDEX_ON = False to use this newer corrected code.
            # *Model performance is unaffected by this fix.*
            mask_inf = torch.zeros(len(batch_ind), len(batch_ind)).to(B_reduced)
            mask_inf[batch_ind, batch_ind] = torch.Tensor([float("-inf")])
            mask_inf = mask_inf[: B_reduced.shape[0], : B_reduced.shape[1]]

        B_reduced = B_reduced + mask_inf
        # softmax
        # B_reduced = torch.softmax(B_reduced, dim=1)
        B_reduced = F.gumbel_softmax(B_reduced, tau=self.tau, dim=1, hard=False)

        return B_reduced

    def update_C0(self, C, batch_ind):
        C, batch_ind = self.check_device(C, batch_ind)  # pylint: disable=unbalanced-tuple-unpacking
        # in total data matrix, control first, treated second
        for i, ib in enumerate(batch_ind):
            if ib < self.n_unit:
                self.C0[ib] = C[i].detach()

    def self_expressive_loss(self, C, B_reduced):
        C, B_reduced = self.check_device(C, B_reduced)  # pylint: disable=unbalanced-tuple-unpacking

        err = C - torch.matmul(B_reduced, self.C0)
        err_mse = torch.mean(err[~torch.isnan(err)] ** 2)

        # L2 regularization
        reg = torch.mean(B_reduced[~torch.isnan(B_reduced)] ** 2)
        return self.lam_express * (err_mse + self.reg_B * reg)

    def reconstruction_loss(self, x, x_hat, mask):
        if self.lam_recon == 0:
            return 0
        x, x_hat, mask = self.check_device(x, x_hat, mask)  # pylint: disable=unbalanced-tuple-unpacking
        err = (x - x_hat) * mask
        err_mse = torch.sum(err ** 2) / torch.sum(mask)
        return err_mse * self.lam_recon

    def prognostic_loss(self, B_reduced, y_batch, y_control, y_mask):
        B_reduced, y_batch, y_control, y_mask = self.check_device(  # pylint: disable=unbalanced-tuple-unpacking
            B_reduced, y_batch, y_control, y_mask
        )
        # y_batch: B, DY
        # y_mask: B (1 if control, 0 if treated)
        # y_all: N0, DY
        # B_reduced: B, N0

        y_hat = torch.matmul(B_reduced, y_control)
        return torch.sum(((y_batch - y_hat) ** 2) * y_mask.unsqueeze(-1)) / torch.sum(y_mask) * self.lam_prognostic

    def prognostic_loss2(self, y, y_hat, mask, y_pre_hat=None, y_pre_full=None, robust=0, verbose=False, use_treated=False):
        y, y_hat, mask = self.check_device(y, y_hat, mask)  # pylint: disable=unbalanced-tuple-unpacking
        if robust == 0:
            if use_treated:
                err = (y - y_hat).permute(0,2,1) *(1- mask)
            else:
                err = (y - y_hat).permute(0,2,1) * mask
            err_mse = torch.sum(err ** 2) * self.lam_prognostic / torch.sum(mask)
            if verbose:
                print('control post error', err_mse)
        elif robust == 1:
            err1 = (y - y_hat).permute(0,2,1) * mask
            err1_mse = torch.sum(err1 ** 2) #/ torch.sum(mask)
            
            err2 = (y - y_hat).permute(0,2,1) * (1-mask)
            err2_mse = torch.sum(err2 ** 2) #/ torch.sum(1-mask)
            err_mse = (err1_mse + err2_mse)/torch.sum(torch.ones_like(mask)) * self.lam_prognostic #/2
            if verbose:
                print(torch.sum(mask), mask.shape, y.shape)
                print('control post error', err1_mse/ torch.sum(mask))
                print('treated post error', err2_mse/ torch.sum(1-mask))
        elif robust == 2:
            assert y_pre_full is not None
            assert y_pre_hat is not None
            y_pre_full, y_pre_hat = self.check_device(y_pre_full, y_pre_hat)
            err1 = (y - y_hat).permute(0,2,1) * mask
            err1_mse = torch.sum(err1 ** 2) #/ torch.sum(mask)
            
            err2 = (y - y_hat).permute(0,2,1) * (1-mask)
            err2_mse = torch.sum(err2 ** 2) #/ torch.sum(1-mask)
            
            err_pre = torch.sum((y_pre_hat - y_pre_full) ** 2) / torch.sum(torch.ones_like(y_pre_hat))
            err_mse = ((err1_mse + err2_mse)/torch.sum(torch.ones_like(mask)) + err_pre) * self.lam_prognostic #/3
            if verbose:
                print('control post error', err1_mse/ torch.sum(mask))
                print('treated post error', err2_mse/ torch.sum(1-mask))
                print('pre error', err_pre)
        else:
            print("robust levels only accepts 0,1,2")
            raise NotImplementedError()
        return err_mse

    def forward(self, x, t, mask, batch_ind, y_batch, y_control, y_mask):
        (  # pylint: disable=unbalanced-tuple-unpacking
            x,
            t,
            mask,
            batch_ind,
            y_batch,
            y_control,
            y_mask,
        ) = self.check_device(x, t, mask, batch_ind, y_batch, y_control, y_mask)
        C = self.get_representation(x, t, mask)
        x_hat = self.get_reconstruction(C, t, mask)
        
        if not self.use_lasso:
            B_reduced = self.get_B_reduced(batch_ind)
        else:
            B_reduced = self.lasso_classifier
        self_expressive_loss = self.self_expressive_loss(C, B_reduced)
        reconstruction_loss = self.reconstruction_loss(x, x_hat, mask)
        prognostic_loss = self.prognostic_loss(B_reduced, y_batch, y_control, y_mask)
        return self_expressive_loss + reconstruction_loss + prognostic_loss, C


class RegularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=True, device=DEVICE):
        super(RegularEncoder, self).__init__()
        self.input_dim = input_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional).to(device)
        if bidirectional:
            self.hidden_dim = hidden_dim * 2
        else:
            self.hidden_dim = hidden_dim

        attn_v_init = torch.ones(self.hidden_dim).to(device)
        self.attn_v = nn.Parameter(attn_v_init)

    def forward(self, x, t, mask):
        # T, B, Dh
        h, _ = self.lstm(x)  # pylint: disable=not-callable

        # T, B
        attn_score = torch.matmul(h, self.attn_v) / math.sqrt(self.hidden_dim)
        attn_weight = torch.softmax(attn_score, dim=0)

        # B, Dh
        C = torch.sum(h * attn_weight.unsqueeze(-1), dim=0)
        return C


class GRUDEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device=DEVICE):
        super(GRUDEncoder, self).__init__()
        self.input_dim = input_dim
        self.device = device

        self.grud = GRUD.GRUD(input_dim, hidden_dim, device=device).to(device)
        self.hidden_dim = hidden_dim

        attn_v_init = torch.ones(self.hidden_dim).to(device)
        self.attn_v = nn.Parameter(attn_v_init)

    def forward(self, x, t, mask):
        grud_in = self.grud.get_input_for_grud(t, x, mask)

        # T, B, Dh
        h = self.grud(grud_in).permute((1, 0, 2))  # pylint: disable=not-callable

        # T, B
        attn_score = torch.matmul(h, self.attn_v) / math.sqrt(self.hidden_dim)
        attn_weight = torch.softmax(attn_score, dim=0)

        # B, Dh
        C = torch.sum(h * attn_weight.unsqueeze(-1), dim=0)
        return C


class RegularDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len, device=DEVICE):
        super(RegularDecoder, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.lstm = nn.LSTM(hidden_dim, hidden_dim).to(device)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, C, t, mask):
        # C: B, Dh
        out, hidden = self.lstm(C.unsqueeze(0))  # pylint: disable=not-callable
        out = self.lin(out)

        out_list = [out]
        # run the remaining iterations
        for t in range(self.max_seq_len - 1):
            out, hidden = self.lstm(C.unsqueeze(0), hidden)  # pylint: disable=not-callable
            out = self.lin(out)
            out_list.append(out)

        return torch.cat(out_list, dim=0)


class LSTMTimeDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len, device=DEVICE):
        super(LSTMTimeDecoder, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim).to(device)
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.time_lin = nn.Linear(1, hidden_dim)

    def forward(self, C, t, mask):
        t_delta = t[1:] - t[:-1]
        t_delta_mat = torch.cat((torch.zeros_like(t_delta[0:1, ...]), t_delta), dim=0)
        time_encoded = self.time_lin(t_delta_mat[:, :, 0:1])

        # C: B, Dh
        lstm_in = torch.cat((C.unsqueeze(0), time_encoded[0:1, ...]), dim=-1)
        out, hidden = self.lstm(lstm_in)  # pylint: disable=not-callable
        out = self.lin(out)

        out_list = [out]
        # run the remaining iterations
        for t in range(self.max_seq_len - 1):
            lstm_in = torch.cat((C.unsqueeze(0), time_encoded[(t + 1) : (t + 2), ...]), dim=-1)
            out, hidden = self.lstm(lstm_in, hidden)  # pylint: disable=not-callable
            out = self.lin(out)
            out_list.append(out)

        return torch.cat(out_list, dim=0)


class GRUDDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len, device=DEVICE):
        super(GRUDDecoder, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.grud = GRUD.GRUD(hidden_dim, hidden_dim, device=device).to(device)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, C, t, mask):
        # C: B, Dh
        x = C[None, :, :].repeat(t.shape[0], 1, 1)
        mask = torch.ones_like(x)
        t = t[:, :, 0:1].repeat(1, 1, x.shape[2])

        grud_in = self.grud.get_input_for_grud(t, x, mask)

        # T, B, Dh
        h = self.grud(grud_in).permute((1, 0, 2))  # pylint: disable=not-callable
        out = self.lin(h)

        return out


class LinearDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len, device=DEVICE):
        super(LinearDecoder, self).__init__()
        assert output_dim == 1
        self.device = device
        self.max_seq_len = max_seq_len
        self.lin = nn.Linear(hidden_dim, max_seq_len).to(device)

    def forward(self, C, t, mask):
        # C: B, Dh -> B, T
        out = self.lin(C)  # pylint: disable=not-callable
        out = out.T.unsqueeze(-1)

        return out


class LinearHelper(nn.Module):
    def __init__(self, input_dim, output_dim, samples_taken=25, device=DEVICE):
        super(LinearHelper, self).__init__()
        self.device = device
        self.lin = nn.Linear(input_dim, output_dim).to(device)
        self.samples_taken = samples_taken

    def forward(self, x, t, mask):
        # C: B, Dh -> B, T
        # temporary implementation for cubic spline interpolation
        # since generated data is already of nice form
        x = x.permute(1,0,2)
        x = torch.flatten(x, start_dim=1)
        out = self.lin(x)  # pylint: disable=not-callable

        return out