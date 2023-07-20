import os

from sklearn import linear_model
import numpy as np
import torch
import torch.optim as optim

import util.batching_utils as batching

# from config import DEVICE


def create_paths(*args):
    for base_path in args:
        if not os.path.exists(base_path):
            os.makedirs(base_path)


def pre_train_reconstruction_loss(
    nsc,
    x_full,
    t_full,
    mask_full,
    x_full_val=None,
    t_full_val=None,
    mask_full_val=None,
    niters=5000,
    model_path="models/sync/{}.pth",
    batch_size=None,
):
    if x_full_val is None:
        x_full_val = x_full
        t_full_val = t_full
        mask_full_val = mask_full

    enc = nsc.encoder
    dec = nsc.decoder

    optimizer = optim.Adam(list(dec.parameters()) + list(enc.parameters()))

    test_freq = 500

    best_loss = 1e9

    for itr in range(1, niters + 1):

        optimizer.zero_grad()

        if batch_size is not None:
            x, t, mask = batching.get_batch_standard(  # pylint: disable=unbalanced-tuple-unpacking
                batch_size, x_full, t_full, mask_full
            )
        else:
            x, t, mask = x_full, t_full, mask_full

        C = nsc.get_representation(x, t, mask)
        x_hat = nsc.get_reconstruction(C, t, mask)
        loss = nsc.reconstruction_loss(x, x_hat, mask)

        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():

                C = nsc.get_representation(x_full_val, t_full_val, mask_full_val)
                x_hat = nsc.get_reconstruction(C, t_full_val, mask_full_val)
                loss = nsc.reconstruction_loss(x_full_val, x_hat, mask_full_val)

                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                if loss < best_loss:
                    best_loss = loss

                    torch.save(enc.state_dict(), model_path.format("encoder.pth"))
                    torch.save(dec.state_dict(), model_path.format("decoder.pth"))


def pre_train_reconstruction_prognostic_loss(
    nsc,
    x_full,
    t_full,
    mask_full,
    y_full,
    y_mask_full,
    x_full_val=None,
    t_full_val=None,
    mask_full_val=None,
    y_full_val=None,
    y_mask_full_val=None,
    y_pre_full=None,
    y_pre_full_val=None,
    niters=5000,
    model_path="models/sync/{}.pth",
    batch_size=None,
    robust=0,
    use_treated=False,
):
    if x_full_val is None:
        x_full_val = x_full
        t_full_val = t_full
        mask_full_val = mask_full
        y_full_val = y_full
        y_mask_full_val = y_mask_full

    enc = nsc.encoder
    dec = nsc.decoder

    assert nsc.decoder_Y is not None
    dec_Y = nsc.decoder_Y
    if robust == 2:
        assert nsc.pre_decoder_Y is not None
        pre_dec_Y = nsc.pre_decoder_Y

        optimizer = optim.Adam(list(dec.parameters()) + list(enc.parameters()) + list(dec_Y.parameters()) + list(pre_dec_Y.parameters()))
    else:
        optimizer = optim.Adam(list(dec.parameters()) + list(enc.parameters()) + list(dec_Y.parameters()))

    y_mask_full = torch.stack([y_mask_full] * dec_Y.max_seq_len, dim=0).unsqueeze(-1)

    test_freq = 100

    best_loss = 1e9

    for itr in range(1, niters + 1):

        optimizer.zero_grad()

        if batch_size is not None:
            x, t, mask, y, y_mask, y_pre = batching.get_batch_standard(  # pylint: disable=unbalanced-tuple-unpacking
                batch_size, x_full, t_full, mask_full, y_full, y_mask_full, y_pre_full
            )
        else:
            x, t, mask, y, y_mask, y_pre = x_full, t_full, mask_full, y_full, y_mask_full, y_pre_full

        C = nsc.get_representation(x, t, mask)
        x_hat = nsc.get_reconstruction(C, t, mask)
        loss_X = nsc.reconstruction_loss(x, x_hat, mask)
        
        y_hat, y_pre_hat = nsc.get_prognostics(C, t, mask, robust=robust)
        
        # print("y.shape", y.shape)
        # print("y_hat shape", y_hat.shape)
        # print("y mask", y_mask)
        # raise Exception('debug stop')
        loss_Y = nsc.prognostic_loss2(y, y_hat, y_mask, y_pre_hat=y_pre_hat, y_pre_full=y_pre, robust=robust, use_treated=use_treated)

        loss = loss_X + loss_Y
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                if x_full_val.shape[1] < 5000:
                    C = nsc.get_representation(x_full_val, t_full_val, mask_full_val)
                    x_hat = nsc.get_reconstruction(C, t_full_val, mask_full_val)
                    loss_X = nsc.reconstruction_loss(x_full_val, x_hat, mask_full_val)

                    y_hat, y_pre_hat = nsc.get_prognostics(C, t_full_val, mask_full_val, robust=robust)
                    loss_Y = nsc.prognostic_loss2(y_full_val, y_hat, y_mask_full_val,y_pre_hat=y_pre_hat, y_pre_full=y_pre_full_val, robust=robust, verbose=True, use_treated=use_treated)

                    loss = loss_X + loss_Y
                else:
                    loss_X = 0
                    loss_Y = 0
                    n_fold = x_full_val.shape[1] // 500

                    for fold in range(n_fold):
                        (  # pylint: disable=unbalanced-tuple-unpacking
                            x_full_vb,
                            t_full_vb,
                            mask_full_vb,
                            y_full_vb,
                            y_mask_full_vb,
                            y_pre_full_vb,
                        ) = batching.get_folds(
                            fold, n_fold, x_full_val, t_full_val, mask_full_val, y_full_val, y_mask_full_val, y_pre_full_val,
                        )

                        C = nsc.get_representation(x_full_vb, t_full_vb, mask_full_vb)
                        x_hat = nsc.get_reconstruction(C, t_full_vb, mask_full_vb)
                        loss_X += nsc.reconstruction_loss(x_full_vb, x_hat, mask_full_vb)

                        y_hat, y_pre_hat = nsc.get_prognostics(C, t_full_vb, mask_full_vb, robust=robust)
                        loss_Y += nsc.prognostic_loss2(y_full_vb, y_hat, y_mask_full_vb,y_pre_hat=y_pre_hat, y_pre_full=y_pre_full_vb,robust=robust, verbose=True, use_treated=use_treated)

                    loss = loss_X + loss_Y
                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                print("Iter {:04d} | Reconstruction Loss {:.6f}".format(itr, loss_X.item()))
                print("Iter {:04d} | Supervised Loss {:.6f}".format(itr, loss_Y.item()))
                if loss < best_loss:
                    best_loss = loss

                    torch.save(enc.state_dict(), model_path.format("encoder.pth"))
                    torch.save(dec.state_dict(), model_path.format("decoder.pth"))
                    torch.save(dec_Y.state_dict(), model_path.format("decoder_Y.pth"))
                    if robust == 2:
                        torch.save(pre_dec_Y.state_dict(), model_path.format("pre_decoder_Y.pth"))
    return best_loss


def train_cfr(
    cfr,
    x_full,
    t_full,
    mask_full,
    y_full,
    y_mask_full,
    x_full_val=None,
    t_full_val=None,
    mask_full_val=None,
    y_full_val=None,
    y_mask_full_val=None,
    niters=5000,
    model_path="",
    batch_size=None,
):
    if x_full_val is None:
        x_full_val = x_full
        t_full_val = t_full
        mask_full_val = mask_full
        y_full_val = y_full
        y_mask_full_val = y_mask_full

    enc = cfr.encoder

    assert cfr.decoder_Y is not None
    dec_Y = cfr.decoder_Y

    optimizer = optim.Adam(list(enc.parameters()) + list(dec_Y.parameters()))

    y_mask_full = torch.stack([y_mask_full] * dec_Y.max_seq_len, dim=0).unsqueeze(-1)

    test_freq = 500

    best_loss = 1e9

    for itr in range(1, niters + 1):

        optimizer.zero_grad()

        assert batch_size is None
        x, t, mask, y, y_mask = x_full, t_full, mask_full, y_full, y_mask_full

        loss = cfr(x, t, mask, y, y_mask)
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                loss = cfr(x_full_val, t_full_val, mask_full_val, y_full_val, y_mask_full_val)

                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                if loss < best_loss:
                    best_loss = loss

                    torch.save(enc.state_dict(), model_path.format("encoder.pth"))
                    torch.save(dec_Y.state_dict(), model_path.format("decoder_Y.pth"))
                    torch.save(cfr.state_dict(), model_path.format("cfr.pth"))
    return best_loss


def load_pre_train_and_init(
    nsc, x_full, t_full, mask_full, batch_ind_full, model_path="models/sync/{}.pth", init_decoder_Y=False, init_pre_dec_Y=False,
):
    enc = nsc.encoder
    dec = nsc.decoder

    enc.load_state_dict(torch.load(model_path.format("encoder.pth")))
    dec.load_state_dict(torch.load(model_path.format("decoder.pth")))

    if init_decoder_Y:
        dec_Y = nsc.decoder_Y
        dec_Y.load_state_dict(torch.load(model_path.format("decoder_Y.pth")))
    if init_pre_dec_Y:
        pre_dec_Y = nsc.pre_decoder_Y
        pre_dec_Y.load_state_dict(torch.load(model_path.format("pre_decoder_Y.pth")))

    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)
        nsc.update_C0(C, batch_ind_full)


def load_nsc(nsc, x_full, t_full, mask_full, batch_ind_full, model_path="models/sync/{}.pth"):
    nsc.load_state_dict(torch.load(model_path.format("nsc.pth")))
    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)
        nsc.update_C0(C, batch_ind_full)
    nsc.eval()


def train_B_self_expressive(
    nsc,
    x_full,
    t_full,
    mask_full,
    batch_ind_full,
    niters=20000,
    model_path="models/sync/{}.pth",
    batch_size=None,
    lr=1e-3,
    test_freq=1000,
):

    # mini-batch training not implemented
    assert batch_size is None

    optimizer = optim.Adam([nsc.B], lr=lr)

    best_loss = 10000

    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)

    for itr in range(1, niters + 1):

        optimizer.zero_grad()

        B_reduced = nsc.get_B_reduced(batch_ind_full)

        loss = nsc.self_expressive_loss(C, B_reduced)

        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                if np.isnan(loss.item()):
                    return 1
                if loss < best_loss:
                    best_loss = loss
                    torch.save(nsc.state_dict(), model_path.format("nsc.pth"))
    return 0

def train_B_self_expressive_lasso(
    nsc,
    x_full,
    t_full,
    mask_full,
    batch_ind_full,
    niters=20000,
    lr=1e-3,
    test_freq=1000,
    alpha=.01
):
    clf = linear_model.Lasso(alpha=alpha, fit_intercept=False,max_iter=10000)

    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)

        #print('C0:', nsc.C0.shape)
        #print('C', C.shape)
        
        B_reduced = torch.ones((C.shape[0], nsc.C0.shape[0]))
        #print(B_reduced.device)
        #print('B-reduced', B_reduced.shape)
        for i in range(C.shape[0]):
            clf.fit(torch.transpose(nsc.C0.cpu(),0,1), C[i,:].cpu())
            #print(C[i,:])
            #print(torch.transpose(nsc.C0,0,1)[:,6:])
            B_reduced[i,:] = torch.tensor(clf.coef_)
        #print(torch.count_nonzero(B_reduced))
        #raise Exception('stop')
    
    
        loss = nsc.self_expressive_loss(C, B_reduced)


        print("After Lasso procedure | Total Loss {:.6f}".format(loss.item()))
        if np.isnan(loss.item()):
            return [1,None]
    return [0, B_reduced]


def train_all_losses(
    nsc,
    x_full,
    t_full,
    mask_full,
    batch_ind_full,
    y_full,
    y_control,
    y_mask_full,
    niters=10000,
    model_path="models/sync/{}.pth",
    batch_size=None,
):

    # mini-batch training not implemented
    assert batch_size is None

    optimizer = optim.Adam(nsc.parameters(), lr=1e-3)

    test_freq = 1000
    best_loss = 1e9

    for itr in range(1, niters + 1):

        optimizer.zero_grad()

        loss, C = nsc(
            x_full, t_full, mask_full, batch_ind_full, y_batch=y_full, y_control=y_control, y_mask=y_mask_full
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            C = nsc.get_representation(x_full, t_full, mask_full)
            nsc.update_C0(C, batch_ind_full)

        if itr % test_freq == 0:
            with torch.no_grad():
                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                if np.isnan(loss.item()):
                    return 1
                if loss < best_loss:
                    best_loss = loss
                    torch.save(nsc.state_dict(), model_path.format("nsc.pth"))
    return 0
