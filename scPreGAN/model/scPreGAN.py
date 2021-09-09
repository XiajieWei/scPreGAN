from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda import is_available as cuda_is_available
from torch import Tensor
from torch.utils.data import random_split
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
import scanpy as sc
from scipy import sparse
import pandas as pd
import random

from scPreGAN.model.Discriminator import Discriminator
from scPreGAN.model.Generator import Generator
from scPreGAN.model.Encoder import Encoder


def calc_gradient_penalty(disc, real_data, fake_data, batch_size, use_cuda, k, p):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda and cuda_is_available():
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = disc(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = torch.pow(gradients.norm(2, dim=1), p).mean() * k
    return gradient_penalty

class Model:
    def __init__(self, n_features, z_dim=16, min_hidden_size=256, use_cuda=True, manual_seed=3060):
        if manual_seed is None:
            manual_seed = random.randint(1, 10000)
        print("Random Seed: ", manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if use_cuda:
            torch.cuda.manual_seed_all(manual_seed)
        
        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform(m.weight, 1e-2)
                m.bias.data.fill_(0.01)

        D_A = Discriminator(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1)
        D_B = Discriminator(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1)

        G_A = Generator(z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features)
        G_B = Generator(z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features)
        E = Encoder(n_features=n_features, min_hidden_size=min_hidden_size, z_dim=z_dim)

        init_weights(E)
        init_weights(G_A)
        init_weights(G_B)
        init_weights(D_A)
        init_weights(D_B)

        if use_cuda and torch.cuda.is_available():
            D_A = D_A.cuda()
            D_B = D_B.cuda()
            G_A = G_A.cuda()
            G_B = G_B.cuda()
            E = E.cuda()

        self.E = E
        self.G_A = G_A
        self.G_B = G_B
        self.D_A = D_A
        self.D_B = D_B
        self.use_cuda = use_cuda
        print("Successfully created the model")

    def load_anndata(self, data_path, condition_key, condition, cell_type_key,
                     out_of_sample_prediction=False, prediction_cell_type=None):
        adata = sc.read(data_path)
        if out_of_sample_prediction and prediction_cell_type is not None:
            case_adata = adata[
                ~(adata.obs[cell_type_key] == prediction_cell_type) & (adata.obs[condition_key] == condition['case'])]
        elif out_of_sample_prediction == False:
            case_adata = adata[adata.obs[condition_key] == condition['case']]
        else:
            raise Exception("Must provide prediction_cell_type if out_of_sample_prediction is True.")

        control_adata = adata[adata.obs[condition_key] == condition['control']]

        if sparse.issparse(adata.X):
            control_pd = pd.DataFrame(data=control_adata.X.A, index=control_adata.obs_names,
                                      columns=control_adata.var_names)
            case_pd = pd.DataFrame(data=case_adata.X.A, index=case_adata.obs_names, columns=case_adata.var_names)
        else:
            control_pd = pd.DataFrame(data=control_adata.X, index=control_adata.obs_names,
                                      columns=control_adata.var_names)
            case_pd = pd.DataFrame(data=case_adata.X, index=case_adata.obs_names, columns=case_adata.var_names)
        print("Successfully loaded the model")
        return control_pd, case_pd

    def train(self, train_data, valid_data=None, niter=20000,
              model_path='./model', log_path='./logger',
              lr_e=0.0001, lr_g=0.001, lr_d=0.001, batch_size=64,
              lambda_adv=0.001, lambda_recon=1, lambda_encoding=0.1,
              betas=(0.5, 0.9), k=2, p=6):

        A_pd, B_pd = train_data
        A_tensor = Tensor(np.array(A_pd))
        B_tensor = Tensor(np.array(B_pd))

        if self.use_cuda and torch.cuda.is_available():
            A_tensor = A_tensor.cuda()
            B_tensor = B_tensor.cuda()

        A_Dataset = torch.utils.data.TensorDataset(A_tensor)
        B_Dataset = torch.utils.data.TensorDataset(B_tensor)
        A_train_loader = torch.utils.data.DataLoader(dataset=A_Dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     drop_last=True)

        B_train_loader = torch.utils.data.DataLoader(dataset=B_Dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     drop_last=True)
        if valid_data is not None:
            A_pd_val, B_pd_val = valid_data

            A_tensor_val = Tensor(np.array(A_pd_val))
            B_tensor_val = Tensor(np.array(B_pd_val))

            if self.use_cuda and torch.cuda.is_available():
                A_tensor_val = A_tensor_val.cuda()
                B_tensor_val = B_tensor_val.cuda()

            A_Dataset_val = torch.utils.data.TensorDataset(A_tensor_val)
            B_Dataset_val = torch.utils.data.TensorDataset(B_tensor_val)

            A_valid_loader = torch.utils.data.DataLoader(dataset=A_Dataset_val,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         drop_last=True)
            B_valid_loader = torch.utils.data.DataLoader(dataset=B_Dataset_val,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         drop_last=True)
        A_train_loader_it = iter(A_train_loader)
        B_train_loader_it = iter(B_train_loader)
        # loss function
        recon_criterion = nn.MSELoss()
        encoding_criterion = nn.MSELoss()
        # Optimizer
        optimizerD_A = torch.optim.Adam(self.D_A.parameters(), lr=lr_d, betas=betas)
        optimizerD_B = torch.optim.Adam(self.D_B.parameters(), lr=lr_d, betas=betas)
        optimizerG_A = torch.optim.Adam(self.G_A.parameters(), lr=lr_g, betas=betas)
        optimizerG_B = torch.optim.Adam(self.G_B.parameters(), lr=lr_g, betas=betas)
        optimizerE = torch.optim.Adam(self.E.parameters(), lr=lr_e)

        self.D_A.train()
        self.D_B.train()
        self.G_A.train()
        self.G_B.train()
        self.E.train()

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_path)

        # Training
        for iteration in range(1, niter + 1):
            if iteration % 10000 == 0:
                for param_group in optimizerD_A.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizerD_B.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizerG_A.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizerG_B.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizerE.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9

            # Train discriminator
            D_A_loss_item = 0.0
            D_B_loss_item = 0.0
            for count in range(0, 5):
                try:
                    real_A = A_train_loader_it.next()[0]
                    real_B = B_train_loader_it.next()[0]
                except StopIteration:
                    A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(B_train_loader)
                    real_A = A_train_loader_it.next()[0]
                    real_B = B_train_loader_it.next()[0]

                if self.use_cuda and cuda_is_available():
                    real_A = real_A.cuda()
                    real_B = real_B.cuda()

                self.D_A.zero_grad()
                self.D_B.zero_grad()

                out_A = self.D_A(real_A)
                out_B = self.D_B(real_B)

                real_A_z = self.E(real_A)
                AB = self.G_B(real_A_z)

                real_B_z = self.E(real_B)
                BA = self.G_A(real_B_z)

                out_BA = self.D_A(BA.detach())
                out_AB = self.D_B(AB.detach())

                D_A_gradient_penalty = calc_gradient_penalty(self.D_A, real_A.detach(), BA.detach(),
                                                             batch_size=batch_size, use_cuda=self.use_cuda,
                                                             k=k, p=p)
                D_B_gradient_penalty = calc_gradient_penalty(self.D_B, real_B.detach(), AB.detach(),
                                                             batch_size=batch_size, use_cuda=self.use_cuda,
                                                             k=k, p=p)

                D_A_real_loss = -torch.mean(out_A)
                D_B_real_loss = -torch.mean(out_B)
                D_A_fake_loss = torch.mean(out_BA)
                D_B_fake_loss = torch.mean(out_AB)
                D_A_loss = D_A_real_loss + D_A_fake_loss + D_A_gradient_penalty
                D_B_loss = D_B_real_loss + D_B_fake_loss + D_B_gradient_penalty

                D_A_loss.backward()
                D_B_loss.backward()
                optimizerD_A.step()
                optimizerD_B.step()

                D_A_loss_item += D_A_loss.item()
                D_B_loss_item += D_B_loss.item()
                writer.add_scalar('D_A_loss', D_A_loss, global_step=iteration)
                writer.add_scalar('D_B_loss', D_B_loss, global_step=iteration)

            # Train encoder and decoder
            try:
                real_A = A_train_loader_it.next()[0]
                real_B = B_train_loader_it.next()[0]
            except StopIteration:
                A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(B_train_loader)
                real_A = A_train_loader_it.next()[0]
                real_B = B_train_loader_it.next()[0]

            if self.use_cuda and cuda_is_available():
                real_A = real_A.cuda()
                real_B = real_B.cuda()

            self.G_A.zero_grad()
            self.G_B.zero_grad()
            self.E.zero_grad()

            real_A_z = self.E(real_A)
            AA = self.G_A(real_A_z)
            AB = self.G_B(real_A_z)

            AA_z = self.E(AA)
            AB_z = self.E(AB)
            ABA = self.G_A(AB_z)

            real_B_z = self.E(real_B)
            BA = self.G_A(real_B_z)
            BB = self.G_B(real_B_z)
            BA_z = self.E(BA)
            BB_z = self.E(BB)
            BAB = self.G_B(BA_z)

            out_AA = self.D_A(AA)
            out_AB = self.D_B(AB)
            out_BA = self.D_A(BA)
            out_BB = self.D_B(BB)
            out_ABA = self.D_A(ABA)
            out_BAB = self.D_B(BAB)

            # adversarial loss
            G_AA_adv_loss = -torch.mean(out_AA)
            G_BA_adv_loss = -torch.mean(out_BA)
            G_ABA_adv_loss = -torch.mean(out_ABA)

            G_BB_adv_loss = -torch.mean(out_BB)
            G_AB_adv_loss = -torch.mean(out_AB)
            G_BAB_adv_loss = -torch.mean(out_BAB)

            G_A_adv_loss = G_AA_adv_loss + G_BA_adv_loss + G_ABA_adv_loss
            G_B_adv_loss = G_BB_adv_loss + G_AB_adv_loss + G_BAB_adv_loss
            adv_loss = (G_A_adv_loss + G_B_adv_loss) * lambda_adv

            # reconstruction loss
            l_rec_AA = recon_criterion(AA, real_A)
            l_rec_BB = recon_criterion(BB, real_B)
            recon_loss = (l_rec_AA + l_rec_BB) * lambda_recon

            # encoding loss
            tmp_real_A_z = real_A_z.detach()
            tmp_real_B_z = real_B_z.detach()
            l_encoding_AA = encoding_criterion(AA_z, tmp_real_A_z)
            l_encoding_BB = encoding_criterion(BB_z, tmp_real_B_z)
            l_encoding_BA = encoding_criterion(BA_z, tmp_real_B_z)
            l_encoding_AB = encoding_criterion(AB_z, tmp_real_A_z)
            encoding_loss = (l_encoding_AA + l_encoding_BB + l_encoding_BA + l_encoding_AB) * lambda_encoding

            G_loss = adv_loss + recon_loss + encoding_loss

            G_loss.backward()

            optimizerG_A.step()
            optimizerG_B.step()
            optimizerE.step()


            writer.add_scalar('adv_loss', adv_loss, global_step=iteration)
            writer.add_scalar('recon_loss', recon_loss, global_step=iteration)
            writer.add_scalar('encoding_loss', encoding_loss, global_step=iteration)
            writer.add_scalar('G_loss', G_loss, global_step=iteration)


            if iteration % 100 == 0:
                print(
                    '[%d/%d] adv_loss: %.4f  recon_loss: %.4f encoding_loss: %.4f G_loss: %.4f D_A_loss: %.4f  D_B_loss: %.4f'
                    % (iteration, niter, adv_loss.item(), recon_loss.item(),
                       encoding_loss.item(), G_loss.item(), D_A_loss_item, D_B_loss_item))

                # Validation
                if valid_data is not None:
                    D_A_loss_val = 0.0
                    D_B_loss_val = 0.0
                    adv_loss_val = 0.0
                    recon_loss_val = 0.0
                    encoding_loss_val = 0.0
                    G_loss_val = 0.0
                    counter = 0

                    A_valid_loader_it = iter(A_valid_loader)
                    B_valid_loader_it = iter(B_valid_loader)

                    max_length = max(len(A_valid_loader), len(B_valid_loader))
                    with torch.no_grad():
                        for iteration_val in range(1, max_length):
                            try:
                                cellA_val = A_valid_loader_it.next()[0]
                                cellB_val = B_valid_loader_it.next()[0]
                            except StopIteration:
                                A_valid_loader_it, B_valid_loader_it = iter(A_valid_loader), iter(B_valid_loader)
                                cellA_val = A_valid_loader_it.next()[0]
                                cellB_val = B_valid_loader_it.next()[0]

                            counter += 1

                            real_A_z = self.E(cellA_val)
                            real_B_z = self.E(cellB_val)
                            AB = self.G_B(real_A_z)
                            BA = self.G_A(real_B_z)
                            AA = self.G_A(real_A_z)
                            BB = self.G_B(real_B_z)
                            AA_z = self.E(AA)
                            BB_z = self.E(BB)
                            AB_z = self.E(AB)
                            BA_z = self.E(BA)
                            ABA = self.G_A(AB_z)
                            BAB = self.G_B(BA_z)

                            outA_val = self.D_A(cellA_val)
                            outB_val = self.D_B(cellB_val)
                            out_AA = self.D_A(AA)
                            out_BB = self.D_B(BB)
                            out_AB = self.D_B(AB)
                            out_BA = self.D_A(BA)
                            out_ABA = self.D_A(ABA)
                            out_BAB = self.D_B(BAB)

                            D_A_real_loss_val = -torch.mean(outA_val)
                            D_B_real_loss_val = -torch.mean(outB_val)
                            D_A_fake_loss_val = torch.mean(out_BA)
                            D_B_fake_loss_val = torch.mean(out_AB)

                            D_A_loss_val += (D_A_real_loss_val + D_A_fake_loss_val).item()
                            D_B_loss_val += (D_B_real_loss_val + D_B_fake_loss_val).item()


                            G_AA_adv_loss_val = -torch.mean(out_AA)
                            G_BA_adv_loss_val = -torch.mean(out_BA)
                            G_ABA_adv_loss_val = -torch.mean(out_ABA)

                            G_BB_adv_loss_val = -torch.mean(out_BB)
                            G_AB_adv_loss_val = -torch.mean(out_AB)
                            G_BAB_adv_loss_val = -torch.mean(out_BAB)

                            G_A_adv_loss_val = G_AA_adv_loss_val + G_BA_adv_loss_val + G_ABA_adv_loss_val
                            G_B_adv_loss_val = G_BB_adv_loss_val + G_AB_adv_loss_val + G_BAB_adv_loss_val
                            adv_loss_val += (G_A_adv_loss_val + G_B_adv_loss_val).item() * lambda_adv

                            # reconstruction loss
                            l_rec_AA_val = recon_criterion(AA, cellA_val)
                            l_rec_BB_val = recon_criterion(BB, cellB_val)
                            recon_loss_val += (l_rec_AA_val + l_rec_BB_val).item() * lambda_recon

                            # encoding loss
                            l_encoding_AA_val = encoding_criterion(AA_z, real_A_z)
                            l_encoding_BB_val = encoding_criterion(BB_z, real_B_z)
                            l_encoding_BA_val = encoding_criterion(BA_z, real_B_z)
                            l_encoding_AB_val = encoding_criterion(AB_z, real_A_z)
                            encoding_loss_val = (l_encoding_AA_val + l_encoding_BB_val +
                                                 l_encoding_BA_val + l_encoding_AB_val).item() * lambda_encoding

                            G_loss_val += adv_loss_val + recon_loss_val + encoding_loss_val

                    print(
                        '[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f'
                        % (iteration, niter, adv_loss_val / counter, recon_loss_val / counter,
                           encoding_loss_val / counter, G_loss / counter, D_A_loss_val / counter,
                           D_B_loss_val / counter))

                    writer.add_scalar('adv_loss_val', adv_loss_val / counter, global_step=iteration)
                    writer.add_scalar('recon_loss_val', recon_loss_val / counter, global_step=iteration)
                    writer.add_scalar('encoding_loss_val', encoding_loss_val / counter, global_step=iteration)
                    writer.add_scalar('G_loss', G_loss / counter, global_step=iteration)
                    writer.add_scalar('D_A_loss_val', D_A_loss_val / counter, global_step=iteration)
                    writer.add_scalar('D_B_loss_val', D_B_loss_val / counter, global_step=iteration)

                    print(
                        '[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f'
                        % (iteration, niter, adv_loss_val / counter, recon_loss_val / counter,
                           encoding_loss_val / counter, G_loss / counter, D_A_loss_val / counter,
                           D_B_loss_val / counter))

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.E, os.path.join(model_path, 'E.pth'))
        torch.save(self.G_A, os.path.join(model_path, 'G_A.pth'))
        torch.save(self.G_B, os.path.join(model_path, 'G_B.pth'))
        torch.save(self.D_A, os.path.join(model_path, 'D_A.pth'))
        torch.save(self.D_B, os.path.join(model_path, 'D_B.pth'))
        writer.close()
        print("Training finished.")

    def predict(self, control_adata, cell_type_key, condition_key):
        if sparse.issparse(control_adata.X):
            control_tensor = Tensor(control_adata.X.A)
        else:
            control_tensor = Tensor(control_adata.X)
        if self.use_cuda and cuda_is_available():
            control_tensor = control_tensor.cuda()
        control_z = self.E(control_tensor)
        case_pred = self.G_B(control_z)
        pred_perturbed_adata = sc.AnnData(X=case_pred.cpu().detach().numpy(),
                                               obs={condition_key: ["Predicted data"] * len(case_pred),
                                                    cell_type_key: control_adata.obs[cell_type_key].tolist()})
        pred_perturbed_adata.var_names = control_adata.var_names
        print("Predicting data finished")
        return pred_perturbed_adata


