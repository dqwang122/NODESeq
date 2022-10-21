import os
import argparse
import logging
from secrets import choice
import time
import datetime
import random
import string
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torchmetrics.functional as plfunc

from torchdyn.numerics import Euler, RungeKutta4, Tsitouras45, DormandPrince45, MSZero, Euler, HyperEuler
from torchdyn.numerics import odeint, odeint_mshooting, Lorenz
from torchdyn.core import ODEProblem, MultipleShootingProblem, NeuralODE

from plot import plotlosscurve, plottimecurve, plotnfecurve
from flogging import *
from utils import *
from data import CLSDataset, LMDataset, Vocabulary, collate_func

random.seed(233)
npr.seed(233)
torch.random.manual_seed(233)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vae')
parser.add_argument('--adjoint', type=eval, default=True)
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--max_len', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--bz', type=int, default=32)
parser.add_argument('--rtol', type=float, default=1e-2)
parser.add_argument('--atol', type=float, default=1e-2)
parser.add_argument('--solver', type=str, default='dopri5')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='trainlog/test')
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()





############# module for cls ode #############

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out

class Encoder(nn.Module):

    def __init__(self, embed, embed_dim, nhidden, latent_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.embed = embed
        self.lstm = nn.LSTM(embed_dim, nhidden, num_layers=num_layers, dropout=0.1, bidirectional=True, batch_first=True)

    def forward(self, x):
        enc_input = self.embed(x)
        enc_out, enc_hidden_state = self.lstm(enc_input)  # (bz, L, 2 * dim)
        return enc_out, enc_hidden_state

class Decoder(nn.Module):
    def __init__(self, embed, embed_dim, latent_dim, nhidden, num_layers=1):
        super().__init__()
        self.embed = embed
        self.dec_input_proj = nn.Linear(embed_dim+latent_dim, latent_dim)
        self.lstm = nn.LSTM(latent_dim, nhidden, num_layers=num_layers, dropout=0.1, bidirectional=False, batch_first=True)

    def forward(self, trg, z):
        dec_input = self.embed(trg)
        dec_enhance_input = torch.cat([dec_input, z], dim=-1)
        dec_enhance_input = self.dec_input_proj(dec_enhance_input)
        dec_out, dec_hidden_state = self.lstm(dec_enhance_input)
        return dec_out


class VAEModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, nlayer=2, w2v=None):
        super(VAEModel, self).__init__()
        self.w2v = w2v
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.enc = Encoder(self.embed, embed_dim, hidden_dim, latent_dim, nlayer)
        self.mu_proj = nn.Linear(2 * hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(2 * hidden_dim, latent_dim)
        self.dec = Decoder(self.embed, embed_dim, latent_dim, hidden_dim, nlayer)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)
        self.init_model()
        
    def forward(self, src, trg):
        bz, L = src.size()
        enc_out, enc_hidden_state = self.enc(src)       # (bz, L, 2 * dim)
        enc_repr = enc_out[:,-1,:]          # (bz, 2 * dim)
        mu, logvar = self.mu_proj(enc_repr), self.logvar_proj(enc_repr)     # (bz, latent)
        z = self.reparameterize(mu, logvar) 

        # z = odeint(self.ode, z, range(l)).permute(1, 0, 2)        # [batch_size, dec_hidden_size]

        z = z.unsqueeze(1).expand((bz, L-1, z.size(-1)))
        dec_output = self.dec(trg[:, :-1], z)    
        score = self.out_proj(dec_output)           

        return score, (mu, logvar)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu 

    def init_model(self, checkpoint=None):
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            logger.info("Initialize the model!")
            init_module(self.enc)
            init_module(self.dec)
        if self.w2v.all() != None:
            self.embed.weight.data = torch.Tensor(self.w2v).type_as(self.embed.weight.data)
            # self.enc.embed.requires_grad_(False)


class VAEODE(VAEModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, nlayer=2, w2v=None):
        super().__init__(vocab_size, embed_dim, hidden_dim, latent_dim, nlayer, w2v)
        self.ode = NeuralODE(LatentODEfunc(latent_dim, latent_dim * 2),solver=args.solver, rtol=args.rtol, atol=args.atol, atol_adjoint=args.atol, rtol_adjoint=args.rtol)

    def forward(self, src, trg):
        bz, L = src.size()
        enc_out, enc_hidden_state = self.enc(src)       # (bz, L, 2 * dim)
        enc_repr = enc_out[:,0,:]          # (bz, 2 * dim)
        mu, logvar = self.mu_proj(enc_repr), self.logvar_proj(enc_repr)     # (bz, latent_dim)
        z0 = self.reparameterize(mu, logvar) 

        t_span = torch.arange(L-1).type_as(z0)
        eval_times, z_span = self.ode(z0, t_span)               # enc_output: (L, bz, latent_dim)
        z_span = z_span.permute(1,0,2)
        dec_output = self.dec(trg[:, :-1], z_span)              
        score = self.out_proj(dec_output)           

        return score, (mu, logvar)
    



class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module, lr=1e-3, kl_coef=0.1, custom_log = None):
        super().__init__()
        self.lr = lr
        self.kl_coef = kl_coef
        self.model = model
        self.custom_log = custom_log.info if custom_log else self.print
        self.loss_func = nn.NLLLoss(ignore_index=0, reduction='mean')
    
    def forward(self, x, y):
        return self.model(x, y)

    def loss(self, pred, trg, mu, logvar):
        pred = F.log_softmax(pred, dim=-1)
        flat_pred = pred.contiguous().view(pred.size(0) * pred.size(1), -1)
        flat_trg = trg[:,1:].contiguous().view(-1)
        recov_loss = self.loss_func(flat_pred, flat_trg)
        kl_dist = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recov_loss + self.kl_coef * kl_dist
        return loss, (recov_loss.item(), kl_dist.item())

    def metric(self, pred, trg):
        bz, L, d = pred.size()
        predit = F.log_softmax(pred, dim=-1)
        flat_pred = predit.max(-1)[1].reshape(-1)
        flat_trg = trg[:,1:].reshape(-1)
        mask = (flat_trg != 0)
        masked_pred = mask * flat_pred
        incorrect = (masked_pred != flat_trg).sum()
        accu = 1 -  incorrect / mask.sum()

        recov_loss = self.loss_func(predit.view(bz*L, -1), flat_trg)
        ppl = torch.exp(recov_loss)

        return accu.item(), ppl.item()
    
    def training_step(self, batch, batch_idx):
        start_time = time.time()
        x, y = batch   
        y_hat, (mu, logvar) = self.model(x, y)   
        loss, (recov_loss, kl_dist) = self.loss(y_hat, y, mu, logvar)
        acc, ppl = self.metric(y_hat, y)

        if hasattr(self.model.enc, 'ode'):
            nfe = self.model.enc.ode.vf.nfe
        else:
            nfe = 0

        # print(batch_idx)
        end_time = time.time()-start_time
        self.log_dict({'train_loss': loss, 'recov_loss':recov_loss, 'kl_dist':kl_dist,'acc': acc, 'ppl':ppl, 'time': end_time, 'nfe':nfe})
        return {'loss': loss, 'recov_loss':recov_loss, 'kl_dist':kl_dist, 'accu': acc, 'ppl':ppl, 'time': end_time, 'nfe': nfe}   

    def training_epoch_end(self, outputs):
        avg_loss = sum([x['loss'] for x in outputs]) / len(outputs)
        avg_rec_loss = sum([x['recov_loss'] for x in outputs]) / len(outputs)
        avg_kl_loss = sum([x['kl_dist'] for x in outputs]) / len(outputs)
        avg_acc = sum([x['accu'] for x in outputs]) / len(outputs)
        avg_ppl = sum([x['ppl'] for x in outputs]) / len(outputs)
        total_time = sum([x['time'] for x in outputs])
        nfe = outputs[-1]['nfe']
        self.custom_log('Epoch {} | Iter {} | time: {:.4f}, NFE {}, training avg loss: {:.4f}, recov loss: {:.4f}, kl loss: {:.4f}, training avg acc {:.4f}, avg ppl {:.4f}'
                                .format(self.current_epoch, self.global_step, total_time, nfe, avg_loss, avg_rec_loss, avg_kl_loss, avg_acc, avg_ppl))
        

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat, (mu, logvar) = self.model(x, y)
        loss, (recov_loss, kl_dist) = self.loss(y_hat, y, mu, logvar)
        acc, ppl = self.metric(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_recov_loss':recov_loss, 'val_kl_dist':kl_dist, 'val_accu': acc, 'val_ppl': ppl})
        return {'val_loss': loss, 'val_accu': acc, 'val_recov_loss':recov_loss, 'val_kl_dist':kl_dist, 'val_ppl':ppl}

    def validation_epoch_end(self, outputs):
        avg_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        avg_rec_loss = sum([x['val_recov_loss'] for x in outputs]) / len(outputs)
        avg_kl_loss = sum([x['val_kl_dist'] for x in outputs]) / len(outputs)
        avg_acc = sum([x['val_accu'] for x in outputs]) / len(outputs)
        avg_ppl = sum([x['val_ppl'] for x in outputs]) / len(outputs)
        self.custom_log('Epoch {} | Iter {} | valid avg loss: {:.4f}, recov loss: {:.4f}, kl loss: {:.4f}, valid avg acc {:.4f}, avg ppl {:.4f}'
                                .format(self.current_epoch, self.global_step, avg_loss, avg_rec_loss, avg_kl_loss, avg_acc, avg_ppl))
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-5)
        sched = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt),
                 'monitor': 'train_loss', 
                 'interval': 'step',
                 'frequency': 10  }
        return [opt], [sched]
        # return opt


if __name__ == '__main__':
    embed_dim = 50
    hidden_size = 32
    latent_dim = 32
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # init custom logger
    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            logger.info('Create new workspace...')
            os.makedirs(args.train_dir)
        init_logger(os.path.join(args.train_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'))
    logger.info(args)
    logger.info(device)

    # load training dataset
    vocab = Vocabulary(f"{DATAPATH}/vocab.txt", max_size=30000)
    train_dataset = LMDataset(mode='train', path=DATAPATH, vocab=vocab, maxlen=args.max_len)
    test_dataset = LMDataset(mode='test', path=DATAPATH, vocab=vocab, maxlen=args.max_len)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bz, collate_fn=collate_func,shuffle=True, num_workers=32)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bz, collate_fn=collate_func, shuffle=False)

    # model
    glove = load_w2v(vocab.get_vocab_dict(), embed_dim)
    if args.model == "ode":
        model = VAEODE(len(vocab), embed_dim, hidden_dim=hidden_size, latent_dim=latent_dim, w2v=glove).to(device)
    else:
        model = VAEModel(len(vocab), embed_dim, hidden_dim=hidden_size, latent_dim=latent_dim, w2v=glove).to(device)
    model_number = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(model_number)


    try:
        learn = Learner(model, lr=args.lr, custom_log=logger)
        exper_logger = TensorBoardLogger(args.train_dir, name=args.model + 'logger')
        exper_logger.log_hyperparams(args)
        trainer = pl.Trainer(logger=exper_logger, 
                                default_root_dir=args.train_dir, 
                                enable_progress_bar=True,
                                max_epochs=args.max_epoch, gpus=1)
        trainer.fit(learn, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    except KeyboardInterrupt:
        version_path = os.path.join(args.train_dir, 'version_{}'.format(exper_logger.version))
        ckpt_path = os.path.join(version_path, 'ckpt.pth')
        trainer.save_checkpoint(ckpt_path)
        logger.info('Stored ckpt at {}'.format(ckpt_path))
    
    version_path = os.path.join(args.train_dir, 'version_{}'.format(exper_logger.version))
    ckpt_path = os.path.join(version_path, 'ckpt.pth')
    trainer.save_checkpoint(ckpt_path)
    logger.info('Stored ckpt at {}'.format(ckpt_path))
