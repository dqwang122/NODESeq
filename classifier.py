import os
import argparse
import time
import datetime
import random
import string
import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics.functional as plfunc
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from torchdyn.numerics import Euler, RungeKutta4, Tsitouras45, DormandPrince45, MSZero, Euler, HyperEuler
from torchdyn.numerics import odeint, odeint_mshooting, Lorenz
from torchdyn.core import ODEProblem, MultipleShootingProblem, NeuralODE

from plot import plotlosscurve, plottimecurve, plotnfecurve
from flogging import *
from utils import *
from data import CLSDataset, Vocabulary, collate_func

random.seed(233)
npr.seed(233)
torch.random.manual_seed(233)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lr')
parser.add_argument('--adjoint', type=eval, default=True)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--bz', type=int, default=200)
parser.add_argument('--rtol', type=float, default=1e-2)
parser.add_argument('--atol', type=float, default=1e-2)
parser.add_argument('--solver', type=str, default='dopri5')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='trainlog/test')
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()





############# module for cls ode #############

class ODEfunc(nn.Module):

    def __init__(self, embed_size, nhidden):
        super(ODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(nhidden, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out

class ODEEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dim=10, nhidden=25):
        super(ODEEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.enc_proj = nn.Linear(embed_dim, nhidden)
        self.ode = NeuralODE(ODEfunc(embed_dim, nhidden), solver=args.solver, rtol=args.rtol, atol=args.atol)

    def forward(self, x):
        enc_input = self.embed(x)
        ode_input = self.enc_proj(enc_input)
        t_span = torch.Tensor([0, 1]).type_as(ode_input)
        eval_times, enc_output = self.ode(ode_input, t_span)
        return enc_output[-1]

class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_dim=10, nhidden=25):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.enc_proj = nn.Linear(embed_dim, nhidden)
        self.elu = nn.ELU(inplace=True)
        self.norm = nn.LayerNorm(nhidden)
        self.fc1 = nn.Linear(nhidden, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)

    def forward(self, x):
        enc_input = self.embed(x)
        enc_input = self.enc_proj(enc_input)
        enc_out = self.fc1(enc_input)
        # enc_input = self.norm(enc_input)
        enc_out = self.elu(enc_out)
        # enc_input = self.norm(enc_input)
        enc_out = self.fc2(enc_out)
        return enc_out

class LRClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, nlabel=2, w2v=None):
        super(LRClassifier, self).__init__()
        self.enc = encoder
        self.dec = nn.Linear(hidden_dim, nlabel)
        self.w2v = w2v
        self.init_model()
        
    def forward(self, src): 
        x = self.enc(src)
        score = self.dec(x).mean(1)
        score = F.softmax(score, dim=-1)
        return score

    def init_model(self, checkpoint=None):
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            logger.info("Initialize the model!")
            init_module(self.enc)
            init_module(self.dec)
        if self.w2v.all() != None:
            self.enc.embed.weight.data = torch.Tensor(self.w2v).type_as(self.enc.embed.weight.data)
            # self.enc.embed.requires_grad_(False)

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module, lr=1e-3, custom_log = None):
        super().__init__()
        self.lr = lr
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.custom_log = custom_log.info if custom_log else self.print
    
    def forward(self, x):
        return self.model(x)

    def loss(self, pred, truth):
        return self.loss_func(pred, truth)

    def metric(self, pred, truth):
        return plfunc.accuracy(pred, truth)
    
    def training_step(self, batch, batch_idx):
        start_time = time.time()
        x, y = batch   
        y_hat = self.model(x)   
        loss = self.loss(y_hat, y)
        acc = self.metric(y_hat, y)

        if hasattr(self.model.enc, 'ode'):
            nfe = self.model.enc.ode.vf.nfe
        else:
            nfe = 0

        end_time = time.time()-start_time
        self.log_dict({'train_loss': loss.item(), 'acc': acc, 'time': end_time, 'nfe':nfe})
        return {'loss': loss, 'accu': acc, 'time': end_time, 'nfe': nfe}   

    def training_epoch_end(self, outputs):
        avg_loss = sum([x['loss'] for x in outputs]) / len(outputs)
        avg_acc = sum([x['accu'] for x in outputs]) / len(outputs)
        total_time = sum([x['time'] for x in outputs])
        nfe = outputs[-1]['nfe']
        self.custom_log('Epoch {} | Iter {} | time: {:.4f}, NFE {}, training avg loss: {:.4f}, training avg acc {:.4f}'
                                .format(self.current_epoch, self.global_step, total_time, nfe, avg_loss, avg_acc))
        

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = self.metric(y_hat, y)
        self.log_dict({'val_loss': loss.item(), 'val_accu': acc})
        return {'val_loss': loss.item(), 'val_accu': acc.item()}

    def validation_epoch_end(self, outputs):
        avg_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        avg_acc = sum([x['val_accu'] for x in outputs]) / len(outputs)
        self.custom_log('Epoch {} | Iter {} | valid avg loss: {:.4f}, valid avg acc {:.4f}'
                                .format(self.current_epoch, self.global_step, avg_loss, avg_acc))
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-5)
        return opt


if __name__ == '__main__':
    embed_dim = 50
    hidden_size = 32
    lr = args.lr
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
    vocab = Vocabulary(f"{DATAPATH}/vocab.txt")
    train_dataset = CLSDataset(mode='train', path=DATAPATH, vocab=vocab)
    test_dataset = CLSDataset(mode='test', path=DATAPATH, vocab=vocab)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bz, collate_fn=collate_func,shuffle=True, num_workers=32)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bz, collate_fn=collate_func, shuffle=False)

    # model
    if args.model == 'ode':
        encoder = ODEEncoder(vocab_size=len(vocab), embed_dim=embed_dim, nhidden=hidden_size)
    else:
        encoder = Encoder(vocab_size=len(vocab), embed_dim=embed_dim, nhidden=hidden_size)
    glove = load_w2v(vocab.get_vocab_dict(), embed_dim)
    model = LRClassifier(encoder, hidden_dim=hidden_size, w2v=glove).to(device)
    model_number = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(model_number)


    try:
        learn = Learner(model, lr=args.lr, custom_log=logger)
        # exper_logger = CSVLogger(args.train_dir, name=args.model)
        exper_logger = TensorBoardLogger(args.train_dir, name=args.model + 'logger')
        exper_logger.log_hyperparams(args)
        trainer = pl.Trainer(logger=exper_logger, 
                                default_root_dir=args.train_dir, 
                                enable_progress_bar=False,
                                max_epochs=args.max_epoch, gpus=1)
        trainer.fit(learn, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            trainer.save_checkpoint(ckpt_path)
            logger.info('Stored ckpt at {}'.format(ckpt_path))
    
    ckpt_path = os.path.join(args.train_dir, 'last.ckpt')
    trainer.save_checkpoint(ckpt_path)
    logger.info('Stored ckpt at {}'.format(ckpt_path))


    # plotlosscurve([train_loss, test_loss], ['train','test'], '{}/loss.jpg'.format(args.train_dir))
    # plottimecurve(train_time, 'train', '{}/traintime.jpg'.format(args.train_dir))

    # if args.model == 'ode':
    #     logger.info('Training number of nfe ', train_nfe)
    #     plotnfecurve(train_nfe, train_time, 'train', '{}/nfetime.jpg'.format(args.train_dir))
