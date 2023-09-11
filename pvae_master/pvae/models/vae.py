# Base VAE class definition

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from pvae_master.pvae.utils import get_mean_param
from pvae_master.pvae.models.architectures import EncMob, DecMob, DecBernouilliWrapper, EncWrapped
from pvae_master.pvae.distributions.riemannian_normal import RiemannianNormal
from pvae_master.pvae.manifolds.poincareball import PoincareBall
import importlib

class VAE(nn.Module):
    def __init__(self, prior_dist, posterior_dist, likelihood_dist, params, lossconfig, enc=EncMob, dec=DecMob):
        super(VAE, self).__init__()
        self.pz = RiemannianNormal(loc=nn.Parameter(torch.zeros(1, 5), requires_grad=False), scale=nn.Parameter(torch.zeros(1, 1), requires_grad=False), manifold=PoincareBall(dim=5, c=1.7)) #FIX
        self.px_z = dist.RelaxedBernoulli
        self.qz_x = RiemannianNormal(loc=nn.Parameter(torch.zeros(1, 5)), scale=nn.Parameter(torch.zeros(1, 1), requires_grad=False), manifold=PoincareBall(dim=5, c=1.7)) #FIX
        self.enc = enc(manifold=PoincareBall(dim=5, c=1.7), data_size=256, non_lin=nn.ReLU(inplace=False), num_hidden_layers=3, hidden_dim=200, prior_iso=True)
        self.dec = dec(manifold=PoincareBall(dim=5, c=1.7), data_size=256, non_lin=nn.ReLU(inplace=False), num_hidden_layers=3, hidden_dim=200) #FIX
        self.modelName = None
        self.params = params
        self.data_size = params.data_size
        self.prior_std = params.prior_std
        # from latent diffusion
        self.image_key = 'image'
        self.loss = self.instantiate_from_config(lossconfig)
        #personally added
        #self.linearlayer = nn.Linear


        if self.px_z == dist.RelaxedBernoulli:
            self.px_z.log_prob = lambda self, value: \
                -F.binary_cross_entropy_with_logits(
                    self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value),
                    value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value,
                    reduction='none'
                )

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        raise NotImplementedError

    def generate(self, N, K): #same as log images?
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            px_z_params = self.dec(self.pz(*self.pz_params).sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, \
            means.view(-1, *means.size()[2:]), \
            samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample(torch.Size([1])).squeeze(0))

        return get_mean_param(px_z_params)

    def forward(self, x, K=1):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    # from latent diffusion paper's VAE:========================================================================

    def encode(self, x): #fixed
        #h = self.enc(x)
        qz_x = RiemannianNormal(*self.enc(x))
        #return self.qz_x(*h)
        z = qz_x.rsample(torch.Size([1]))
        return torch.Tensor(z)

    def decode(self, z): #fixed
        px_z = self.px_z(*self.dec(z))
        tmp = px_z.sample()
        print(tmp.shape)
        return tmp

    #def forward(self, input, sample_posterior=True):
    #    posterior = self.encode(input)
    #    if sample_posterior:
    #        z = posterior.sample()
    #    else:
    #        z = posterior.mode()
    #    dec = self.decode(z)
    #    return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            #last_layer=self.get_last_layer(), 
                                            split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                #last_layer=self.get_last_layer(), 
                                                split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        #last_layer=self.get_last_layer(), 
                                        split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            #last_layer=self.get_last_layer(), 
                                            split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x




    def get_obj_from_str(self, string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)

    def instantiate_from_config(self, config):
        if not "target" in config:
            if config == '__is_first_stage__':
                return None
            elif config == "__is_unconditional__":
                return None
            raise KeyError("Expected key `target` to instantiate.")
        return self.get_obj_from_str(config["target"])(**config.get("params", dict()))
    
    

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std_scale)

    def init_last_layer_bias(self, dataset): pass
