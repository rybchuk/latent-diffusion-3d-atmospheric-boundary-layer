import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

class LPIPSWithDiscriminator3D(nn.Module):
    def __init__(self, disc_start, cont1_start, cont2_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", continuity_weight1=0.0, continuity_weight2=0.0, patch_count=None):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.continuity_weight1 = continuity_weight1
        self.continuity_weight2 = continuity_weight2
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.continuity1_iter_start = cont1_start
        self.continuity2_iter_start = cont2_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.patch_count = patch_count

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = 0
            for k in range(inputs.shape[4]):
                p_loss += self.perceptual_loss(inputs[:,:,:,:,k].contiguous(), reconstructions[:,:,:,:,k].contiguous())
            p_loss /= inputs.shape[4]
            p_loss = p_loss[:,:,:,:,None]
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Mass conservation loss
        continuity_weight = 0.0
        if global_step > self.continuity2_iter_start:
            continuity_weight = adopt_weight(self.continuity_weight2, global_step, threshold=self.continuity2_iter_start)
        elif global_step > self.continuity1_iter_start:
            continuity_weight = adopt_weight(self.continuity_weight1, global_step, threshold=self.continuity1_iter_start)
        cont_loss = 0
        if continuity_weight > 0:
            for batchnum in range(reconstructions.shape[0]):   
                batchmem = reconstructions[batchnum,:,:,:,:]  # KEY ASSUMPTION: channels are ordered u, v, w
                du, dy, dz = 15, 15, 15  # TODO: take grid resolution as a config input
                facx, facy, facz = 0.25*du, 0.25*dy, 0.25*dz

                # Calculate gradients, assuming AMR-Wind numerical stencil
                dudx = facx*(-batchmem[0,:,:,:].roll(shifts=(-1,-1,-1), dims=(0,1,2)) + batchmem[0,:,:,:].roll(shifts=(0,-1,-1), dims=(0,1,2)) \
                             -batchmem[0,:,:,:].roll(shifts=(-1, 0,-1), dims=(0,1,2)) + batchmem[0,:,:,:].roll(shifts=(0, 0,-1), dims=(0,1,2)) \
                             -batchmem[0,:,:,:].roll(shifts=(-1,-1, 0), dims=(0,1,2)) + batchmem[0,:,:,:].roll(shifts=(0,-1, 0), dims=(0,1,2)) \
                             -batchmem[0,:,:,:].roll(shifts=(-1, 0, 0), dims=(0,1,2)) + batchmem[0,:,:,:].roll(shifts=(0, 0, 0), dims=(0,1,2)))
                dvdy = facy*(-batchmem[1,:,:,:].roll(shifts=(-1,-1,-1), dims=(0,1,2)) - batchmem[1,:,:,:].roll(shifts=(0,-1,-1), dims=(0,1,2)) \
                             +batchmem[1,:,:,:].roll(shifts=(-1, 0,-1), dims=(0,1,2)) + batchmem[1,:,:,:].roll(shifts=(0, 0,-1), dims=(0,1,2)) \
                             -batchmem[1,:,:,:].roll(shifts=(-1,-1, 0), dims=(0,1,2)) - batchmem[1,:,:,:].roll(shifts=(0,-1, 0), dims=(0,1,2)) \
                             +batchmem[1,:,:,:].roll(shifts=(-1, 0, 0), dims=(0,1,2)) + batchmem[1,:,:,:].roll(shifts=(0, 0, 0), dims=(0,1,2)))
                dwdz = facy*(-batchmem[2,:,:,:].roll(shifts=(-1,-1,-1), dims=(0,1,2)) - batchmem[2,:,:,:].roll(shifts=(0,-1,-1), dims=(0,1,2)) \
                             -batchmem[2,:,:,:].roll(shifts=(-1, 0,-1), dims=(0,1,2)) - batchmem[2,:,:,:].roll(shifts=(0, 0,-1), dims=(0,1,2)) \
                             +batchmem[2,:,:,:].roll(shifts=(-1,-1, 0), dims=(0,1,2)) + batchmem[2,:,:,:].roll(shifts=(0,-1, 0), dims=(0,1,2)) \
                             +batchmem[2,:,:,:].roll(shifts=(-1, 0, 0), dims=(0,1,2)) + batchmem[2,:,:,:].roll(shifts=(0, 0, 0), dims=(0,1,2)))
                cont_loss += torch.abs(torch.sum(dudx + dvdy + dwdz))
        cont_loss *= continuity_weight / reconstructions.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if (self.patch_count is None) or (self.patch_count == 1):  # Whole-image discriminator
                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(reconstructions.contiguous())
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                g_loss = -torch.mean(logits_fake)
            else:  # Patch-based discriminator
                g_loss = 0
                assert reconstructions.shape[2] % self.patch_count == 0, "Dimension not divisible by patch_count!"
                assert reconstructions.shape[3] % self.patch_count == 0, "Dimension not divisible by patch_count!"
                assert reconstructions.shape[4] % self.patch_count == 0, "Dimension not divisible by patch_count!"
                patch_i = int(reconstructions.shape[2] / self.patch_count)
                patch_j = int(reconstructions.shape[3] / self.patch_count)
                patch_k = int(reconstructions.shape[4] / self.patch_count)
                for i in range(self.patch_count):
                    for j in range(self.patch_count):
                        for k in range(self.patch_count):
                            patch = reconstructions[:,:,
                                i*patch_i:i*patch_i+patch_i,
                                j*patch_j:j*patch_j+patch_j,
                                k*patch_k:k*patch_k+patch_k]
                            if cond is None:
                                assert not self.disc_conditional
                                logits_fake = self.discriminator(patch.contiguous())
                            else:
                                assert self.disc_conditional
                                logits_fake = self.discriminator(torch.cat((patch.contiguous(), cond), dim=1))
                            g_loss += -torch.mean(logits_fake)
                g_loss /= self.patch_count**3

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + cont_loss

            if continuity_weight > 0:
                log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                    "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                    "{}/cont_loss".format(split): cont_loss.detach(),
                    }
            else:
                log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                    "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                    "{}/cont_loss".format(split): cont_loss,
                    }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if (self.patch_count is None) or (self.patch_count == 1):  # Whole-image discriminator
                if cond is None:
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(reconstructions.contiguous().detach())
                else:
                    logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            else:  # Patch-based discriminator
                d_loss = 0
                patch_i = int(reconstructions.shape[2] / self.patch_count)
                patch_j = int(reconstructions.shape[3] / self.patch_count)
                patch_k = int(reconstructions.shape[4] / self.patch_count)
                for i in range(self.patch_count):
                    for j in range(self.patch_count):
                        for k in range(self.patch_count):
                            patch_inp = inputs[:,:,
                                i*patch_i:i*patch_i+patch_i,
                                j*patch_j:j*patch_j+patch_j,
                                k*patch_k:k*patch_k+patch_k]   
                            patch_rec = reconstructions[:,:,
                                i*patch_i:i*patch_i+patch_i,
                                j*patch_j:j*patch_j+patch_j,
                                k*patch_k:k*patch_k+patch_k]                

                            if cond is None:
                                logits_real = self.discriminator(patch_inp.contiguous().detach())
                                logits_fake = self.discriminator(patch_rec.contiguous().detach())
                            else:
                                logits_real = self.discriminator(torch.cat((patch_inp.contiguous().detach(), cond), dim=1))
                                logits_fake = self.discriminator(torch.cat((patch_rec.contiguous().detach(), cond), dim=1))

                            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                            d_loss += disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log