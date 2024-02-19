from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection, CLIPTextModel
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.modeling_outputs import BaseModelOutputWithPooling

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 


def encode_text_word_embedding(text_encoder: CLIPTextModel, input_ids: torch.tensor, word_embeddings: torch.tensor,
                               num_vstar: int = 1) -> BaseModelOutputWithPooling:
    """
    Encode text by replacing the '$' with the PTEs extracted with the inversion adapter.
    Heavily based on hugginface implementation of CLIP.
    """
    existing_indexes = (input_ids == 259).nonzero(as_tuple=True)[0]  # 259 is the index of '$' in the vocabulary
    existing_indexes = existing_indexes.unique()
    if len(existing_indexes) > 0:  # if there are '$' in the text
        _, counts = torch.unique((input_ids == 259).nonzero(as_tuple=True)[0], return_counts=True)
        cum_sum = torch.cat((torch.zeros(1, device=input_ids.device).int(), torch.cumsum(counts, dim=0)[:-1]))
        first_vstar_indexes = (input_ids == 259).nonzero()[cum_sum][:,
                              1]  # get the index of the first '$' in each sentence
        rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_vstar)])
        word_embeddings = word_embeddings.to(input_ids.device)

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    seq_length = input_ids.shape[-1]
    position_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_length]
    input_embeds = text_encoder.text_model.embeddings.token_embedding(input_ids)

    if len(existing_indexes) > 0:
        assert word_embeddings.shape[0] == input_embeds.shape[0]
        if len(word_embeddings.shape) == 2:
            word_embeddings = word_embeddings.unsqueeze(1)
        input_embeds[torch.arange(input_embeds.shape[0]).repeat_interleave(
            num_vstar).reshape(input_embeds.shape[0], num_vstar)[existing_indexes.cpu()], rep_idx.T] = \
            word_embeddings.to(input_embeds.dtype)[existing_indexes]  # replace the '$' with the PTEs

    position_embeddings = text_encoder.text_model.embeddings.position_embedding(position_ids)
    hidden_states = input_embeds + position_embeddings

    bsz, seq_len = input_shape

    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )

    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=None,
        causal_attention_mask=causal_attention_mask,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)

    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype) # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class InversionAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim:int, output_dim, config, num_encoder_layers, dropout=0.5):
        super().__init__()
        self.config = config
        self.encoder_layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(num_encoder_layers)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, None, None)
            x = x[0]
        x = x[:, 0, :]
        x = self.post_layernorm(x)
        return self.layers(x)


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None, opt=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.opt = opt

        print(f'[INFO] loading stable diffusion...')
        
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        # if is_xformers_available():
        #     self.unet.enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(self.device)
        self.vision_encoder.requires_grad_(False)
        self.inversion_adapter = InversionAdapter(input_dim=self.vision_encoder.config.hidden_size,
            hidden_dim=self.vision_encoder.config.hidden_size * 4,
            output_dim=1024 * 8,
            num_encoder_layers=1,
            config=self.vision_encoder.config).to(self.device)
        ckpt = torch.load(opt.inversion_ckpt)
        self.inversion_adapter.load_state_dict(ckpt)

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt, sketch_image=None):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        # text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        # with torch.no_grad():
        #     text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        text_input = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length, padding="max_length",
                                           truncation=True, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            clip_features = self.vision_encoder(
                sketch_image.permute(0, 3, 1, 2).to(self.device)).last_hidden_state
            word_embeddings = self.inversion_adapter(clip_features.to(self.device))
            word_embeddings = word_embeddings.reshape((1, 8, -1))
            encoder_hidden_states = encode_text_word_embedding(self.text_encoder, text_input,
                                                                word_embeddings,
                                                                num_vstar=8).last_hidden_state
        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, encoder_hidden_states])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, q_unet = None, pose = None, shading = None, grad_clip = None, as_latent = False, t5 = False):
        
        # interp to 512x512 to be fed into vae.
        assert torch.isnan(pred_rgb).sum() == 0, print(pred_rgb)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
        elif self.opt.latent == True:
            latents = pred_rgb
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)        

        if t5: # Anneal time schedule
            t = torch.randint(self.min_step, 500 + 1, [1], dtype=torch.long, device=self.device)
        else:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if self.opt.sds is False:
                if q_unet is not None:
                    if pose is not None:
                        noise_pred_q = q_unet(latents_noisy, t, c = pose, shading = shading).sample
                    else:
                        raise NotImplementedError()

                    if self.opt.v_pred:
                        sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
                        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                        while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
                            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                        while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
                            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                        noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy


        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        if q_unet is None or self.opt.sds:
            grad = w * (noise_pred - noise)
        else:
            grad = w * (noise_pred - noise_pred_q)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        pseudo_loss = torch.mul((w*noise_pred).detach(), latents.detach()).detach().sum()

        return loss, pseudo_loss, latents

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
