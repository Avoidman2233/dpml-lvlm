# file: attributecompute_full.py
# 请直接用此文件全文替换你的旧文件

import os
import os.path as osp
import json
import random
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from itertools import combinations

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, MetricMeter, AverageMeter
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": "CoOp",
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_cls):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_cls = n_cls

    def forward(self, prompts_list, tokenized_prompts_lists):
        for idx in range(len(prompts_list)):
            prompts = prompts_list[idx]
            tokenized_prompts = tokenized_prompts_lists[idx]

            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

            if idx == 0:
                final_feat = x.unsqueeze(0)
            else:
                final_feat = torch.cat([final_feat, x.unsqueeze(0)], dim=0)

        return final_feat


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # two attributes
        n_att = cfg.TRAINER.COOP.N_ATT1
        self.n_att = n_att

        # attributes
        att1_text = cfg.TRAINER.COOP.ATT1_TEXT
        att2_text = cfg.TRAINER.COOP.ATT2_TEXT
        att3_text = cfg.TRAINER.COOP.ATT3_TEXT
        att4_text = cfg.TRAINER.COOP.ATT4_TEXT
        att5_text = cfg.TRAINER.COOP.ATT5_TEXT
        att_list = [att1_text, att2_text, att3_text, att4_text, att5_text]

        prompt_prefix = " ".join(["X"] * n_ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        self.all_combinations = []
        for r in range(1, len(att_list) + 1):
            combos = combinations(att_list, r)
            self.all_combinations.extend(combos)
        random.shuffle(self.all_combinations)

        all_comb_len = len(self.all_combinations)

        # make att_weight a trainable Parameter (float32)
        self.att_weight = nn.Parameter(torch.randn(all_comb_len, dtype=torch.float32))

        att_ctx_list = []
        ctx_list = []
        prompt_list = []

        tokenized_prompt_list = []
        embedding_list = []

        for combine in self.all_combinations:
            comb_len = len(combine)
            comb_list = nn.ParameterList()

            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            ctx_list.append(nn.Parameter(ctx_vectors))

            prompt = []
            temp_string = ""
            prefix = " ".join(["X"] * n_att)
            for idx in range(comb_len):
                word = combine[idx]
                temp_att = torch.empty(n_att, ctx_dim, dtype=dtype)
                nn.init.normal_(temp_att, std=0.01)
                comb_list.append(nn.Parameter(temp_att))
                temp_string = temp_string + prefix + " " + word + " "
            prompt = [temp_string + prompt_prefix + " " + name + "." for name in classnames]
            prompt_list.append(prompt)
            att_ctx_list.append(comb_list)

            tokenized_prompt = torch.cat([clip.tokenize(p) for p in prompt])
            tokenized_prompt_list.append(tokenized_prompt)

            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompt).type(dtype)
                embedding_list.append(embedding)

        self.ctx_list = nn.ParameterList(ctx_list)
        # store ParameterList objects inside a ModuleList
        self.att_ctx_list = nn.ModuleList(att_ctx_list)

        self.tokenized_prompt_list = tokenized_prompt_list
        self.embedding_lists = embedding_list

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        # sanity checks
        assert len(self.all_combinations) == len(self.tokenized_prompt_list)
        assert len(self.tokenized_prompt_list) == len(self.embedding_lists)

    def forward(self):
        ctx_list = self.ctx_list
        att_ctx_list = self.att_ctx_list
        embedding_list = self.embedding_lists
        n_att = self.n_att
        n_ctx = self.n_ctx

        prompts_list = []

        if self.class_token_position == "end":
            for idx in range(len(embedding_list)):
                embedding = embedding_list[idx].cuda()
                ctx = ctx_list[idx].cuda()
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
                att_ctxs = att_ctx_list[idx].cuda()
                att_len = len(att_ctxs)
                prefix = embedding[:, :1, :]
                if att_len == 1:
                    att_ctx = att_ctxs[0]

                    if att_ctx.dim() == 2:
                        att_ctx = att_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

                    mid_att1 = embedding[:, 1 + n_att: 1 + n_att + 1, :]
                    suffix = embedding[:, 1 + n_att + 1 + n_ctx:, :]

                    prompt = torch.cat([
                        prefix,
                        att_ctx,
                        mid_att1,
                        ctx,
                        suffix
                    ], dim=1)
                elif att_len == 2:
                    att_ctx1 = att_ctxs[0]
                    att_ctx2 = att_ctxs[1]

                    if att_ctx1.dim() == 2:
                        att_ctx1 = att_ctx1.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx2 = att_ctx2.unsqueeze(0).expand(self.n_cls, -1, -1)

                    mid_att1 = embedding[:, 1 + n_att: 1 + n_att + 1, :]
                    mid_att2 = embedding[:, 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1, :]
                    suffix = embedding[:, 1 + n_att + 1 + n_att + 1 + n_ctx:, :]

                    prompt = torch.cat([
                        prefix,
                        att_ctx1,
                        mid_att1,
                        att_ctx2,
                        mid_att2,
                        ctx,
                        suffix
                    ], dim=1)
                elif att_len == 3:
                    att_ctx1 = att_ctxs[0]
                    att_ctx2 = att_ctxs[1]
                    att_ctx3 = att_ctxs[2]

                    if att_ctx1.dim() == 2:
                        att_ctx1 = att_ctx1.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx2 = att_ctx2.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx3 = att_ctx3.unsqueeze(0).expand(self.n_cls, -1, -1)

                    mid_att1 = embedding[:, 1 + n_att: 1 + n_att + 1, :]
                    mid_att2 = embedding[:, 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1, :]
                    mid_att3 = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1 + n_att + 1, :]
                    suffix = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_ctx:, :]

                    prompt = torch.cat([
                        prefix,
                        att_ctx1,
                        mid_att1,
                        att_ctx2,
                        mid_att2,
                        att_ctx3,
                        mid_att3,
                        ctx,
                        suffix
                    ], dim=1)
                elif att_len == 4:
                    att_ctx1 = att_ctxs[0]
                    att_ctx2 = att_ctxs[1]
                    att_ctx3 = att_ctxs[2]
                    att_ctx4 = att_ctxs[3]

                    if att_ctx1.dim() == 2:
                        att_ctx1 = att_ctx1.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx2 = att_ctx2.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx3 = att_ctx3.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx4 = att_ctx4.unsqueeze(0).expand(self.n_cls, -1, -1)

                    mid_att1 = embedding[:, 1 + n_att: 1 + n_att + 1, :]
                    mid_att2 = embedding[:, 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1, :]
                    mid_att3 = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1 + n_att + 1, :]
                    mid_att4 = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att + 1, :]
                    suffix = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_ctx:, :]

                    prompt = torch.cat([
                        prefix,
                        att_ctx1,
                        mid_att1,
                        att_ctx2,
                        mid_att2,
                        att_ctx3,
                        mid_att3,
                        att_ctx4,
                        mid_att4,
                        ctx,
                        suffix
                    ], dim=1)
                elif att_len == 5:
                    att_ctx1 = att_ctxs[0]
                    att_ctx2 = att_ctxs[1]
                    att_ctx3 = att_ctxs[2]
                    att_ctx4 = att_ctxs[3]
                    att_ctx5 = att_ctxs[4]

                    if att_ctx1.dim() == 2:
                        att_ctx1 = att_ctx1.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx2 = att_ctx2.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx3 = att_ctx3.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx4 = att_ctx4.unsqueeze(0).expand(self.n_cls, -1, -1)
                        att_ctx5 = att_ctx5.unsqueeze(0).expand(self.n_cls, -1, -1)

                    mid_att1 = embedding[:, 1 + n_att: 1 + n_att + 1, :]
                    mid_att2 = embedding[:, 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1, :]
                    mid_att3 = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1 + n_att + 1, :]
                    mid_att4 = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att + 1, :]
                    mid_att5 = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att: 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att + 1, :]
                    suffix = embedding[:, 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_att + 1 + n_ctx:, :]

                    prompt = torch.cat([
                        prefix,
                        att_ctx1,
                        mid_att1,
                        att_ctx2,
                        mid_att2,
                        att_ctx3,
                        mid_att3,
                        att_ctx4,
                        mid_att4,
                        att_ctx5,
                        mid_att5,
                        ctx,
                        suffix
                    ], dim=1)
                else:
                    raise ValueError("Wrong att_len")
                prompts_list.append(prompt)
        else:
            raise ValueError("class_token_position must be 'end'")

        return prompts_list


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompt_list = self.prompt_learner.tokenized_prompt_list
        self.image_encoder = clip_model.visual
        self.n_cls = self.prompt_learner.n_cls
        self.text_encoder = TextEncoder(clip_model, self.n_cls)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        # directly use prompt_learner.att_weight (it's registered as a Parameter)
        att_weight = F.softmax(self.prompt_learner.att_weight, dim=0)

        prompts_list = self.prompt_learner()

        tokenized_prompts_list = self.tokenized_prompt_list
        text_features = self.text_encoder(prompts_list, tokenized_prompts_list)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = 0
        for idx in range(text_features.size(0)):
            logits += att_weight[idx] * logit_scale * image_features @ (text_features[idx, :, ]).t()

        return logits


@TRAINER_REGISTRY.register()
class AttributeCompute(TrainerX):
    """Context Optimization (CoOp)."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        self.model = CustomCLIP(cfg, classnames, clip_model)

        # disable gradients except prompt_learner (and any VPT)
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                # freeze specific param if needed
                if "ZS_image" in name:
                    param.requires_grad_(False)

        # move model to device first
        self.model.to(self.device)

        # data parallel if multiple GPUs
        device_count = torch.cuda.device_count()
        if device_count > 1:
            self.model = nn.DataParallel(self.model)

        # build main optimizer after possible DataParallel wrapping
        self.optim = build_optimizer(self.model, cfg.OPTIM)

        # build optimizer for att_weight using the real module (module if DataParallel)
        model_for_params = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(model_for_params.prompt_learner, "att_weight") or not isinstance(model_for_params.prompt_learner.att_weight, nn.Parameter):
            raise RuntimeError("prompt_learner.att_weight is not registered as nn.Parameter.")
        self.optim_weight = torch.optim.Adam(
            [model_for_params.prompt_learner.att_weight],
            lr=0.02,
            betas=(0.5, 0.999),
            weight_decay=0
        )

        # scheduler
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", model_for_params.prompt_learner, self.optim, self.sched)

        # scaler for amp
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # history lists for plotting
        self.train_loss_history = []
        self.lr_history = []

    def forward_backward_weight(self, image, label, model, optimizer, prec):
        # standard backward for weight optimizer (no debug)
        model_for_params = model.module if hasattr(model, "module") else model
        p = model_for_params.prompt_learner.att_weight

        if prec == 'amp':
            with autocast():
                output = model(image)
                loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            output = model(image)
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_summary = {'loss_val': loss.item()}
        return loss_summary

    def forward_backward_prompt(self, image, label, model, optimizer, prec):
        if prec == 'amp':
            with autocast():
                output = model(image)
                loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            output = model(image)
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_summary = {'loss_val': loss.item()}
        return loss_summary

    def forward_backward(self, batch_train, batch_val):
        model = self.model

        image_train, label_train = self.parse_batch_train(batch_train)
        image_val, label_val = self.parse_batch_train(batch_val)

        prec = self.cfg.TRAINER.COOP.PREC
        optim_prompt = self.optim
        optim_weight = self.optim_weight
        loss_summary = {}

        loss_prompt = self.forward_backward_prompt(image_train, label_train, model, optim_prompt, prec)
        loss_summary.update(loss_prompt)
        loss_weight = self.forward_backward_weight(image_val, label_val, model, optim_weight, prec)
        loss_summary.update(loss_weight)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_batch_val(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            batch_val = next(iter(self.val_loader))
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, batch_val)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        # record epoch-level loss & lr for plotting
        epoch_loss = None
        if "loss_val" in losses.meters:
            epoch_loss = losses.meters["loss_val"].avg
        else:
            # fallback: average any available metric
            if len(losses.meters) > 0:
                epoch_loss = next(iter(losses.meters.values())).avg
        if epoch_loss is None:
            epoch_loss = 0.0

        self.train_loss_history.append(float(epoch_loss))
        self.lr_history.append(float(self.get_current_lr()))

        # print attribute combination confidences
        combination = self.model.prompt_learner.all_combinations
        score = (F.softmax(self.model.prompt_learner.att_weight, dim=0)).detach().cpu().numpy()
        for comb, sc in zip(combination, score):
            print(f'words: {comb}, conf: {sc:.3f}')

        # plot and save loss & lr curve
        try:
            self._save_loss_lr_plot()
        except Exception as e:
            print("Warning: failed to save loss/lr plot:", e)

    def _save_loss_lr_plot(self):
        # determine output directory
        out_dir = None
        if hasattr(self, "cfg") and getattr(self.cfg, "OUTPUT_DIR", None):
            out_dir = self.cfg.OUTPUT_DIR
        if out_dir is None:
            out_dir = "./runs"
        os.makedirs(out_dir, exist_ok=True)

        epochs = np.arange(1, len(self.train_loss_history) + 1)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, marker="o")
        plt.title("Train Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.lr_history, marker="o")
        plt.title("Learning Rate per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.grid(True)

        plt.tight_layout()
        out_path = osp.join(out_dir, "loss_lr_curve.png")
        plt.savefig(out_path)
        plt.close()

        # also save numeric history for convenience
        hist_path = osp.join(out_dir, "loss_lr_history.json")
        try:
            with open(hist_path, "w") as f:
                json.dump({"loss": self.train_loss_history, "lr": self.lr_history}, f)
        except Exception:
            pass

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} from {} (epoch = {})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def write_scalar(self, tag, value, n_iter):
        """Write scalar for tensorboard if available, else print to console."""
        try:
            if hasattr(self, "_tb_writer") and self._tb_writer is not None:
                self._tb_writer.add_scalar(tag, value, n_iter)
                return
        except Exception:
            pass

        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = None
            if hasattr(self, "log_dir"):
                log_dir = getattr(self, "log_dir")
            elif hasattr(self, "cfg") and hasattr(self.cfg, "OUTPUT_DIR"):
                try:
                    log_dir = self.cfg.OUTPUT_DIR
                except Exception:
                    log_dir = None

            self._tb_writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else SummaryWriter()
            self._tb_writer.add_scalar(tag, value, n_iter)
            return
        except Exception:
            pass

        # fallback
        try:
            print(f"[write_scalar] step={n_iter} {tag}: {value}")
        except Exception:
            pass

