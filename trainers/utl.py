import os.path as osp
import os
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import random
from sklearn.decomposition import PCA
import numpy as np
from dassl.utils import mkdir_if_missing

_tokenizer = _Tokenizer()


def load_classes_name(path):
    txt = open(path, 'r')
    lines = txt.readlines()
    label_list = []
    for line in lines:
        if line.endswith('\n'):
            label_ = line.split(': ')[1][1:-3]
        else:
            label_ = line.split(': ')[1][1:-2]
        label_list.append(label_)
    return label_list


def open_word_select(open_names_train_path, word_source):
    # IN classes load
    # txt = open('imagenet1000_clsidx_to_labels.txt', 'r')
    # lines = txt.readlines()
    # label_list = []
    # for line in lines:
    #     if line.endswith('\n'):
    #         label_ = line.split(': ')[1][1:-3]
    #     else:
    #         label_ = line.split(': ')[1][1:-2]
    #     label_list.append(label_)
    if word_source == 0:
        print('WORD_SOURCE: IN21K, ')
        label_list = load_classes_name('imagenet1000_clsidx_to_labels.txt')
        # print(len(label_list))

        # Load splits
        with open('./imagenet_osr_splits_winter21.pkl', 'rb') as handle:
            precomputed_info = pickle.load(handle)

        osr_wnids_easy = precomputed_info['easy_i21k_classes']
        osr_wnids_hard = precomputed_info['hard_i21k_classes']
        osr_winds = []
        osr_winds.extend(osr_wnids_easy)
        osr_winds.extend(osr_wnids_hard)
        # print(osr_winds)

        txt = open('imagenet21k_wordnet_ids.txt', 'r')
        lines = txt.readlines()
        ids_list = []
        for line in lines:
            if line.endswith('\n'):
                ids = line[:-1]
            else:
                ids = line
            ids_list.append(ids)

        txt = open('imagenet21k_wordnet_lemmas.txt', 'r')
        lines = txt.readlines()
        open_list = []
        for line in lines:
            if line.endswith('\n'):
                lem = line[:-1]
            else:
                lem = line
            open_list.append(lem)
        # print(open_list)
        open_name_list = []
        for i in range(len(ids_list)):
            a = 1
            for unit in osr_winds:
                if unit == ids_list[i]:
                    # print(ids_list[i])
                    a = 0
            if a == 1:
                open_name_list.append(open_list[i])

        label_unit_list = []

        for unit in label_list:
            for u in unit.split(', '):
                label_unit_list.append(u)

        select_open_list = []
        for unit in open_name_list:
            a = 1
            for u in unit.split(', '):
                if u in set(label_unit_list):
                    a = 0
                    # print(u)
            if a == 1:
                select_open_list.append(unit)

        with open(open_names_train_path, 'w') as file:
            for item in select_open_list:
                file.write("%s\n" % item)

    return select_open_list


def UnknownLabelSmoothing(target=None, smoothing=0, smoothing_open=0, open_label_num=0, num_classes=1000):
    close_target = torch.nn.functional.one_hot(target, num_classes=num_classes)
    b_z = close_target.size()[0]
    labels = ((1 - smoothing) * close_target) + (smoothing / num_classes)
    open_target_init = torch.ones(open_label_num)
    # smoothing_open = 0.001
    open_target = smoothing_open * open_target_init
    open_target = open_target.unsqueeze(0).expand(b_z, -1)
    open_target = open_target.cuda()
    target = torch.cat([labels, open_target], dim=1)
    return target


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

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptopenLearner(nn.Module):

    def __init__(self, cfg, classnames, opennames_train, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.UTL.N_CTX
        o_cls = cfg.TRAINER.UTL.UNKNOWN_TEXTS_NUM
        n_opl_name = cfg.TRAINER.UTL.N_UTX
        opl_name_len = n_opl_name
        pca_components_num = cfg.TRAINER.UTL.PCA_CPS_NUM
        lcw_normal_init = cfg.TRAINER.UTL.NORMAL_INIT
        learnable = cfg.TRAINER.UTL.BASIS_LEARNABLE
        n_components = pca_components_num
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # context random initialization
        if cfg.TRAINER.UTL.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        #######
        open_vec_dir = os.path.join(cfg.DATASET.ROOT, "open_vec",str(cfg.TRAINER.UTL.OPEN_WORDS_NUM), str(n_opl_name), str(pca_components_num))
        mkdir_if_missing(open_vec_dir)
        open_vec_path = os.path.join(open_vec_dir, "open_vectors.txt")

        if os.path.exists(open_vec_path):
            print(f"Loading open words from {open_vec_path}")
            centers = np.loadtxt(open_vec_path)
        else:
            print('open words pca:')

            prompts_opl = [name for name in opennames_train]
            tokenized_prompts_opl = torch.cat([clip.tokenize(p) for p in prompts_opl])
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts_opl).type(torch.float32)

            opl_vecters = (embedding[:, 1: 1 + opl_name_len, :]).data.cpu().numpy()
            print(opl_vecters.shape)
            points = opl_vecters.reshape(opl_vecters.shape[0], 1, -1).squeeze()
            pca = PCA(n_components=n_components)
            pca.fit(points)
            print(np.sum(pca.explained_variance_ratio_))
            centers = pca.components_
            print(centers.shape)
            np.savetxt(open_vec_path, centers)
        print('open words pca:')

        # use the linear combination of given words' pca components as open_label vectors
        open_vectors = torch.from_numpy(centers).type(torch.float32).to("cuda")
        print(open_vectors.size())
        self.open_vectors = open_vectors
        if learnable:
            self.open_vectors = nn.Parameter(open_vectors)
        w_vectors = torch.empty(o_cls, n_components, dtype=torch.float32)
        print(w_vectors.size())
        if lcw_normal_init:
            print('linear combination weights matrix init: lcw_normal_init')
            torch.nn.init.normal_(w_vectors, std=0.02)
        self.open_w = nn.Parameter(w_vectors)

        # unknown_vector = unknown_vector.data.cpu().numpy()
        # unknown_vector = unknown_vector.reshape(-1, opl_name_len, 512)
        # unknown_vector = torch.from_numpy(unknown_vector).to(dtype).to("cuda")
        # print(unknown_vector.size())
        #
        # self.open_names = unknown_vector

        opl_prompt_prefix = " ".join(["X"] * n_opl_name)
        print(f'OPEN WORDS PCA COMPONENTS linear C: "{opl_prompt_prefix}"')
        print(f"PCA components : {n_components}")
        print(f"Number of open labels : {o_cls}")
        print(f"LENS of NAME : {n_opl_name}")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts_n = [prompt_prefix + " " + name + "." for name in classnames]

        prompts_o = [prompt_prefix + " " + opl_prompt_prefix + "." for k in range(o_cls)]

        tokenized_prompts_n = torch.cat([clip.tokenize(p) for p in prompts_n])
        tokenized_prompts_o = torch.cat([clip.tokenize(p) for p in prompts_o])
        tokenized_prompts = torch.cat([tokenized_prompts_n, tokenized_prompts_o])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.o_cls = o_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.n_opl_name = n_opl_name
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.co_cls = o_cls + n_cls
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.co_cls, -1, -1)

        unknown_vector = torch.mm(self.open_w, self.open_vectors)
        unknown_vector = torch.reshape(unknown_vector, (-1, self.n_opl_name, 512))
        names = unknown_vector

        # print(len(names))
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":

            prompts = []
            for i in range(self.co_cls):
                prefix_i = prefix[i: i + 1, :, :]
                ctx_i = ctx[i: i + 1, :, :]
                if i < self.n_cls:
                    suffix_i = suffix[i: i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            ctx_i,  # (1, n_ctx, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                elif i >= self.n_cls and i < self.co_cls:
                    name_i = names[i - self.n_cls:i - self.n_cls + 1, :, :]
                    suffix_i = suffix[i: i + 1, self.n_opl_name:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (n_cls, 1, dim)
                            ctx_i,  # (n_cls, n_ctx, dim)
                            name_i,
                            suffix_i,  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                else:
                    print("error")
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts

class UnknownTextsLearner(nn.Module):
    def __init__(self, cfg, opennames_train, clip_model):
        super().__init__()
        o_cls = cfg.TRAINER.UTL.UNKNOWN_TEXTS_NUM
        n_opl_name = cfg.TRAINER.UTL.N_UTX
        opl_name_len = n_opl_name

        pca_components_num = cfg.TRAINER.UTL.PCA_CPS_NUM
        lcw_normal_init = cfg.TRAINER.UTL.NORMAL_INIT
        learnable = cfg.TRAINER.UTL.BASIS_LEARNABLE

        n_components = pca_components_num

        opl_prompt_prefix = " ".join(["X"] * n_opl_name)
        print(f'OPEN WORDS PCA COMPONENTS linear C: "{opl_prompt_prefix}"')
        print(f"PCA components : {n_components}")
        print(f"Number of open labels : {o_cls}")
        print(f"LENS of NAME : {n_opl_name}")

        open_vec_dir = os.path.join(cfg.DATASET.ROOT, "open_vec", str(cfg.TRAINER.UTL.OPEN_WORDS_NUM), str(n_opl_name),
                                    str(pca_components_num))
        mkdir_if_missing(open_vec_dir)
        open_vec_path = os.path.join(open_vec_dir, "open_vectors.txt")

        if os.path.exists(open_vec_path):
            print(f"Loading open words from {open_vec_path}")
            centers = np.loadtxt(open_vec_path)
        else:
            print('open words pca:')

            prompts_opl = [name for name in opennames_train]
            tokenized_prompts_opl = torch.cat([clip.tokenize(p) for p in prompts_opl])
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts_opl).type(torch.float32)

            opl_vecters = (embedding[:, 1: 1 + opl_name_len, :]).data.cpu().numpy()
            print(opl_vecters.shape)
            points = opl_vecters.reshape(opl_vecters.shape[0], 1, -1).squeeze()
            pca = PCA(n_components=n_components)
            pca.fit(points)
            print(np.sum(pca.explained_variance_ratio_))
            centers = pca.components_
            print(centers.shape)
            np.savetxt(open_vec_path, centers)
        print('open words pca:')

        # use the linear combination of given words' pca components as open_label vectors
        open_vectors = torch.from_numpy(centers).type(torch.float32).to("cuda")
        print(open_vectors.size())
        self.open_vectors = open_vectors
        if learnable:
            self.open_vectors = nn.Parameter(open_vectors)
        w_vectors = torch.empty(o_cls, n_components, dtype=torch.float32)
        print(w_vectors.size())
        if lcw_normal_init:
            print('linear combination weights matrix init: lcw_normal_init')
            torch.nn.init.normal_(w_vectors, std=0.02)
        self.open_w = nn.Parameter(w_vectors)
        self.o_cls = o_cls
        self.n_opl_name = n_opl_name
        self.opl_prompt_prefix = opl_prompt_prefix

    def forward(self):

        unknown_vector = torch.mm(self.open_w, self.open_vectors)
        unknown_vector = torch.reshape(unknown_vector, (-1, self.n_opl_name, 512))
        names = unknown_vector

        return names


class PromptLearner(nn.Module):

    def __init__(self, cfg, classnames, opennames_train, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.UTL.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # context random initialization
        if cfg.TRAINER.UTL.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        #######

        self.unknowntexts = UnknownTextsLearner(cfg, opennames_train, clip_model)
        self.opl_prompt_prefix = self.unknowntexts.opl_prompt_prefix

        classnames = [name.replace("_", " ") for name in classnames]
        prompts_n = [prompt_prefix + " " + name + "." for name in classnames]

        prompts_o = [prompt_prefix + " " + self.opl_prompt_prefix + "." for k in range(o_cls)]

        tokenized_prompts_n = torch.cat([clip.tokenize(p) for p in prompts_n])
        tokenized_prompts_o = torch.cat([clip.tokenize(p) for p in prompts_o])
        tokenized_prompts = torch.cat([tokenized_prompts_n, tokenized_prompts_o])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS


        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        o_cls = self.unknowntexts.o_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.co_cls = o_cls + n_cls
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.co_cls, -1, -1)


        names = self.unknowntexts()

        # print(len(names))
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":

            prompts = []
            for i in range(self.co_cls):
                prefix_i = prefix[i: i + 1, :, :]
                ctx_i = ctx[i: i + 1, :, :]
                if i < self.n_cls:
                    suffix_i = suffix[i: i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            ctx_i,  # (1, n_ctx, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                elif i >= self.n_cls and i < self.co_cls:
                    name_i = names[i - self.n_cls:i - self.n_cls + 1, :, :]
                    suffix_i = suffix[i: i + 1, self.n_opl_name:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (n_cls, 1, dim)
                            ctx_i,  # (n_cls, n_ctx, dim)
                            name_i,
                            suffix_i,  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                else:
                    print("error")
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, opennames_train, clip_model):
        super().__init__()
        self.prompt_learner = PromptopenLearner(cfg, classnames, opennames_train, clip_model)
        self.o_cls = self.prompt_learner.o_cls
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class UTL(TrainerX):
    """OPEN Context Optimization (CoOp).
    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.UTL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.UTL.PREC == "fp32" or cfg.TRAINER.UTL.PREC == "amp":
            clip_model.float()

        classnames = self.dm.dataset.classnames
        print(len(classnames))

        if cfg.TRAINER.UTL.OPW_PCA_LC:
            print("***************Loading UNKNOWN TEXT LEARNING FROM PCA COMPONENTS LC**********************************")
            WORD_SOURCE_NAME = ('WORD_SOURCE_IN21k')
            open_names_train_dir = os.path.join(cfg.DATASET.ROOT, WORD_SOURCE_NAME, )
            mkdir_if_missing(open_names_train_dir)
            open_names_train_path = os.path.join(open_names_train_dir, "open_train_name.txt")
            if os.path.exists(open_names_train_path):
                print(f"Loading open train names from {open_names_train_path}")
                with open(open_names_train_path, 'r') as file:
                    lines = file.readlines()

                opennames_train = [line.strip() for line in lines]
            else:
                opennames_train = open_word_select(open_names_train_path, cfg.TRAINER.UTL.WORD_SOURCE)

            if  cfg.TRAINER.UTL.OPEN_WORDS_NUM== 0:
                opennames_train = opennames_train
            else:
                opennames_train = np.array(opennames_train)
                opennames_train = np.random.choice(opennames_train, cfg.TRAINER.UTL.OPEN_WORDS_NUM, replace=False)
            opennames_train = (opennames_train).tolist()
            print('open words  pca init:', len(opennames_train))

        else:
            print('flag error')
            opennames_train = None

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, opennames_train, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        print("------------all para--------")
        print(sum(p.numel() for p in self.model.parameters()))
        print("--------learnable-----")
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.UTL.PREC == "amp" else None
        self.OPEN_LABEL_NUM = self.model.o_cls
        self.OPEN_LABEL_SMOOTHING = cfg.TRAINER.UTL.OLS

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        target = UnknownLabelSmoothing(target=label, open_label_num=self.OPEN_LABEL_NUM, \
                                       smoothing_open=self.OPEN_LABEL_SMOOTHING, num_classes=self.num_classes)

        prec = self.cfg.TRAINER.UTL.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, target)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            print('error')
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model.pth.tar-50"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def log_others_test(self, open_split=None, msp=False):

        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if open_split is None:
            print('open_split error')

        if open_split == "open_easy" and self.easy_loader is not None:
            open_loader = self.easy_loader

        if open_split == "open_hard" and self.hard_loader is not None:
            open_loader = self.hard_loader

        if open_split == "open_lt" and self.lt_loader is not None:
            open_loader = self.lt_loader

        if open_split == "open_nl" and self.nl_loader is not None:
            open_loader = self.nl_loader

        if open_split == "open_others" and self.others_loader is not None:
            open_loader = self.others_loader
        data_loader = self.test_loader

        pred_k, pred_u = [], []

        print(f"Evaluate on the *test* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            # print(output)
            if msp:
                # print("MSP")
                output = torch.nn.Softmax(dim=-1)(output)
                # print(output)
            output = output[:, :self.num_classes]
            # print(output)
            close_score = torch.max(output, 1)[0]
            # print(close_score)
            pred_k.append(close_score.data.cpu().numpy())

        pred_k = np.concatenate(pred_k, 0)
        print(pred_k.shape)

        print(f"Evaluate on the *{open_split}* set")
        for batch_idx, batch in enumerate(tqdm(open_loader)):
            input = self.parse_batch_open(batch)
            output = self.model_inference(input)
            if msp:
                output = torch.nn.Softmax(dim=-1)(output)
            output = output[:, :self.num_classes]
            # print(output)
            open_score = torch.max(output, 1)[0]
            # print(open_score)
            pred_u.append(open_score.data.cpu().numpy())

        pred_u = np.concatenate(pred_u, 0)

        close_score_path = self.cfg.OUTPUT_DIR + '/' + 'close_scores.txt'
        open_score_path = self.cfg.OUTPUT_DIR + '/' + 'open_scores.txt'

        np.savetxt(close_score_path, pred_k)
        np.savetxt(open_score_path, pred_u)

    @torch.no_grad()
    def log_close_feature(self, msp=False):

        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        data_loader = self.test_loader

        pred_k, label_k = [], []

        print(f"Evaluate on the *test* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            # print(output)
            if msp:
                # print("MSP")
                output = torch.nn.Softmax(dim=-1)(output)
                # print(output)
            output = output[:, :self.num_classes]

            pred_k.append(output.data.cpu().numpy())
            label_k.append(label.data.cpu().numpy())
        pred_k = np.concatenate(pred_k, 0)
        label_k = np.concatenate(label_k, 0)

        print(pred_k.shape)
        print(label_k.shape)

        close_score_path = self.cfg.OUTPUT_DIR + '/' + 'close_feature.txt'
        np.savetxt(close_score_path, pred_k)
        close_label_path = self.cfg.OUTPUT_DIR + '/' + 'close_label.txt'
        np.savetxt(close_label_path, label_k)

    @torch.no_grad()
    def log_open_feature(self, open_split=None, msp=False):

        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if open_split is None:
            print('open_split error')

        if open_split == "open_easy" and self.easy_loader is not None:
            open_loader = self.easy_loader

        if open_split == "open_hard" and self.hard_loader is not None:
            open_loader = self.hard_loader

        if open_split == "open_lt" and self.lt_loader is not None:
            open_loader = self.lt_loader

        if open_split == "open_nl" and self.nl_loader is not None:
            open_loader = self.nl_loader

        if open_split == "open_others" and self.others_loader is not None:
            open_loader = self.others_loader
        data_loader = self.test_loader

        pred_k, pred_u = [], []

        print(f"Evaluate on the *test* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            # print(output)
            if msp:
                # print("MSP")
                output = torch.nn.Softmax(dim=-1)(output)
                # print(output)
            output = output[:, :self.num_classes]
            pred_k.append(output.data.cpu().numpy())

        pred_k = np.concatenate(pred_k, 0)
        print(pred_k.shape)

        print(f"Evaluate on the *{open_split}* set")
        for batch_idx, batch in enumerate(tqdm(open_loader)):
            input = self.parse_batch_open(batch)
            output = self.model_inference(input)
            if msp:
                output = torch.nn.Softmax(dim=-1)(output)
            output = output[:, :self.num_classes]
            pred_u.append(output.data.cpu().numpy())

        pred_u = np.concatenate(pred_u, 0)

        close_score_path = self.cfg.OUTPUT_DIR + '/' + 'known_feature.txt'
        open_score_path = self.cfg.OUTPUT_DIR + '/' + 'unknown_feature.txt'

        np.savetxt(close_score_path, pred_k)
        np.savetxt(open_score_path, pred_u)