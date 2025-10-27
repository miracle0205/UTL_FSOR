import os.path as osp
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
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
from tqdm import tqdm
from dassl.evaluation import build_evaluator, metric_ood, compute_oscr
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
def open_description_access(split):
    descriptions = []
    text_name = '/data/gaofei/data/tinyimagenet/knowndescription' + str(split) +'.txt'
    # 打开文件并读取每一行
    with open(text_name, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用 strip() 方法移除每行末尾的换行符和可能的空白字符
            descriptions.append(line.strip())

    # 打印列表，查看内容
    print(descriptions)
    return descriptions
def open_word_access(open_words_num):
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
    select_open_list = np.array(select_open_list)
    if open_words_num == 0:
        select_open_list = select_open_list
    else:
        # print(open_words_num)
        select_open_list = np.random.choice(select_open_list, open_words_num, replace=False)
    open_list = (select_open_list).tolist()
    return open_list

# def UnknownLabelSmoothing(target=None, smoothing=0, smoothing_open=0, open_label_num=0, num_classes=1000):
#     close_target = torch.nn.functional.one_hot(target, num_classes=num_classes)
#     b_z = close_target.size()[0]
#     labels = ((1 - smoothing) * close_target) + (smoothing / num_classes)
#     open_target_init = torch.ones(open_label_num)
#     # smoothing_open = 0.001
#     open_target = smoothing_open * open_target_init
#     open_target = open_target.unsqueeze(0).expand(b_z, -1)
#     open_target = open_target.cuda()
#     target = torch.cat([labels, open_target], dim=1)
#     return target
def UnknownLabelSmoothing(target=None, smoothing=0, smoothing_open=0, open_label_num=0, num_classes=1000):
    close_target = torch.nn.functional.one_hot(target, num_classes=num_classes)
    b_z = close_target.size()[0]
    labels = ((1 - smoothing) * close_target) + (smoothing / num_classes)
    # 设置随机种子
    seed = 42  # 你可以选择任意整数作为种子
    torch.manual_seed(seed)

    # 随机生成一个从 0 到 smoothing_open 范围内的张量
    random_values = torch.rand(open_label_num) * smoothing_open
    # print(random_values)
    open_target_init = torch.ones(open_label_num)

    # 进行点乘
    open_target = random_values * open_target_init

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

# class D_PromptopenLearner(nn.Module):
#
#     def __init__(self, cfg, classnames, opennames_train, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.DPUTL.N_CTX
#         o_cls = cfg.TRAINER.DPUTL.UNKNOWN_TEXTS_NUM
#         n_opl_name = cfg.TRAINER.DPUTL.N_UTX
#         opl_name_len = n_opl_name
#         pca_components_num = cfg.TRAINER.DPUTL.PCA_CPS_NUM
#         lcw_normal_init = cfg.TRAINER.DPUTL.NORMAL_INIT
#         learnable = cfg.TRAINER.DPUTL.BASIS_LEARNABLE
#         n_components = pca_components_num
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
#
#         # context random initialization
#         if cfg.TRAINER.DPUTL.CSC:
#             print("Initializing class-specific contexts")
#             open_ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             close_ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#
#         else:
#             print("Initializing a generic context")
#             close_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             open_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(close_ctx_vectors, std=0.02)
#         nn.init.normal_(open_ctx_vectors, std=0.02)
#         prompt_prefix = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#         self.open_ctx = nn.Parameter(open_ctx_vectors)  # to be optimized
#         self.close_ctx = nn.Parameter(close_ctx_vectors)  # to be optimized
#
#         open_vec_dir = os.path.join(cfg.DATASET.ROOT, "open_vec",str(cfg.TRAINER.DPUTL.OPEN_WORDS_NUM), str(n_opl_name), str(pca_components_num))
#         mkdir_if_missing(open_vec_dir)
#         open_vec_path = os.path.join(open_vec_dir, "open_vectors.txt")
#
#         if os.path.exists(open_vec_path):
#             print(f"Loading open words from {open_vec_path}")
#             centers = np.loadtxt(open_vec_path)
#         else:
#             print('open words pca:')
#
#             prompts_opl = [name for name in opennames_train]
#             tokenized_prompts_opl = torch.cat([clip.tokenize(p) for p in prompts_opl])
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(tokenized_prompts_opl).type(torch.float32)
#
#             opl_vecters = (embedding[:, 1: 1 + opl_name_len, :]).data.cpu().numpy()
#             print(opl_vecters.shape)
#             points = opl_vecters.reshape(opl_vecters.shape[0], 1, -1).squeeze()
#             pca = PCA(n_components=n_components)
#             pca.fit(points)
#             print(np.sum(pca.explained_variance_ratio_))
#             centers = pca.components_
#             print(centers.shape)
#             np.savetxt(open_vec_path, centers)
#         print('open words pca:')
#
#         # use the linear combination of given words' pca components as open_label vectors
#         open_vectors = torch.from_numpy(centers).type(torch.float32).to("cuda")
#         print(open_vectors.size())
#         self.open_vectors = open_vectors
#         if learnable:
#             self.open_vectors = nn.Parameter(open_vectors)
#         w_vectors = torch.empty(o_cls, n_components, dtype=torch.float32)
#         print(w_vectors.size())
#         if lcw_normal_init:
#             print('linear combination weights matrix init: lcw_normal_init')
#             torch.nn.init.normal_(w_vectors, std=0.02)
#         self.open_w = nn.Parameter(w_vectors)
#
#         # unknown_vector = unknown_vector.data.cpu().numpy()
#         # unknown_vector = unknown_vector.reshape(-1, opl_name_len, 512)
#         # unknown_vector = torch.from_numpy(unknown_vector).to(dtype).to("cuda")
#         # print(unknown_vector.size())
#         #
#         # self.open_names = unknown_vector
#
#         opl_prompt_prefix = " ".join(["X"] * n_opl_name)
#         print(f'OPEN WORDS PCA COMPONENTS linear C: "{opl_prompt_prefix}"')
#         print(f"PCA components : {n_components}")
#         print(f"Number of open labels : {o_cls}")
#         print(f"LENS of NAME : {n_opl_name}")
#
#         classnames = [name.replace("_", " ") for name in classnames]
#         prompts_n = [prompt_prefix + " " + name + "." for name in classnames]
#         prompts_o = [prompt_prefix + " " + opl_prompt_prefix + "." for k in range(o_cls)]
#
#         tokenized_prompts_n = torch.cat([clip.tokenize(p) for p in prompts_n])
#         tokenized_prompts_o = torch.cat([clip.tokenize(p) for p in prompts_o])
#         open_tokenized_prompts = torch.cat([tokenized_prompts_n, tokenized_prompts_o])
#         with torch.no_grad():
#             open_embedding = clip_model.token_embedding(open_tokenized_prompts).type(dtype)
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("open_token_prefix", open_embedding[:, :1, :])  # SOS
#         self.register_buffer("open_token_suffix", open_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#
#         close_tokenized_prompts = tokenized_prompts_n
#         with torch.no_grad():
#             close_embedding = clip_model.token_embedding(close_tokenized_prompts).type(dtype)
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("close_token_prefix", open_embedding[:, :1, :])  # SOS
#         self.register_buffer("close_token_suffix", open_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.o_cls = o_cls
#         self.n_ctx = n_ctx
#         self.ctx_dim = ctx_dim
#         self.n_opl_name = n_opl_name
#         self.open_tokenized_prompts = open_tokenized_prompts  # torch.Tensor
#         self.close_tokenized_prompts = close_tokenized_prompts  # torch.Tensor
#         self.co_cls = o_cls + n_cls
#         self.class_token_position = "end"
#
#
#     def forward(self):
#         open_ctx = self.open_ctx
#         if open_ctx.dim() == 2:
#             open_ctx = open_ctx.unsqueeze(0).expand(self.co_cls, -1, -1)
#
#         unknown_vector = torch.mm(self.open_w, self.open_vectors)
#         unknown_vector = torch.reshape(unknown_vector, (-1, self.n_opl_name, 512))
#         names = unknown_vector
#
#         # print(len(names))
#         open_prefix = self.open_token_prefix
#         open_suffix = self.open_token_suffix
#
#         close_ctx = self.open_ctx
#         if close_ctx.dim() == 2:
#             close_ctx = close_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#
#         close_prefix = self.close_token_prefix
#         close_suffix = self.close_token_suffix
#
#         if self.class_token_position == "end":
#
#             open_prompts = []
#             for i in range(self.co_cls):
#                 prefix_i = open_prefix[i: i + 1, :, :]
#                 ctx_i = open_ctx[i: i + 1, :, :]
#                 if i < self.n_cls:
#                     suffix_i = open_suffix[i: i + 1, :, :]
#                     open_prompt = torch.cat(
#                         [
#                             prefix_i,  # (1, 1, dim)
#                             ctx_i,  # (1, n_ctx, dim)
#                             suffix_i,  # (1, *, dim)
#                         ],
#                         dim=1,
#                     )
#                 elif i >= self.n_cls and i < self.co_cls:
#                     name_i = names[i - self.n_cls:i - self.n_cls + 1, :, :]
#                     suffix_i = open_suffix[i: i + 1, self.n_opl_name:, :]
#                     open_prompt = torch.cat(
#                         [
#                             prefix_i,  # (n_cls, 1, dim)
#                             ctx_i,  # (n_cls, n_ctx, dim)
#                             name_i,
#                             suffix_i,  # (n_cls, *, dim)
#                         ],
#                         dim=1,
#                     )
#                 else:
#                     print("error")
#                 open_prompts.append(open_prompt)
#             open_prompts = torch.cat(open_prompts, dim=0)
#
#
#             close_prompts = torch.cat(
#                 [
#                     close_prefix,  # (n_cls, 1, dim)
#                     close_ctx,  # (n_cls, n_ctx, dim)
#                     close_suffix,  # (n_cls, *, dim)
#                 ],
#                 dim=1,
#             )
#
#
#         else:
#             raise ValueError
#         return open_prompts, close_prompts

class UnknownTextsLearner(nn.Module):
    def __init__(self, cfg, opennames_train, classnames, clip_model):
        super().__init__()

        o_cls = cfg.TRAINER.OCDPUTL.UNKNOWN_TEXTS_NUM
        n_opl_name = cfg.TRAINER.OCDPUTL.N_UTX
        opl_name_len = n_opl_name
        pca_components_num = cfg.TRAINER.OCDPUTL.PCA_CPS_NUM
        lcw_normal_init = cfg.TRAINER.OCDPUTL.NORMAL_INIT
        learnable = cfg.TRAINER.OCDPUTL.BASIS_LEARNABLE
        n_components = pca_components_num
        open_words = cfg.TRAINER.OCDPUTL.OPEN_WORDS

        opl_prompt_prefix = " ".join(["X"] * n_opl_name)
        print(f'OPEN WORDS PCA COMPONENTS linear C: "{opl_prompt_prefix}"')
        print(f"PCA components : {n_components}")
        print(f"Number of open labels : {o_cls}")
        print(f"LENS of NAME : {n_opl_name}")

        if open_words:
            open_vec_dir = os.path.join(cfg.DATASET.ROOT, "open_vec", str(cfg.TRAINER.OCDPUTL.OPEN_WORDS_NUM),
                                        str(n_opl_name), str(pca_components_num))

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
        else:
            print('known words pca:')
            open_vec_dir = os.path.join(cfg.DATASET.ROOT, "known_vec", str(1000),
                                        str(n_opl_name), str(pca_components_num))

            mkdir_if_missing(open_vec_dir)
            open_vec_path = os.path.join(open_vec_dir, "known_vectors.txt")

            if os.path.exists(open_vec_path):
                print(f"Loading known words from {open_vec_path}")
                centers = np.loadtxt(open_vec_path)
            else:
                print('known words pca:')

                prompts_opl = [name for name in classnames]
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
            print('known words pca:')
        # use the linear combination of given words' pca components as open_label vectors
        open_vectors = torch.from_numpy(centers).type(torch.float32).to("cuda")
        print(open_vectors.size())
        self.open_vectors = open_vectors.to("cuda")
        if learnable:
            self.open_vectors = nn.Parameter(open_vectors)

        # prompts_opl = [name for name in opennames_train]
        # tokenized_prompts_opl = torch.cat([clip.tokenize(p) for p in prompts_opl])
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts_opl).type(torch.float32)
        # opl_vecters = (embedding[:, 1: 1 + opl_name_len, :]).data.cpu().numpy()
        # print(opl_vecters.shape)
        # points = opl_vecters.reshape(opl_vecters.shape[0], 1, -1).squeeze()
        # print(points.shape)
        # open_vectors = torch.from_numpy(points).type(torch.float32).to("cuda")
        # print(open_vectors.size())
        # self.open_vectors = open_vectors.to("cuda")
        # n_components = points.shape[0]
        #*****************************************************
        w_vectors = torch.empty(o_cls, n_components, dtype=torch.float32)
        print(w_vectors.size())
        if lcw_normal_init:
            print('linear combination weights matrix init: lcw_normal_init')
            torch.nn.init.normal_(w_vectors, std=0.02)
        else:
            print('linear combination weights matrix init: lcw_orthogonal_init')
            nn.init.orthogonal_(w_vectors)
        self.open_w = nn.Parameter(w_vectors)

        self.o_cls = o_cls
        self.n_opl_name = n_opl_name
        self.opl_prompt_prefix = opl_prompt_prefix

    def forward(self):
        self.open_vectors = self.open_vectors.to(self.open_w.device)
        unknown_vector = torch.mm(self.open_w,self.open_vectors)
        unknown_vector = torch.reshape(unknown_vector, (-1, self.n_opl_name, 512))
        names = unknown_vector

        return names

class PromptopenLearner(nn.Module):

    def __init__(self, cfg, classnames, opennames_train, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.OCDPUTL.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # context random initialization
        # if cfg.TRAINER.OCDPUTL.CSC:
        #     print("Initializing class-specific contexts")
        #     ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        # else:
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        #######

        self.unknowntexts = UnknownTextsLearner(cfg, opennames_train, classnames, clip_model)
        self.opl_prompt_prefix = self.unknowntexts.opl_prompt_prefix
        o_cls = self.unknowntexts.o_cls
        self.n_opl_name = self.unknowntexts.n_opl_name
        self.o_cls = o_cls
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

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.OCDPUTL.N_CTX
        ctx_init = cfg.TRAINER.OCDPUTL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.OCDPUTL.CSC:
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

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts

class D_PromptLearner(nn.Module):

    def __init__(self, cfg, classnames, opennames_train, clip_model):
        super().__init__()
        self.openpromptlearner = PromptopenLearner(cfg, classnames, opennames_train, clip_model)
        self.closepromptlearner = PromptLearner(cfg, classnames, clip_model)
    def forward(self):
        open_prompts = self.openpromptlearner()
        close_prompts = self.closepromptlearner()
        return open_prompts, close_prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, opennames_train, clip_model):
        super().__init__()
        self.prompt_learner = D_PromptLearner(cfg, classnames, opennames_train, clip_model)
        self.o_cls = self.prompt_learner.openpromptlearner.o_cls
        self.c_cls = self.prompt_learner.closepromptlearner.n_cls
        self.open_tokenized_prompts = self.prompt_learner.openpromptlearner.tokenized_prompts
        self.close_tokenized_prompts = self.prompt_learner.closepromptlearner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        open_prompts, close_prompts = self.prompt_learner()

        open_tokenized_prompts = self.open_tokenized_prompts
        open_text_features = self.text_encoder(open_prompts, open_tokenized_prompts)
        open_text_features = open_text_features / open_text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        open_logits = logit_scale * image_features @ open_text_features.t()

        close_tokenized_prompts = self.close_tokenized_prompts
        close_text_features = self.text_encoder(close_prompts, close_tokenized_prompts)
        close_text_features = close_text_features / close_text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        close_logits = logit_scale * image_features @ close_text_features.t()
        return open_logits, close_logits


@TRAINER_REGISTRY.register()
class OCDPUTL(TrainerX):
    """OPEN Context Optimization (CoOp).
    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.OCDPUTL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.OCDPUTL.PREC == "fp32" or cfg.TRAINER.OCDPUTL.PREC == "amp":
            clip_model.float()

        classnames = self.dm.dataset.classnames
        print(len(classnames))
        self.num_class = len(classnames)
        self.split = cfg.DATASET.SPLIT
        if cfg.TRAINER.OCDPUTL.OPW_PCA_LC:
            opennames_train = open_word_access(cfg.TRAINER.OCDPUTL.OPEN_WORDS_NUM)
            print('open words  pca init:', len(opennames_train))
            # opennames_train = open_description_access(self.split)
            # print('open words  pca init:', len(opennames_train))

        else:
            print('flag error')
            opennames_train = None

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, opennames_train, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)
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
        self.scaler = GradScaler() if cfg.TRAINER.OCDPUTL.PREC == "amp" else None
        self.OPEN_LABEL_NUM = self.cfg.TRAINER.OCDPUTL.UNKNOWN_TEXTS_NUM
        self.OPEN_LABEL_SMOOTHING = cfg.TRAINER.OCDPUTL.OLS
        self.open_close_ratio = cfg.TRAINER.OCDPUTL.OPEN_CLOSE_RATIO
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

        prec = self.cfg.TRAINER.OCDPUTL.PREC
        if prec == "amp":
            with autocast():
                open_logits, close_logits = self.model(image)

                loss1 = F.cross_entropy(open_logits, target)
                loss2 = F.cross_entropy(close_logits, label)
                lamda = self.open_close_ratio

                loss = (1 - lamda) * loss1 + lamda * loss2
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
            "acc": compute_accuracy(close_logits, label)[0].item(),
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
        model_file = 'model-best.pth.tar'

        # if epoch is not None:
        #     model_file = "model.pth.tar-" + str(epoch)

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
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        _, logits = self.model(input)
        return logits

    def open_model_inference(self, input):
        logits, _ = self.model(input)
        return logits

    @torch.no_grad()
    def open_test(self, open_split=None, msp=False):
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
        _pred_k, _pred_u, _labels = [], [], []
        _target = []

        print(f"Evaluate on the *test* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.open_model_inference(input)
            # print(output)
            if msp:
                # print("MSP")
                output = torch.nn.Softmax(dim=-1)(output)
                # print(output)
            output = output[:, :self.num_classes]
            _pred_k.append(output.data.cpu().numpy())
            _labels.append(label.data.cpu().numpy())

        print(f"Evaluate on the *{open_split}* set")
        for batch_idx, batch in enumerate(tqdm(open_loader)):
            input = self.parse_batch_open(batch)
            output = self.open_model_inference(input)
            if msp:
                output = torch.nn.Softmax(dim=-1)(output)
            output = output[:, :self.num_classes]
            _pred_u.append(output.data.cpu().numpy())

        _pred_k = np.concatenate(_pred_k, 0)
        _pred_u = np.concatenate(_pred_u, 0)
        _labels = np.concatenate(_labels, 0)
        print(_pred_k.shape)
        print(_pred_k.shape)

        x1 = np.max(_pred_k, axis=1)
        x2 = np.max(_pred_u, axis=1)

        results = metric_ood(x1, x2)['Bas']
        # OSCR
        _oscr_socre = compute_oscr(_pred_k, _pred_u, _labels)
        results['OSCR'] = _oscr_socre * 100.
        if msp:
            print(f"openset preformance MSP *{msp}* on the *{open_split}* set")
            print(" AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['AUROC'], results['OSCR']))
        else:
            print(f"openset preformance MLS on the *{open_split}* set")
            print(" AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['AUROC'], results['OSCR']))

    # @torch.no_grad()
    # def OOD_test(self, open_split=None, msp=False):
    #     """A generic testing pipeline."""
    #     self.set_model_mode("eval")
    #     self.evaluator.reset()
    #
    #     if open_split is None:
    #         print('open_split error')
    #
    #     if open_split == "OOD_test" and self.others_loader is not None:
    #         open_loader = self.others_loader
    #     data_loader = self.test_loader
    #     score_in, score_out, = [], []
    #
    #     print(f"Evaluate on the *test* set")
    #     for batch_idx, batch in enumerate(tqdm(data_loader)):
    #         input, label = self.parse_batch_test(batch)
    #         output = self.open_model_inference(input)
    #         # print(output)
    #         if msp:
    #             # print("MSP")
    #             output = torch.nn.Softmax(dim=-1)(output)
    #             # print(output)
    #         output = output[:, :self.num_classes]
    #         output = output.data.cpu().numpy()
    #         scores_i = np.max(output, axis=1, keepdims=True)
    #         print(scores_i)
    #
    #         score_i = torch.max
    #
    #     print(f"Evaluate on the *{open_split}* set")
    #     for batch_idx, batch in enumerate(tqdm(open_loader)):
    #         input = self.parse_batch_open(batch)
    #         output = self.open_model_inference(input)
    #         if msp:
    #             output = torch.nn.Softmax(dim=-1)(output)
    #         output = output[:, :self.num_classes]
    #
    #         .append(output.data.cpu().numpy())


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

        pred_c, pred_k, pred_u = [], [], []

        print(f"Evaluate on the *open test* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.open_model_inference(input)
            # print(output)
            if msp:
                # print("MSP")
                output = torch.nn.Softmax(dim=-1)(output)
                # print(output)
            output = output[:, :self.num_classes]
            open_known_score = torch.max(output, 1)[0]
            pred_k.append(open_known_score.data.cpu().numpy())

        pred_k = np.concatenate(pred_k, 0)
        print(pred_k.shape)

        print(f"Evaluate on the *open{open_split}* set")
        for batch_idx, batch in enumerate(tqdm(open_loader)):
            input = self.parse_batch_open(batch)
            output = self.open_model_inference(input)
            if msp:
                output = torch.nn.Softmax(dim=-1)(output)
            output = output[:, :self.num_classes]
            open_score = torch.max(output, 1)[0]
            pred_u.append(open_score.data.cpu().numpy())

        pred_u = np.concatenate(pred_u, 0)

        print(f"Evaluate on the *close* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if msp:
                output = torch.nn.Softmax(dim=-1)(output)
            output = output[:, :self.num_classes]
            # print(output)
            close_score = torch.max(output, 1)[0]
            # print(open_score)
            pred_c.append(close_score.data.cpu().numpy())
        pred_c = np.concatenate(pred_c, 0)

        close_score_path = self.cfg.OUTPUT_DIR + '/' + 'close_scores.txt'
        open_score_path = self.cfg.OUTPUT_DIR + '/' + 'open_scores.txt'
        acc_score_path = self.cfg.OUTPUT_DIR + '/' + 'acc.txt'

        np.savetxt(close_score_path, pred_k)
        np.savetxt(open_score_path, pred_u)
        np.savetxt(acc_score_path, pred_c)

    @torch.no_grad()
    def log_close_feature(self, msp=False):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        data_loader = self.test_loader
        pred_c, label_c = [], []
        print(f"Evaluate on the *close* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if msp:
                output = torch.nn.Softmax(dim=-1)(output)
            output = output[:, :self.num_classes]
            pred_c.append(output.data.cpu().numpy())
            label_c.append(label.data.cpu().numpy())
        pred_c = np.concatenate(pred_c, 0)
        label_c = np.concatenate(label_c, 0)
        print(pred_c.shape)
        acc_score_path = self.cfg.OUTPUT_DIR + '/' + 'close_feature.txt'
        np.savetxt(acc_score_path, pred_c)
        label_path = self.cfg.OUTPUT_DIR + '/' + 'close_label.txt'
        np.savetxt(label_path, label_c)

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

        pred_c, pred_k, pred_u = [], [], []

        print(f"Evaluate on the *open test* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.open_model_inference(input)
            # print(output)
            if msp:
                # print("MSP")
                output = torch.nn.Softmax(dim=-1)(output)
                # print(output)
            output = output[:, :self.num_classes]

            pred_k.append(output.data.cpu().numpy())

        pred_k = np.concatenate(pred_k, 0)
        print(pred_k.shape)

        print(f"Evaluate on the *open{open_split}* set")
        for batch_idx, batch in enumerate(tqdm(open_loader)):
            input = self.parse_batch_open(batch)
            output = self.open_model_inference(input)
            if msp:
                output = torch.nn.Softmax(dim=-1)(output)
            output = output[:, :self.num_classes]
            pred_u.append(output.data.cpu().numpy())

        pred_u = np.concatenate(pred_u, 0)
        print(pred_u.shape)

        close_score_path = self.cfg.OUTPUT_DIR + '/' + 'known_feature.npy'
        open_score_path = self.cfg.OUTPUT_DIR + '/' + 'unknown_feature.npy'

        np.save(close_score_path, pred_k)
        np.save(open_score_path, pred_u)