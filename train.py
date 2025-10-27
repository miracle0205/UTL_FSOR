import os
import argparse
import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_lt
import datasets.imagenet_200
import datasets.imagenet_500
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.cifar10_open
import datasets.cifar100_open
import datasets.cifar100_random_open
import datasets.cifar10_N_open
import datasets.tinyimagenet
import datasets.tinyimagenet_random
import datasets.tinyimagenet_H
import datasets.tinyimagenet_random_200
import datasets.miniimagenet
import datasets.tieredimagenet
import datasets.imagenet_OOD
import trainers.coop
import trainers.coop1
import trainers.cocoop
import trainers.utl
import trainers.outl
import trainers.cutl
import trainers.ocdputl
import trainers.ukdputl
import trainers.r_tuning
import trainers.coop_group
import trainers.zsclip
import trainers.r_ctt
import trainers.coop_group_open
import trainers.r_coop_group
import trainers.r_tuning_m
import trainers.coop_open_m
import trainers.coop_open_words_init
import trainers.gls_coop_open
import trainers.coop_openora
import trainers.coop_group_open_m
import trainers.r_coop_group_m
import trainers.r_coop_group_n
import trainers.coop_open_random_init
import trainers.utldp
import trainers.ulsgroup
import trainers.utlg
import trainers.ocdputlgroupk
import trainers.coopn
import trainers.ocdputlGeneralized
import time

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir
    if args.model_hdir:
        cfg.MODEL_HDIR = args.model_hdir
    if args.weight_dir:
        cfg.WEIGHT_DIR = args.weight_dir

def extend_cfg(cfg):
    """
    Add new config variables.
    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.COOP.NGROUP = 1
    cfg.TRAINER.COOP.SMOOTHING = 1.0

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CSC = False  # class-specific context
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.RCOOP = CN()
    cfg.TRAINER.RCOOP.OWTN = 0
    cfg.TRAINER.RCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.RCOOP.CSC = False  # class-specific context
    cfg.TRAINER.RCOOP.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.RCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SPLIT = 0
    cfg.DATASET.KNOWN_CLASS_NUM = 0
    cfg.DATASET.UNKNOWN_CLASS_NUM = 0

    cfg.TRAINER.UTL = CN()
    cfg.TRAINER.UTL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.UTL.CSC = False  # class-specific context
    cfg.TRAINER.UTL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.UTL.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.UTL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.UTL.UNKNOWN_TEXTS_NUM = 0
    cfg.TRAINER.UTL.ULSG = False
    cfg.TRAINER.UTL.GM = 0
    cfg.TRAINER.UTL.UNKNOWNTEXT_GROUP = 0
    cfg.TRAINER.UTL.N_UTX = 0  #
    cfg.TRAINER.UTL.NORMAL_INIT = False
    cfg.TRAINER.UTL.ORA_INIT = False

    cfg.TRAINER.UTL.OPW_PCA_LC = False
    cfg.TRAINER.UTL.PCA_CPS_NUM = 0
    cfg.TRAINER.UTL.OLS = 0.001
    cfg.TRAINER.UTL.OPEN_WORDS_NUM = 0

    cfg.TRAINER.UTL.UT_RANDOM_INIT = False
    cfg.TRAINER.UTL.BASIS_LEARNABLE = False
    cfg.TRAINER.UTL.WORD_SOURCE = None



    cfg.TRAINER.CUTL = CN()
    cfg.TRAINER.CUTL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.CUTL.CSC = False  # class-specific context
    cfg.TRAINER.CUTL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.CUTL.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.CUTL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.CUTL.UNKNOWN_TEXTS_NUM = 0
    cfg.TRAINER.CUTL.N_UTX = 0  #
    cfg.TRAINER.CUTL.NORMAL_INIT = False
    cfg.TRAINER.CUTL.ORA_INIT = False

    cfg.TRAINER.CUTL.OPW_PCA_LC = False
    cfg.TRAINER.CUTL.PCA_CPS_NUM = 0
    cfg.TRAINER.CUTL.OLS = 0.001
    cfg.TRAINER.CUTL.OPEN_WORDS_NUM = 0

    cfg.TRAINER.CUTL.UT_RANDOM_INIT = False
    cfg.TRAINER.CUTL.BASIS_LEARNABLE = False

    cfg.TRAINER.OUTL = CN()
    cfg.TRAINER.OUTL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.OUTL.CSC = False  # class-specific context
    cfg.TRAINER.OUTL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.OUTL.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.OUTL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.OUTL.UNKNOWN_TEXTS_NUM = 0
    cfg.TRAINER.OUTL.N_UTX = 0  #
    cfg.TRAINER.OUTL.NORMAL_INIT = False
    cfg.TRAINER.OUTL.ORA_INIT = False

    cfg.TRAINER.OUTL.OPW_PCA_LC = False
    cfg.TRAINER.OUTL.PCA_CPS_NUM = 0
    cfg.TRAINER.OUTL.OLS = 0.001
    cfg.TRAINER.OUTL.OPEN_WORDS_NUM = 0

    cfg.TRAINER.OUTL.UT_RANDOM_INIT = False
    cfg.TRAINER.OUTL.BASIS_LEARNABLE = False

    cfg.TRAINER.OCDPUTL = CN()
    cfg.TRAINER.OCDPUTL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.OCDPUTL.CSC = False  # class-specific context
    cfg.TRAINER.OCDPUTL.KGROUP = 1
    cfg.TRAINER.OCDPUTL.KNOWNG = False
    cfg.TRAINER.OCDPUTL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.OCDPUTL.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.OCDPUTL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.OCDPUTL.UNKNOWN_TEXTS_NUM = 0
    cfg.TRAINER.OCDPUTL.ULSG = False
    cfg.TRAINER.OCDPUTL.GM = 0
    cfg.TRAINER.OCDPUTL.UNKNOWNTEXT_GROUP = 0
    cfg.TRAINER.OCDPUTL.N_UTX = 0  #
    cfg.TRAINER.OCDPUTL.NORMAL_INIT = False
    cfg.TRAINER.OCDPUTL.ORA_INIT = False

    cfg.TRAINER.OCDPUTL.OPW_PCA_LC = False
    cfg.TRAINER.OCDPUTL.PCA_CPS_NUM = 0
    cfg.TRAINER.OCDPUTL.OLS = 0.001
    cfg.TRAINER.OCDPUTL.OPEN_WORDS_NUM = 0
    cfg.TRAINER.OCDPUTL.OPEN_CLOSE_RATIO = 0.5


    cfg.TRAINER.OCDPUTL.UT_RANDOM_INIT = False
    cfg.TRAINER.OCDPUTL.BASIS_LEARNABLE = False

    cfg.TRAINER.UKDPUTL = CN()
    cfg.TRAINER.UKDPUTL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.UKDPUTL.CSC = False  # class-specific context
    cfg.TRAINER.UKDPUTL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.UKDPUTL.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.UKDPUTL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.UKDPUTL.UNKNOWN_TEXTS_NUM = 0
    cfg.TRAINER.UKDPUTL.N_UTX = 0  #
    cfg.TRAINER.UKDPUTL.NORMAL_INIT = False
    cfg.TRAINER.UKDPUTL.ORA_INIT = False

    cfg.TRAINER.UKDPUTL.OPW_PCA_LC = False
    cfg.TRAINER.UKDPUTL.PCA_CPS_NUM = 0
    cfg.TRAINER.UKDPUTL.OLS = 0.001
    cfg.TRAINER.UKDPUTL.OPEN_WORDS_NUM = 0

    cfg.TRAINER.UKDPUTL.UT_RANDOM_INIT = False
    cfg.TRAINER.UKDPUTL.BASIS_LEARNABLE = False


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        print('*************************************')
        trainer.test()
        return
    if args.open_iNaturalist_test:
        print('open easy test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_iNaturalist', msp=True)
        return

    if args.open_iNaturalist_mls_test:
        print('open easy test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_iNaturalist', msp=False)
        return

    if args.open_OOD_test:
        print('open ood test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_OOD(open_split='open_iNaturalist', msp=True)
        return

    if args.open_OOD_mls_test:
        print('open ood test mls')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_OOD(open_split='open_iNaturalist', msp=False)
        return

    if args.open_energy_test:
        print('open ood test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_energy(open_split='open_iNaturalist')
        return

    if args.open_energy_mls_test:
        print('open ood test mls')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_energy(open_split='open_iNaturalist')
        return

    if args.open_easy_test:
        print('open easy test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_easy', msp=True)
        return

    if args.open_easy_mls_test:
        print('open easy test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_easy', msp=False)
        return

    if args.OOD_test:
        print('OOD test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.OOD_test(open_split='OOD_test', msp=False)
        return

    if args.open_hard_test:
        print('open hard test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_hard', msp=True)
        return

    if args.open_hard_mls_test:
        print('open hard test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_hard', msp=False)
        return

    if args.open_lt_test:
        print('open lt test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_lt', msp=True)
        return

    if args.open_lt_mls_test:
        print('open lt test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_lt', msp=False)
        return

    if args.open_nl_test:
        print('open nl test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_nl')
        return

    if args.open_others_test:
        print('open others test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_others', msp=True)
        return

    if args.open_others_mls_test:
        print('open others test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test(open_split='open_others', msp=False)
        return

    if args.log_others_test:
        print('log-other-score-test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.log_others_test(open_split='open_others', msp=True)
        return

    if args.log_close_feature:
        print('log-close-feature-test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.log_close_feature(msp=True)
        return

    if args.log_open_feature:
        print('log-open-feature-test')
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.log_open_feature(open_split='iNaturalist')
        return

    if not args.no_train:
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        print('the total time is {} seconds')
        print(end_time-start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/hy-tmp", help="path to dataset")
    parser.add_argument("--output-dir", type=str,
                        default="/root/CoOp-main/output/imagenet/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed0",
                        help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="/root/CoOp-main/configs/trainers/CoOp/rn50_ep50.yaml",
        help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="/root/CoOp-main/configs/datasets/imagenet.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="CoOp", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--open-easy-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-hard-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-lt-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-nl-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-easy-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-hard-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-lt-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-nl-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-others-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-others-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-iNaturalist-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-iNaturalist-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-Places-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-Places-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-SUN-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-SUN-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-Texture-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-Texture-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--OOD-test", action="store_true", help="evaluation only")
    parser.add_argument("--log-others-test", action="store_true", help="evaluation only")
    parser.add_argument("--log-close-feature", action="store_true", help="evaluation only")
    parser.add_argument("--log-open-feature", action="store_true", help="evaluation only")
    parser.add_argument("--open-OOD-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-OOD-mls-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-energy-test", action="store_true", help="evaluation only")
    parser.add_argument("--open-energy-mls-test", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/hy-tmp/CoOp-main/output/imagenet/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed0",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--model-hdir",
        type=str,
        default="/hy-tmp/CoOp-main/output/imagenet/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed0",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--weight-dir",
        type=str,
        default="/hy-tmp/CoOp-main/output/imagenet/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed0",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
