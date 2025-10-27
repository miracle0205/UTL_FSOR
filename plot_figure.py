import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 可选：支持中文字体（如果你用中文图例或标题）
import matplotlib as mpl


# —— 全局字体与字号（罗马体 / Times 系）——
mpl.rcParams.update({
    # 字体族：按顺序匹配，系统缺哪个就用下一个
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "STIXGeneral", "DejaVu Serif"],

    # 坐标轴标题与刻度字号
    "axes.titlesize": 20,     # 图标题
    "axes.labelsize": 20,     # x/y 轴标签（文字）
    "xtick.labelsize": 20,    # x 轴刻度数字
    "ytick.labelsize": 20,    # y 轴刻度数字
    "legend.fontsize": 12,    # 图例字号（若图例开启）
})

# 负号正常显示
mpl.rcParams["axes.unicode_minus"] = False

#（可选）线条与标记更清晰
mpl.rcParams["lines.linewidth"] = 2.0
mpl.rcParams["lines.markersize"] = 6.5

matplotlib.rcParams['axes.unicode_minus'] = False
# —— 统一图尺寸 —— #
LINE_FIGSIZE = (7.6, 3.5)   # 折线图：宽、矮（可再调成 8.2 x 2.4）
BAR_FIGSIZE  = (5.2, 3.6)   # 柱状图
BAR3D_FIGSIZE = (6.8, 4.0)  # 3D 柱状图


# —— 统一样式映射（大小写不敏感）—— #
STYLE_MAP = {
    'stanford cars': {'color': '#1f77b4', 'marker': 'o'},  # 蓝 + 圆
    'cub':           {'color': '#ff7f0e', 'marker': '^'},  # 橙 + 三角
}
DEFAULT_STYLE = {'marker': 'o'}  # 其它系列默认样式

# —— 不同子图的横坐标文字 —— #
XLABEL_MAP = {
    "exp2": "Number of Experts",
    "exp3": "Number of Experts",
    "exp4": r"Value of $\gamma$",      # γ
    "exp5": r"Value of $\gamma$",      # γ
    "exp6": "Threshold of Entropy",
    "exp7": "Threshold of Entropy",
    # 其它没列出的 exp* 若需要也可继续加
}

# ========= 实验数据集中管理（原样使用你给的数据） =========
exp_data = {
    "exp1": {
        "title": "Exp1: Module Effectiveness",
        "type": "line",  # 改为折线图
        "x": ['Parameter-based', 'PM-MOE', 'OT-based', 'CAMoL'],
        "series": {
            'Stanford Cars': [61.19, 64.02, 64.37, 65.70],
            'CUB': [74.28, 74.42, 75.06, 75.38],
        },
        "ylim": (74, 75.5),          # 你给的范围
        "show_labels": False,
        "show_legend": False,
    },
    "exp2": {
        "title": "Exp2: #Experts vs Accuracy",
        "type": "line",
        "x": [2, 3, 4, 5],
        'Stanford Cars': [65.55, 65.70, 65.14, 64.54],  # 单系列的写法（无 series 字段）
        "ylim": (64, 66),
        "show_labels": False,
        "yticks_n": 3,
        "show_legend": False,
    },
    "exp3": {
        "title": "Exp2: #Experts vs Accuracy",
        "type": "line",
        "x": [2, 3, 4, 5],
        'CUB': [74.62, 75.18, 75.38, 74.88],            # 单系列
        "ylim": (74.5, 75.5),
        "show_labels": False,
        "show_legend": False,
    },
    "exp4": {
        "title": "Exp3: Partial OT Forward Ratio",
        "type": "line",
        "x": [1, 0.8, 0.6, 0.4],
        'Stanford Cars': [64.28, 65.7, 63.94, 63.94],   # 单系列
        "ylim": (63, 66),
        "show_labels": False,
        "yticks_n": 3,
        "show_legend": False,
    },
    "exp5": {
        "title": "Exp3: Partial OT Forward Ratio",
        "type": "line",
        "x": [1, 0.8, 0.6, 0.4],
        'CUB': [75.04, 75.38, 75.08, 75.09],           # 单系列
        "ylim": (75, 75.5),
        "show_labels": False,
        "yticks_n": 3,
        "show_legend": False,
    },
    "exp6": {
        "title": "Exp4: Shrink Value",
        "type": "line",
        "x": [0, 20, 40, 60],
        'Stanford Cars': [63.93, 63.74, 65.70, 64.07],  # 单系列
        "ylim": (63, 66),
        "show_labels": False,
        "yticks_n": 3,
        "show_legend": False,
    },
    "exp7": {
        "title": "Exp4: Shrink Value",
        "type": "line",
        "x": [0, 20, 40, 60],
        'CUB': [75.12, 75.06, 75.38, 74.97],           # 单系列
        "ylim": (74.80, 75.6),
        "show_labels": False,
        "show_legend": False,
    },
    "exp8": {
    "title": "Exp5: Trainable Config vs #Experts",
    "type": "3dbar",
    "x": [2, 3, 4, 5],
    "series": {
        "Fix": [65.55, 65.7, 65.14, 64.54],
        "FC":   [64.22, 64.1, 63.07, 63.67],
        "DFC":  [62.504, 62.703, 63.06, 62.23],
    },
    "y_order": ["DFC", "FC", "Fix"],  # ✅ gate 在最里面
    "zlim": (61, 66),                  # ✅ 或者直接用你改过的固定 z 轴范围
    }
}

# ========= 工具函数 =========

def auto_title_if_single_series(cfg, series):
    # 若未显式指定 use_series_title=False，且只有一条曲线，则用系列名做标题
    if cfg.get("use_series_title", True) and len(series) == 1:
        return next(iter(series.keys()))
    return cfg.get("title", "")

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def apply_y_ticks(ax, ylim=None, yticks=None, yticks_n=None):
    # 先应用 ylim
    if ylim is not None:
        ax.set_ylim(*ylim)
    # 再设置刻度
    if yticks is not None:
        ax.set_yticks(yticks)
    elif yticks_n is not None:
        lo, hi = ax.get_ylim()
        ax.set_yticks(np.linspace(lo, hi, yticks_n))


def format_axes(ax, title="", ylabel="Accuracy (%)", xlabel=None, ylim=None, show_legend=True):
    # 取消背景网格
    ax.grid(False)

    ax.set_title(title,)
    ax.set_ylabel(ylabel, )
    if xlabel is not None:
        ax.set_xlabel(xlabel)  # ← 新增：设置横坐标说明


    ax.tick_params(axis='both')
    if show_legend:
        ax.legend(loc='upper right', fontsize=10, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)


def annotate_bars(ax, rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

def plot_grouped_bar(ax, x_labels, series_dict, title, show_labels=True, show_legend=True, ylim=None, bar_width=0.2):
    x = np.arange(len(x_labels))
    total_width = bar_width * len(series_dict)
    start = x - (total_width - bar_width) / 2
    for i, (name, values) in enumerate(series_dict.items()):
        offset = i * bar_width
        rects = ax.bar(start + offset, values, width=bar_width, label=name)
        if show_labels:
            annotate_bars(ax, rects)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    format_axes(ax, title, ylim=ylim, show_legend=show_legend)

from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_line(ax, x_vals, series_dict, title, ylim=None, show_labels=True, show_legend=True,yticks=None, yticks_n=None):
    for name, y_vals in series_dict.items():
        key = str(name).lower().strip()  # 名称标准化
        style = STYLE_MAP.get(key, DEFAULT_STYLE)
        ax.plot(
            x_vals, y_vals,
            linewidth=2,
            label=name,
            marker=style.get('marker', 'o'),
            color=style.get('color', None)  # 折线与标记都用该颜色
        )
        if show_labels:
            for xv, yv in zip(x_vals, y_vals):
                ax.annotate(f'{yv:.2f}', xy=(xv, yv), xytext=(0, 4),
                            textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_vals)
    # Y 轴只保留约 5 个刻度
    apply_y_ticks(ax, ylim=ylim, yticks=yticks, yticks_n=yticks_n)

    format_axes(ax, title, xlabel=None, ylim=ax.get_ylim(), show_legend=show_legend)


def save_fig(fig, name, outdir='figs'):
    ensure_dir(outdir)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {name}.pdf and .png")

# —— 单系列折线图（供拆分用；当前我们不拆分，直接画在一张里即可） —— #
def plot_single_series_line(x_vals, y_vals, label, title, ylim=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(x_vals, y_vals, marker='o', linewidth=2, label=label)
    ax.set_xticks(x_vals)
    format_axes(ax, title=f"{title} - {label}", ylim=ylim, show_legend=True)
    return fig

# —— 3D 柱状图（X: experts, Y: configs, Z: value） —— #
def plot_3d_bar(exp_key, cfg):
    x_vals = cfg["x"]
    series = cfg["series"]
    y_labels = cfg.get("y_order", list(series.keys()))
    y_positions = np.arange(len(y_labels))

    width, depth = 0.5, 0.5
    zmin, zmax = 61.0, 66.0

    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111, projection='3d')

    # --- bars ---
    for yi, ylab in enumerate(y_labels):
        xs = np.array(x_vals, dtype=float)
        ys = np.full_like(xs, yi, dtype=float)
        zs_abs = np.array(series[ylab], dtype=float)
        z0 = np.full_like(xs, zmin, dtype=float)
        dz = zs_abs - zmin
        ax.bar3d(xs - width/2, ys - depth/2, z0, width, depth, dz, shade=True)

    # --- axis & ticks (基础设置) ---
    ax.set_xlabel('#Experts')
    ax.set_ylabel('Config')
    ax.set_zlabel('Accuracy (%)')

    ax.set_xticks(x_vals)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_zlim(zmin, zmax)

    # ===== 在这里加入“贴近轴”的设置（放在 view_init 之前）=====
    # 1) 调 3D 轴内部的距离系数
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo['label']['space_factor'] = 0.5   # 越小越靠近（0.7~0.9 试）
            axis._axinfo['label']['pad'] = 0
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
        except Exception:
            pass

    # 2) 再把 pad 调小（必要时可用负值）
    ax.xaxis.labelpad = -4
    ax.yaxis.labelpad = -4
    ax.zaxis.labelpad = 0
    ax.tick_params(axis='x', pad=-4)
    ax.tick_params(axis='y', pad=-4)
    ax.tick_params(axis='z', pad=0)

    # 3) 仅对 3D 图把字号调得适中，避免大字号造成“漂移感”
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel('Experts', fontsize=15)
    ax.set_ylabel('Config',   fontsize=15)
    ax.set_zlabel('Accuracy (%)', fontsize=15)

    # 4) 正交投影减少透视偏移
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass
    # ===== 贴近设置结束 =====

    # 视角与比例
    ax.view_init(elev=22, azim=-60)
    try:
        ax.set_box_aspect((len(x_vals), len(y_labels), (zmax - zmin)))
    except Exception:
        pass

    ax.set_title(cfg["title"], pad=4)
    save_fig(fig, exp_key)


# def plot_3d_bar(exp_key, cfg):
#     ...
#     fig = plt.figure(figsize=(6.2, 4.2))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # —— 柱子绘制（略）——
#
#     # 轴与刻度
#     ax.set_xlabel('#Experts')
#     ax.set_ylabel('Config')
#     ax.set_zlabel('Accuracy (%)')
#
#     ax.set_xticks(x_vals)
#     ax.set_yticks(y_positions)
#     ax.set_yticklabels(y_labels)
#     ax.set_zlim(zmin, zmax)
#
#     # ✅ 关键：缩小标签与刻度文字的“pad”（靠近坐标轴）
#     ax.xaxis.labelpad = 4     # 默认随字号偏大；改小
#     ax.yaxis.labelpad = 4
#     ax.zaxis.labelpad = 6
#     ax.tick_params(axis='x', pad=1)  # 刻度数字到轴的距离
#     ax.tick_params(axis='y', pad=1)
#     ax.tick_params(axis='z', pad=1)
#
#     # ✅ 用正交投影减少透视导致的偏移感（Matplotlib >=3.2）
#     try:
#         ax.set_proj_type('ortho')
#     except Exception:
#         pass
#
#     # 视角和比例
#     ax.view_init(elev=22, azim=-60)
#     try:
#         ax.set_box_aspect((len(x_vals), len(y_labels), (zmax - zmin)))
#     except Exception:
#         pass
#
#     ax.set_title(cfg["title"], pad=6)  # 也把标题 pad 收小些
#     save_fig(fig, exp_key)



# —— 规范化：把“单系列键写法”转换成 series={...} —— #
RESERVED_KEYS = {"title", "type", "x", "series", "ylim", "zlim", "show_labels", "show_legend", "bar_width"}

def normalize_series(cfg):
    if "series" in cfg and isinstance(cfg["series"], dict):
        return cfg["series"]
    # 把除保留键外的所有键当作单系列曲线
    series = {}
    for k, v in cfg.items():
        if k not in RESERVED_KEYS and isinstance(v, (list, tuple, np.ndarray)):
            series[k] = v
    if not series:
        raise ValueError("No data series found. Please provide 'series' dict or single-series entries.")
    return series

# ========= 主函数 =========
def plot_all_exp(data_dict):
    for key, cfg in data_dict.items():
        plot_type = cfg.get("type", "line")

        if plot_type == "bar":
            fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
            series = normalize_series(cfg)
            plot_grouped_bar(ax, cfg["x"], series, cfg["title"],
                             show_labels=cfg.get("show_labels", True),
                             show_legend=cfg.get("show_legend", True),
                             ylim=cfg.get("ylim", None),
                             bar_width=cfg.get("bar_width", 0.2))
            save_fig(fig, key)

        elif plot_type == "line":
            fig, ax = plt.subplots(figsize=LINE_FIGSIZE)   # ✅ 折线图用更“扁”的尺寸
            series = normalize_series(cfg)
            title = auto_title_if_single_series(cfg, series)

            xlabel = XLABEL_MAP.get(key, None)
            plot_line(ax, cfg["x"], series, title,
                      ylim=cfg.get("ylim", None),
                      show_labels=cfg.get("show_labels", True),
                      show_legend=cfg.get("show_legend", True),
                      yticks=cfg.get("yticks", None),
                      yticks_n=cfg.get("yticks_n", None))
            # 覆盖设置 xlabel（与 format_axes 解耦，方便按图定制）
            if xlabel is not None:
                ax.set_xlabel(xlabel)

            save_fig(fig, key)

        elif plot_type == "3dbar":
            fig = plot_3d_bar(key, cfg)  # 若你返回 fig，就不需尺寸；否则在函数里用 BAR3D_FIGSIZE
        else:
            raise ValueError(f"Unknown type: {plot_type}")

def make_3x2_panel(data_dict,
                   left_keys=('exp2','exp4','exp6'),
                   right_keys=('exp3','exp5','exp7'),
                   row_tags=None,
                   figsize=(12, 15),
                   wspace=0.25, hspace=0.35,
                   outfile='panel_exp246_357'):
    import matplotlib.pyplot as plt

    assert len(left_keys) == len(right_keys) == 3
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    def _plot_one(ax, cfg_key):
        cfg = data_dict[cfg_key]
        series = normalize_series(cfg)
        # 你的 plot_line 已经会设置标题、线等；这里不显示图例
        plot_line(ax, cfg["x"], series,
                  title=auto_title_if_single_series(cfg, series),
                  ylim=cfg.get("ylim", None),
                  show_labels=cfg.get("show_labels", False),
                  show_legend=False,
                  yticks=cfg.get("yticks", None),
                  yticks_n=cfg.get("yticks_n", None))

    for i in range(3):
        lk, rk = left_keys[i], right_keys[i]
        _plot_one(axes[i, 0], lk)
        _plot_one(axes[i, 1], rk)

        # ✅ 分别给左右子图设置横坐标说明
        if lk in XLABEL_MAP:
            axes[i, 0].set_xlabel(XLABEL_MAP[lk])
        if rk in XLABEL_MAP:
            axes[i, 1].set_xlabel(XLABEL_MAP[rk])

        # （可选）如果还想加行标签 (a)(b)(c)，这里处理；你现在 row_tags=None 就跳过
        if row_tags is not None:
            axes[i, 0].text(-0.08, 1.08, row_tags[i],
                            transform=axes[i, 0].transAxes,
                            ha='right', va='bottom')

    save_fig(fig, outfile)




# ===== 在主入口里调用（示例） =====
# if __name__ == "__main__":
#     plot_all_exp(exp_data)  # 仍然各自单图
#     make_3x2_panel(exp_data,
#                    left_keys=('exp2','exp4','exp6'),
#                    right_keys=('exp3','exp5','exp7'),
#                    row_tags=('(a)','(b)','(c)'),
#                    figsize=(12, 7),   # 想更扁可加宽或减高
#                    outfile='panel_exp246_357')



# ========= 执行 =========
if __name__ == "__main__":
    plot_all_exp(exp_data)
    # make_3x2_panel(
    #     exp_data,
    #     left_keys=('exp2', 'exp4', 'exp6'),
    #     right_keys=('exp3', 'exp5', 'exp7'),
    #     row_tags=None,  # ← 不传或显式给 None 都行
    #     figsize=(12, 10),
    #     outfile='panel_exp246_357'
    # )


