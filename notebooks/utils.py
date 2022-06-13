# -*- coding: utf-8 -*-
import numpy as np
from skimage import io, segmentation, measure
from scipy import stats
from sklearn import metrics
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import re

def get_plans(plan_file: str) -> np.ndarray:
    """Retrieve plan from file"""
    plan = io.imread(plan_file).astype(np.uint16)

    plan_index = 0
    for z in range(plan.shape[0]):
        if len(np.unique(plan[z])) == 3:
            plan_index = z
    plan = segmentation.relabel_sequential(plan[plan_index])[0]

    return plan


def fuse_plans(plan_stack: np.ndarray) -> np.ndarray:
    """Fuse multiple plan files in a stack into a single image using a majority vote approach."""
    return np.asarray(stats.mode(plan_stack, axis=0))[0].squeeze()


def calculate_jaccard_score(plan_stack: np.ndarray, plan_fused: np.ndarray):
    """Calculate the jaccard score for a stack of plans and a reference fused plan"""

    JC = []
    for idx, plan in enumerate(plan_stack):
        JC.append(metrics.jaccard_score(plan_fused.flatten(), plan.flatten(), average=None))
    df = pd.DataFrame(JC, columns = [f'JC Label: {i}' for i in range(plan.max() + 1)])
    return df


def make_pretty_plan_overlay(stack_of_plans: np.ndarray, fused_plans: np.ndarray):
    """Make a nice figure to show the hippocampal estimations of different observers compared to their consensus"""

    sns.set_style("white")
    props = measure.regionprops_table(fused_plans, properties=['slice'])

    fig, ax = plt.subplots(figsize = (15, 10))
    ax.imshow(fused_plans[props['slice'][0]], cmap='gray_r')

    cmap = plt.cm.get_cmap('viridis')
    color=[cmap(0.9999999), cmap(0.5), cmap(0)]

    font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=24.3)

    for idx, plan in enumerate(stack_of_plans):
        sub = plan[props['slice'][0]]
        binary = sub == sub.max()
        contours = measure.find_contours(binary)
        ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=3, label=f'Hippocampus contour: Observer {idx + 1}',
                color=color[idx])

    ax.legend(fontsize=18, prop=font, framealpha=0.9, loc = 'upper left', bbox_to_anchor=(0.47,0.64))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.add_artist(AnchoredSizeBar(ax.transData, 20, '2 mm', 'lower left',
                                  pad=0.2, color='black', frameon=True,
                                  size_vertical=0.5, fontproperties=font))

    sub = calculate_jaccard_score(stack_of_plans, fused_plans)

    ax.text(0.95, 0.95, 'Jaccard coefficient = {:.2f} \u00B1 {:.2f}'.format(sub['JC Label: 2'].mean(), sub['JC Label: 2'].std()),
            transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=18,
            bbox={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9}, font=font)

    return ax

def get_info_from_report(report_file: str):

    px_size_rg = 0.05
    px_size_xr = 0.1

    info = {}
    with open(report_file, 'rt') as file:
        for line in file:
            if 'Isocenter coordinates' in line:
                vals = re.findall(r'\d+\.\d+', line)
                info['isocenter_coords_x'] = float(vals[0]) * px_size_rg
                info['isocenter_coords_y'] = float(vals[1]) * px_size_rg

            if 'Target coordinates (radiography)' in line:
                vals = re.findall(r'\d+', line)
                info['target_coords_rg_x'] = float(vals[0]) * px_size_rg
                info['target_coords_rg_y'] = float(vals[1]) * px_size_rg

            if 'Target coordinates (plan)' in line:
                vals = re.findall(r'\d+', line)
                info['target_coords_xr_x'] = float(vals[0]) * px_size_xr
                info['target_coords_xr_y'] = float(vals[1]) * px_size_xr

            if 'Transformation parameters' in line:
                _line = line.split(':')[-1]
                val1 = re.findall(r'\d+\.\d+', _line)
                val2 = re.findall(r'\d+', ' '.join(_line.split(',')[-2:]))
                vals = val1 + val2
                info['trafo_r'] = float(vals[0])
                info['trafo_angle'] = float(vals[1])
                info['trafo_x'] = float(vals[1]) * px_size_rg
                info['trafo_y'] = float(vals[2]) * px_size_rg

            if 'Motor origin' in line:
                vals = re.findall(r'\d+\.\d+', line)
                info['motor_origin_x'] = float(vals[0])
                info['motor_origin_y'] = float(vals[1])

            if 'Motor destination' in line:
                vals = re.findall(r'\d+\.\d+', line)
                info['motor_dest_x'] = float(vals[0])
                info['motor_dest_y'] = float(vals[1])

    return info
