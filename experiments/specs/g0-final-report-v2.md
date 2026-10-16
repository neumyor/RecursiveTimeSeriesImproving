# G0 深度重校准报告（v2，冻结版）

按计划 §12 与 main-v2.json `calibrationEvidence.g0-recalibration` 的要求，默认 incumbent 从线性 ridge（m0-linear）换为 PatchTST-lite（m0-patchtst）后，依赖 incumbent 的三个门（G0-3 增益紧度 / G0-4 缺陷定位 / G0-6 任务筛选）用深度基座加三个植入缺陷变体重新推导并重跑；G0-1/2/5/7 与模型无关，沿用 v1 冻结件 `experiments/specs/g0-final-report.json`。规模与 v1 相同（train 20000 / diag 10000 / cert 30000 / test 52000，L=96，H=24，seeds 0-2），运行目录 `experiments/runs/stage0-v2`。

模型集：`m0-patchtst`（干净基座）+ `m0-patchtst-noattn`（注意力消融，时间混合只剩展平头）+ `m0-patchtst-coarsepatch`（patch 48/48，窗口内周期分辨率坍塌）+ `m0-patchtst-channelmix`（通道在通道独立骨干前被跨通道平均，单通道任务上不适用）。全部变体保持基座契约：确定性、逐行独立、fp64 预测、fit <= 240s（实测每格 1-7s，见 `fit_seconds_by_model`）。

## G0-3+4: PASS
```json
{
  "cells": [
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst",
      "seed": 0,
      "gap": 0.1445,
      "eligible_for_tightness": true,
      "top_coordinate": "beyond_l",
      "top_aspect": "long-range-beyond-window",
      "expected": [
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst",
      "seed": 1,
      "gap": 0.1354,
      "eligible_for_tightness": true,
      "top_coordinate": "beyond_l",
      "top_aspect": "long-range-beyond-window",
      "expected": [
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst",
      "seed": 2,
      "gap": 0.1249,
      "eligible_for_tightness": true,
      "top_coordinate": "beyond_l",
      "top_aspect": "long-range-beyond-window",
      "expected": [
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-noattn",
      "seed": 0,
      "gap": 0.1413,
      "eligible_for_tightness": true,
      "top_coordinate": "beyond_l",
      "top_aspect": "long-range-beyond-window",
      "expected": [
        "lag-structure",
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-noattn",
      "seed": 1,
      "gap": 0.1417,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->3",
      "top_aspect": "cross-channel",
      "expected": [
        "lag-structure",
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": false
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-noattn",
      "seed": 2,
      "gap": 0.1235,
      "eligible_for_tightness": true,
      "top_coordinate": "beyond_l",
      "top_aspect": "long-range-beyond-window",
      "expected": [
        "lag-structure",
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-coarsepatch",
      "seed": 0,
      "gap": 0.1448,
      "eligible_for_tightness": true,
      "top_coordinate": "beyond_l",
      "top_aspect": "long-range-beyond-window",
      "expected": [
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-coarsepatch",
      "seed": 1,
      "gap": 0.1346,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[24,95]",
      "top_aspect": "lag-structure",
      "expected": [
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": false
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-coarsepatch",
      "seed": 2,
      "gap": 0.1269,
      "eligible_for_tightness": true,
      "top_coordinate": "beyond_l",
      "top_aspect": "long-range-beyond-window",
      "expected": [
        "periodicity",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-channelmix",
      "seed": 0,
      "gap": 0.8157,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->1",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-channelmix",
      "seed": 1,
      "gap": 0.6668,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->1",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_multiperiod",
      "model": "m0-patchtst-channelmix",
      "seed": 2,
      "gap": 0.6944,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->1",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst",
      "seed": 0,
      "gap": 0.0603,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->3",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst",
      "seed": 1,
      "gap": 0.0621,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst",
      "seed": 2,
      "gap": 0.0605,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-noattn",
      "seed": 0,
      "gap": 0.0587,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[0,23]",
      "top_aspect": "lag-structure",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-noattn",
      "seed": 1,
      "gap": 0.0612,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-noattn",
      "seed": 2,
      "gap": 0.0603,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-coarsepatch",
      "seed": 0,
      "gap": 0.0581,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->3",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-coarsepatch",
      "seed": 1,
      "gap": 0.0607,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-coarsepatch",
      "seed": 2,
      "gap": 0.0586,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-channelmix",
      "seed": 0,
      "gap": 0.0566,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->2",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-channelmix",
      "seed": 1,
      "gap": 0.0597,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "lgssm_xchannel",
      "model": "m0-patchtst-channelmix",
      "seed": 2,
      "gap": 0.0599,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "lag-structure"
      ],
      "localized": true
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst",
      "seed": 0,
      "gap": 0.1584,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[0,23]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": true
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst",
      "seed": 1,
      "gap": 0.1486,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[0,23]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": true
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst",
      "seed": 2,
      "gap": 0.1647,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[24,95]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": true
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst-noattn",
      "seed": 0,
      "gap": 0.2281,
      "eligible_for_tightness": true,
      "top_coordinate": "band(P=313)",
      "top_aspect": "periodicity",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": true
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst-noattn",
      "seed": 1,
      "gap": 0.1859,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": false
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst-noattn",
      "seed": 2,
      "gap": 0.1496,
      "eligible_for_tightness": true,
      "top_coordinate": "band(P=130)",
      "top_aspect": "periodicity",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": true
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst-coarsepatch",
      "seed": 0,
      "gap": 0.1765,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[0,23]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": true
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst-coarsepatch",
      "seed": 1,
      "gap": 0.1798,
      "eligible_for_tightness": true,
      "top_coordinate": "band(P=41)",
      "top_aspect": "periodicity",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": true
    },
    {
      "task": "hmm_ar_switch",
      "model": "m0-patchtst-coarsepatch",
      "seed": 2,
      "gap": 0.172,
      "eligible_for_tightness": true,
      "top_coordinate": "band(P=59)",
      "top_aspect": "periodicity",
      "expected": [
        "lag-structure",
        "periodicity"
      ],
      "localized": true
    },
    {
      "task": "arfima",
      "model": "m0-patchtst",
      "seed": 0,
      "gap": 0.0701,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[24,95]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "arfima",
      "model": "m0-patchtst",
      "seed": 1,
      "gap": 0.075,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[0,23]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "arfima",
      "model": "m0-patchtst",
      "seed": 2,
      "gap": 0.0701,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[24,95]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "arfima",
      "model": "m0-patchtst-noattn",
      "seed": 0,
      "gap": 0.0622,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[0,23]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "arfima",
      "model": "m0-patchtst-noattn",
      "seed": 1,
      "gap": 0.0756,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[0,23]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "arfima",
      "model": "m0-patchtst-noattn",
      "seed": 2,
      "gap": 0.0773,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[24,95]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "arfima",
      "model": "m0-patchtst-coarsepatch",
      "seed": 0,
      "gap": 0.0595,
      "eligible_for_tightness": true,
      "top_coordinate": "band(P=8)",
      "top_aspect": "periodicity",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": false
    },
    {
      "task": "arfima",
      "model": "m0-patchtst-coarsepatch",
      "seed": 1,
      "gap": 0.0661,
      "eligible_for_tightness": true,
      "top_coordinate": "lag[0,23]",
      "top_aspect": "lag-structure",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": true
    },
    {
      "task": "arfima",
      "model": "m0-patchtst-coarsepatch",
      "seed": 2,
      "gap": 0.0604,
      "eligible_for_tightness": true,
      "top_coordinate": "band(P=4)",
      "top_aspect": "periodicity",
      "expected": [
        "lag-structure",
        "long-range-beyond-window"
      ],
      "localized": false
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst",
      "seed": 0,
      "gap": 0.0545,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst",
      "seed": 1,
      "gap": 0.0569,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst",
      "seed": 2,
      "gap": 0.0557,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-noattn",
      "seed": 0,
      "gap": 0.0531,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-noattn",
      "seed": 1,
      "gap": 0.0563,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-noattn",
      "seed": 2,
      "gap": 0.0549,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-coarsepatch",
      "seed": 0,
      "gap": 0.0525,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-coarsepatch",
      "seed": 1,
      "gap": 0.0549,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-coarsepatch",
      "seed": 2,
      "gap": 0.0535,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-channelmix",
      "seed": 0,
      "gap": 0.0514,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-channelmix",
      "seed": 1,
      "gap": 0.0548,
      "eligible_for_tightness": true,
      "top_coordinate": "channel->0",
      "top_aspect": "cross-channel",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    },
    {
      "task": "nonlinear_xchannel",
      "model": "m0-patchtst-channelmix",
      "seed": 2,
      "gap": 0.0555,
      "eligible_for_tightness": true,
      "top_coordinate": "horizon",
      "top_aspect": "horizon-range",
      "expected": [
        "cross-channel",
        "horizon-range"
      ],
      "localized": true
    }
  ],
  "scatter": [
    {
      "lcb": 0.0707,
      "delta": 0.1041,
      "gap": 0.1445
    },
    {
      "lcb": 0.0751,
      "delta": 0.0968,
      "gap": 0.1354
    },
    {
      "lcb": 0.0696,
      "delta": 0.0876,
      "gap": 0.1249
    },
    {
      "lcb": 0.0687,
      "delta": 0.0969,
      "gap": 0.1413
    },
    {
      "lcb": 0.0773,
      "delta": 0.1087,
      "gap": 0.1417
    },
    {
      "lcb": 0.0658,
      "delta": 0.0889,
      "gap": 0.1235
    },
    {
      "lcb": 0.0725,
      "delta": 0.1009,
      "gap": 0.1448
    },
    {
      "lcb": 0.0739,
      "delta": 0.0957,
      "gap": 0.1346
    },
    {
      "lcb": 0.0662,
      "delta": 0.0887,
      "gap": 0.1269
    },
    {
      "lcb": 0.4875,
      "delta": 0.7765,
      "gap": 0.8157
    },
    {
      "lcb": 0.477,
      "delta": 0.6381,
      "gap": 0.6668
    },
    {
      "lcb": 0.3312,
      "delta": 0.6645,
      "gap": 0.6944
    },
    {
      "lcb": 0.0336,
      "delta": 0.034,
      "gap": 0.0603
    },
    {
      "lcb": 0.0293,
      "delta": 0.0314,
      "gap": 0.0621
    },
    {
      "lcb": 0.0338,
      "delta": 0.0347,
      "gap": 0.0605
    },
    {
      "lcb": 0.0333,
      "delta": 0.0336,
      "gap": 0.0587
    },
    {
      "lcb": 0.0291,
      "delta": 0.0315,
      "gap": 0.0612
    },
    {
      "lcb": 0.0338,
      "delta": 0.0351,
      "gap": 0.0603
    },
    {
      "lcb": 0.0331,
      "delta": 0.0334,
      "gap": 0.0581
    },
    {
      "lcb": 0.0295,
      "delta": 0.0317,
      "gap": 0.0607
    },
    {
      "lcb": 0.0339,
      "delta": 0.035,
      "gap": 0.0586
    },
    {
      "lcb": 0.0344,
      "delta": 0.0343,
      "gap": 0.0566
    },
    {
      "lcb": 0.0299,
      "delta": 0.0324,
      "gap": 0.0597
    },
    {
      "lcb": 0.0347,
      "delta": 0.0359,
      "gap": 0.0599
    },
    {
      "lcb": -0.0028,
      "delta": 0.036,
      "gap": 0.1584
    },
    {
      "lcb": -0.0011,
      "delta": 0.0078,
      "gap": 0.1486
    },
    {
      "lcb": 0.0021,
      "delta": 0.0245,
      "gap": 0.1647
    },
    {
      "lcb": 0.0126,
      "delta": 0.0954,
      "gap": 0.2281
    },
    {
      "lcb": 0.0089,
      "delta": 0.0575,
      "gap": 0.1859
    },
    {
      "lcb": -0.0114,
      "delta": 0.0145,
      "gap": 0.1496
    },
    {
      "lcb": -0.0091,
      "delta": 0.0223,
      "gap": 0.1765
    },
    {
      "lcb": -0.0009,
      "delta": 0.0159,
      "gap": 0.1798
    },
    {
      "lcb": -0.0027,
      "delta": 0.0075,
      "gap": 0.172
    },
    {
      "lcb": 0.0121,
      "delta": 0.0138,
      "gap": 0.0701
    },
    {
      "lcb": 0.0212,
      "delta": 0.0322,
      "gap": 0.075
    },
    {
      "lcb": 0.0144,
      "delta": 0.0189,
      "gap": 0.0701
    },
    {
      "lcb": 0.0057,
      "delta": 0.0076,
      "gap": 0.0622
    },
    {
      "lcb": 0.0161,
      "delta": 0.0312,
      "gap": 0.0756
    },
    {
      "lcb": 0.0188,
      "delta": 0.0303,
      "gap": 0.0773
    },
    {
      "lcb": 0.0103,
      "delta": 0.014,
      "gap": 0.0595
    },
    {
      "lcb": 0.0136,
      "delta": 0.0251,
      "gap": 0.0661
    },
    {
      "lcb": 0.0096,
      "delta": 0.0139,
      "gap": 0.0604
    },
    {
      "lcb": 0.0247,
      "delta": 0.0261,
      "gap": 0.0545
    },
    {
      "lcb": 0.0247,
      "delta": 0.026,
      "gap": 0.0569
    },
    {
      "lcb": 0.0236,
      "delta": 0.0266,
      "gap": 0.0557
    },
    {
      "lcb": 0.0245,
      "delta": 0.0259,
      "gap": 0.0531
    },
    {
      "lcb": 0.0249,
      "delta": 0.0262,
      "gap": 0.0563
    },
    {
      "lcb": 0.0238,
      "delta": 0.0269,
      "gap": 0.0549
    },
    {
      "lcb": 0.0245,
      "delta": 0.0257,
      "gap": 0.0525
    },
    {
      "lcb": 0.0249,
      "delta": 0.026,
      "gap": 0.0549
    },
    {
      "lcb": 0.0206,
      "delta": 0.0242,
      "gap": 0.0535
    },
    {
      "lcb": 0.0238,
      "delta": 0.0256,
      "gap": 0.0514
    },
    {
      "lcb": 0.0252,
      "delta": 0.0268,
      "gap": 0.0548
    },
    {
      "lcb": 0.0264,
      "delta": 0.0297,
      "gap": 0.0555
    }
  ],
  "tightness_ratio_median": 0.49,
  "tightness_cells": 54,
  "localization_hits": 49,
  "localization_cells": 54,
  "localization_rate": 0.907
}
```

## G0-6: PASS
```json
{
  "incumbent_model": "m0-patchtst",
  "remaining_gap_ratio_by_task": {
    "lgssm_multiperiod": 0.288,
    "lgssm_xchannel": 0.452,
    "hmm_ar_switch": 0.857,
    "arfima": 0.702,
    "nonlinear_xchannel": 0.529
  },
  "qualified_tasks": 4
}
```

## P2 v2 任务筛选（关键问题）

问题：PatchTST-lite + 一次性 g 修正后，P2 v2 合成任务上剩余 gap 比例是否 >= 30%？阈值 30% 以下 = 任务对（深度 incumbent + 线性修正器）组合过易，P2 v2 任务单受威胁。
本文件只给建议，不改 spec——由人决定。

```json
{
  "incumbent": "m0-patchtst",
  "threshold": 0.3,
  "by_task": {
    "hmm_ar_switch": {
      "remaining_gap_ratio": 0.857,
      "seeds": 3,
      "recommendation": "keep"
    },
    "nonlinear_xchannel": {
      "remaining_gap_ratio": 0.529,
      "seeds": 3,
      "recommendation": "keep"
    },
    "lgssm_xchannel": {
      "remaining_gap_ratio": 0.452,
      "seeds": 3,
      "recommendation": "keep"
    }
  },
  "note": "recommendation only: main-v2.json is not modified; a flagged task carries its measured fraction to the human decision"
}
```

## 各 (task, variant) 预期顶卡坐标的第一性推导

- **lgssm_multiperiod / m0-patchtst**：168 周期振荡器超出 L=96 窗口，其状态估计误差以 168 周期出现在残差里；band(P≈168) 能追踪、beyond_l（96-240 深历史）能钉住。预期 (periodicity, long-range-beyond-window)。
- **lgssm_multiperiod / noattn**：注意力是唯一的跨 patch 机制，消融后两个振荡器的相位积分都受损，且残差新增短滞后自相关——三个修复通道真正并列。预期 (lag-structure, periodicity, long-range-beyond-window)。
- **lgssm_multiperiod / coarsepatch**：48/48 patch 网格相位锁定 24 周期（96=2×48 是 24 的整数倍），但 168 周期分辨率坍塌，慢周期残差变大。预期 (periodicity, long-range-beyond-window)。
- **lgssm_multiperiod / channelmix**：每个通道收到的是平均预测，残差携带该通道相对平均的偏离；其它通道的实现历史修复共享部分，慢 level 泄漏使残差强自相关。预期 (cross-channel, lag-structure)。
- **lgssm_xchannel / 全部变体**：通道独立骨干（设计而非缺陷）对 lags 3-7 的跨通道耦合全盲；channel->i 直接修复，自身历史 lag 间接可见（v1 review C）。channelmix 额外把耦合驱动的逐通道偏离留在残差里。预期 (cross-channel, lag-structure)。
- **hmm_ar_switch / 基座、noattn、coarsepatch**：机制持续期内的 h 步误差自相关（lag-structure）；band 坐标能追踪机制间的慢残差游走——v1 中 m0-linear 的顶卡就是 band 峰，故 periodicity 登记为校准过的并列者。预期 (lag-structure, periodicity)。
- **arfima / 基座、noattn、coarsepatch**：d=0.35 长记忆，ACF 按 k^-0.3 衰减，残差历史（lag，至 335）与深层原始历史（beyond_l，96-240）都携带 L=96 窗口模型看不到的信号。预期 (lag-structure, long-range-beyond-window)。
- **nonlinear_xchannel / 全部变体**：通道独立骨干对 lags 3-7 的 tanh 耦合全盲，同视界已实现残差链也携带耦合（v1 登记理由，缺陷不改变）。预期 (cross-channel, horizon-range)。
- **channelmix 在 arfima / hmm_ar_switch 上排除**：单通道的跨通道平均是恒等操作，fit 直接拒绝（与 rule-crosschannel 在单通道任务上不适用的约定一致）。

登记表（与 stage0.DEEP_EXPECTED_ASPECTS 逐对一致）：
```json
{
  "lgssm_multiperiod/m0-patchtst": [
    "periodicity",
    "long-range-beyond-window"
  ],
  "lgssm_multiperiod/m0-patchtst-noattn": [
    "lag-structure",
    "periodicity",
    "long-range-beyond-window"
  ],
  "lgssm_multiperiod/m0-patchtst-coarsepatch": [
    "periodicity",
    "long-range-beyond-window"
  ],
  "lgssm_multiperiod/m0-patchtst-channelmix": [
    "cross-channel",
    "lag-structure"
  ],
  "lgssm_xchannel/m0-patchtst": [
    "cross-channel",
    "lag-structure"
  ],
  "lgssm_xchannel/m0-patchtst-noattn": [
    "cross-channel",
    "lag-structure"
  ],
  "lgssm_xchannel/m0-patchtst-coarsepatch": [
    "cross-channel",
    "lag-structure"
  ],
  "lgssm_xchannel/m0-patchtst-channelmix": [
    "cross-channel",
    "lag-structure"
  ],
  "hmm_ar_switch/m0-patchtst": [
    "lag-structure",
    "periodicity"
  ],
  "hmm_ar_switch/m0-patchtst-noattn": [
    "lag-structure",
    "periodicity"
  ],
  "hmm_ar_switch/m0-patchtst-coarsepatch": [
    "lag-structure",
    "periodicity"
  ],
  "arfima/m0-patchtst": [
    "lag-structure",
    "long-range-beyond-window"
  ],
  "arfima/m0-patchtst-noattn": [
    "lag-structure",
    "long-range-beyond-window"
  ],
  "arfima/m0-patchtst-coarsepatch": [
    "lag-structure",
    "long-range-beyond-window"
  ],
  "nonlinear_xchannel/m0-patchtst": [
    "cross-channel",
    "horizon-range"
  ],
  "nonlinear_xchannel/m0-patchtst-noattn": [
    "cross-channel",
    "horizon-range"
  ],
  "nonlinear_xchannel/m0-patchtst-coarsepatch": [
    "cross-channel",
    "horizon-range"
  ],
  "nonlinear_xchannel/m0-patchtst-channelmix": [
    "cross-channel",
    "horizon-range"
  ]
}
```

## 未命中（5/54，发现而非待调参数）

1. **lgssm_multiperiod / noattn / seed1**：顶卡 channel->3（cross-channel），预期 (lag-structure, periodicity, long-range-beyond-window)。诊断：注意力消融后通道 3（唯一无 level 潜变量的通道）窗口内建模最弱，其它通道的实现历史携带其缺失结构，跨通道修复在该 seed 上超过 beyond_l。卡片定位到了真实缺陷，只是落在未登记的修复通道上。
2. **lgssm_multiperiod / coarsepatch / seed1**：顶卡 lag[24,95]（lag-structure），预期 (periodicity, long-range-beyond-window)。诊断：粗 patch 使 level/振幅跟踪变粗，残差留下 24-95 滞后带的自相关——lag-structure 是推导时考虑过、但只为 noattn 登记的通道。另两个 seed 命中 beyond_l。
3. **hmm_ar_switch / noattn / seed1**：顶卡 horizon（horizon-range），预期 (lag-structure, periodicity)。诊断：机制证据汇聚受损时，同视界已实现残差链（h 步误差的持续性）是该 seed 上最强的可修复信号。
4. **arfima / coarsepatch / seed0、seed2**：顶卡 band(P=8) / band(P=4)（periodicity），预期 (lag-structure, long-range-beyond-window)。诊断：重演 v1 的 arfima 模式（v1 中 m0-linear 的 3 个 arfima cell 全部被 band 压中而 miss）。深度模型整体改善了 arfima 定位（基座/noattn 6/6 命中，v1 为 0/3），但粗 patch 留下短周期残差波纹，窄带 FIR 先吃到。

## 与 v1 的对照

- 定位率 0.907（49/54）vs v1 的 0.81（17/21）：深度缺陷变体的定位更干净。
- arfima 定位从 0/3（v1，band 顶卡）改善到 7/9。
- 紧度中位数 0.49（54 格全 eligible）vs v1 的 0.606（18/21 eligible）。
- G0-6 资格数不变（4/5）：lgssm_multiperiod 在两个 incumbent 下都低于 30%（v1 线性 0.291，v2 深度 0.288），且都不是 P2 v2 任务；深度 incumbent 没有改变任务资格格局。
- 深度 incumbent 并非处处更强：lgssm_multiperiod 上其 gap（0.125-0.145）略大于 m0-linear（0.112-0.126）；lgssm_xchannel / nonlinear_xchannel 上与线性几乎相同（通道独立性是约束瓶颈，gap 分别 0.060 vs 0.061、0.055 vs 0.056）。

## G0-1 / G0-2 / G0-5 / G0-7：沿用 v1（模型无关）

- **G0-1**（v1 判定 PASS）：oracle negative control: run with m = oracle, so no incumbent is involved; it measures the machinery's behaviour at zero true headroom。
- **G0-2**（v1 判定 PASS）：CI calibration: the HAC/HLN interval and the greedy total-gain estimator are properties of the corrector machinery, not of the model whose residuals feed it; the v1 coverage was measured with m0-linear cells, the coverage claim is estimator-level。
- **G0-5**（v1 判定 PASS）：false-stop: the Bonferroni level wiring, the two-sided-to-one-sided conservatism and the eps*MSE threshold never reference the incumbent; planted-defect gaps just need to sit far above the stop threshold, which the v1 linear defects did。
- **G0-7**（v1 判定 PASS）：observation-0: a pure AR(1)/white-noise analytic control with m == 0 — no incumbent exists in the gate at all。

**ALL: PASS**（G0-3+4 与 G0-6 的重校准判定 + v1 沿用门判定）
