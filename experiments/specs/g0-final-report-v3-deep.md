# G0 v3 深度重校准（固定水平停止规则，G0-3/4/6，PatchTST incumbent 及缺陷变体）

## G0-3+4: PASS
```json
{
  "stats": {
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
        "lcb": 0.079,
        "delta": 0.1041,
        "gap": 0.1445
      },
      {
        "lcb": 0.0836,
        "delta": 0.0968,
        "gap": 0.1354
      },
      {
        "lcb": 0.0746,
        "delta": 0.0876,
        "gap": 0.1249
      },
      {
        "lcb": 0.0764,
        "delta": 0.0969,
        "gap": 0.1413
      },
      {
        "lcb": 0.0892,
        "delta": 0.1087,
        "gap": 0.1417
      },
      {
        "lcb": 0.0711,
        "delta": 0.0889,
        "gap": 0.1235
      },
      {
        "lcb": 0.0798,
        "delta": 0.1009,
        "gap": 0.1448
      },
      {
        "lcb": 0.0823,
        "delta": 0.0957,
        "gap": 0.1346
      },
      {
        "lcb": 0.0713,
        "delta": 0.0887,
        "gap": 0.1269
      },
      {
        "lcb": 0.5816,
        "delta": 0.7765,
        "gap": 0.8157
      },
      {
        "lcb": 0.5585,
        "delta": 0.6381,
        "gap": 0.6668
      },
      {
        "lcb": 0.386,
        "delta": 0.6645,
        "gap": 0.6944
      },
      {
        "lcb": 0.0352,
        "delta": 0.034,
        "gap": 0.0603
      },
      {
        "lcb": 0.0308,
        "delta": 0.0314,
        "gap": 0.0621
      },
      {
        "lcb": 0.0353,
        "delta": 0.0347,
        "gap": 0.0605
      },
      {
        "lcb": 0.035,
        "delta": 0.0336,
        "gap": 0.0587
      },
      {
        "lcb": 0.0306,
        "delta": 0.0315,
        "gap": 0.0612
      },
      {
        "lcb": 0.0353,
        "delta": 0.0351,
        "gap": 0.0603
      },
      {
        "lcb": 0.0348,
        "delta": 0.0334,
        "gap": 0.0581
      },
      {
        "lcb": 0.031,
        "delta": 0.0317,
        "gap": 0.0607
      },
      {
        "lcb": 0.0354,
        "delta": 0.035,
        "gap": 0.0586
      },
      {
        "lcb": 0.0362,
        "delta": 0.0343,
        "gap": 0.0566
      },
      {
        "lcb": 0.0316,
        "delta": 0.0324,
        "gap": 0.0597
      },
      {
        "lcb": 0.0363,
        "delta": 0.0359,
        "gap": 0.0599
      },
      {
        "lcb": 0.0093,
        "delta": 0.036,
        "gap": 0.1584
      },
      {
        "lcb": 0.0029,
        "delta": 0.0078,
        "gap": 0.1486
      },
      {
        "lcb": 0.0133,
        "delta": 0.0245,
        "gap": 0.1647
      },
      {
        "lcb": 0.043,
        "delta": 0.0954,
        "gap": 0.2281
      },
      {
        "lcb": 0.0407,
        "delta": 0.0575,
        "gap": 0.1859
      },
      {
        "lcb": 0.0037,
        "delta": 0.0145,
        "gap": 0.1496
      },
      {
        "lcb": 0.006,
        "delta": 0.0223,
        "gap": 0.1765
      },
      {
        "lcb": 0.0062,
        "delta": 0.0159,
        "gap": 0.1798
      },
      {
        "lcb": 0.0033,
        "delta": 0.0075,
        "gap": 0.172
      },
      {
        "lcb": 0.0217,
        "delta": 0.0138,
        "gap": 0.0701
      },
      {
        "lcb": 0.0288,
        "delta": 0.0322,
        "gap": 0.075
      },
      {
        "lcb": 0.0202,
        "delta": 0.0189,
        "gap": 0.0701
      },
      {
        "lcb": 0.0116,
        "delta": 0.0076,
        "gap": 0.0622
      },
      {
        "lcb": 0.0272,
        "delta": 0.0312,
        "gap": 0.0756
      },
      {
        "lcb": 0.0263,
        "delta": 0.0303,
        "gap": 0.0773
      },
      {
        "lcb": 0.0146,
        "delta": 0.014,
        "gap": 0.0595
      },
      {
        "lcb": 0.0232,
        "delta": 0.0251,
        "gap": 0.0661
      },
      {
        "lcb": 0.0147,
        "delta": 0.0139,
        "gap": 0.0604
      },
      {
        "lcb": 0.0261,
        "delta": 0.0261,
        "gap": 0.0545
      },
      {
        "lcb": 0.0261,
        "delta": 0.026,
        "gap": 0.0569
      },
      {
        "lcb": 0.0253,
        "delta": 0.0266,
        "gap": 0.0557
      },
      {
        "lcb": 0.0258,
        "delta": 0.0259,
        "gap": 0.0531
      },
      {
        "lcb": 0.0263,
        "delta": 0.0262,
        "gap": 0.0563
      },
      {
        "lcb": 0.0255,
        "delta": 0.0269,
        "gap": 0.0549
      },
      {
        "lcb": 0.0258,
        "delta": 0.0257,
        "gap": 0.0525
      },
      {
        "lcb": 0.0264,
        "delta": 0.026,
        "gap": 0.0549
      },
      {
        "lcb": 0.0226,
        "delta": 0.0242,
        "gap": 0.0535
      },
      {
        "lcb": 0.0251,
        "delta": 0.0256,
        "gap": 0.0514
      },
      {
        "lcb": 0.0267,
        "delta": 0.0268,
        "gap": 0.0548
      },
      {
        "lcb": 0.0283,
        "delta": 0.0297,
        "gap": 0.0555
      }
    ],
    "tightness_ratio_median": 0.49,
    "tightness_cells": 54,
    "localization_hits": 49,
    "localization_cells": 54,
    "localization_rate": 0.907
  },
  "pass": true
}
```

## G0-6: PASS
```json
{
  "stats": {
    "incumbent_model": "m0-patchtst",
    "remaining_gap_ratio_by_task": {
      "lgssm_multiperiod": 0.288,
      "lgssm_xchannel": 0.452,
      "hmm_ar_switch": 0.857,
      "arfima": 0.702,
      "nonlinear_xchannel": 0.529
    },
    "qualified_tasks": 4
  },
  "pass": true
}
```

all_pass: True
