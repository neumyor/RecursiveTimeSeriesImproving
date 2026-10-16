# G0 校准报告 v3（固定水平停止规则，修订 II 后全七门重测，线性 incumbent）

## G0-7: PASS
```json
{
  "stats": {
    "ar1": {
      "delta": 0.6135,
      "expected": 0.5625,
      "ratio": 1.091
    },
    "white": {
      "delta": -0.0042,
      "expected": 0.0,
      "ratio": -0.0042
    }
  },
  "pass": true
}
```

## G0-1: PASS
```json
{
  "stats": {
    "cells": 18,
    "lcb_below_zero": 18,
    "saturated": 18,
    "total_lcbs": [
      -0.00038,
      -0.0004,
      -0.00066,
      -0.00012,
      -0.00023,
      -0.00084,
      -0.00407,
      -0.03311,
      -0.00138,
      -0.0018,
      -0.0012,
      -0.00714,
      -0.00066,
      -0.00079,
      -0.00037,
      -0.00141,
      -0.00014,
      -0.00076
    ],
    "saturation_rate_by_cert_size": {
      "5000": 0.889,
      "10000": 1.0,
      "20000": 1.0
    }
  },
  "pass": true
}
```

## G0-2: PASS
```json
{
  "stats": {
    "lgssm_multiperiod": {
      "reps": 12,
      "lcb_coverage": 1.0,
      "ucb_coverage": 1.0,
      "truth_over_gap": [
        0.806,
        0.812,
        0.769,
        0.641,
        0.683,
        0.644,
        0.766,
        0.671,
        0.648,
        0.695,
        0.582,
        0.757
      ]
    },
    "arfima": {
      "reps": 12,
      "lcb_coverage": 1.0,
      "ucb_coverage": 1.0,
      "truth_over_gap": [
        0.328,
        0.275,
        0.385,
        0.257,
        0.249,
        0.13,
        0.409,
        0.354,
        0.049,
        0.155,
        0.56,
        0.188
      ]
    },
    "hmm_ar_switch": {
      "reps": 12,
      "lcb_coverage": 1.0,
      "ucb_coverage": 1.0,
      "truth_over_gap": [
        0.05,
        0.216,
        0.088,
        0.271,
        0.167,
        0.143,
        0.143,
        0.031,
        0.294,
        0.293,
        -0.001,
        0.158
      ]
    }
  },
  "pass": true
}
```

## G0-3+4: PASS
```json
{
  "stats": {
    "cells": [
      {
        "task": "lgssm_multiperiod",
        "model": "m0-linear",
        "seed": 0,
        "gap": 0.1263,
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
        "model": "m0-linear",
        "seed": 1,
        "gap": 0.1205,
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
        "model": "m0-linear",
        "seed": 2,
        "gap": 0.1115,
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
        "model": "m0-short-lookback",
        "seed": 0,
        "gap": 0.241,
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
        "model": "m0-short-lookback",
        "seed": 1,
        "gap": 0.22,
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
        "model": "m0-short-lookback",
        "seed": 2,
        "gap": 0.2162,
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
        "task": "lgssm_xchannel",
        "model": "m0-linear",
        "seed": 0,
        "gap": 0.0608,
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
        "model": "m0-linear",
        "seed": 1,
        "gap": 0.0636,
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
        "model": "m0-linear",
        "seed": 2,
        "gap": 0.0615,
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
        "model": "m0-short-lookback",
        "seed": 0,
        "gap": 0.081,
        "eligible_for_tightness": true,
        "top_coordinate": "lag[0,23]",
        "top_aspect": "lag-structure",
        "expected": [
          "cross-channel",
          "lag-structure",
          "long-range-beyond-window"
        ],
        "localized": true
      },
      {
        "task": "lgssm_xchannel",
        "model": "m0-short-lookback",
        "seed": 1,
        "gap": 0.0832,
        "eligible_for_tightness": true,
        "top_coordinate": "lag[0,23]",
        "top_aspect": "lag-structure",
        "expected": [
          "cross-channel",
          "lag-structure",
          "long-range-beyond-window"
        ],
        "localized": true
      },
      {
        "task": "lgssm_xchannel",
        "model": "m0-short-lookback",
        "seed": 2,
        "gap": 0.0828,
        "eligible_for_tightness": true,
        "top_coordinate": "lag[0,23]",
        "top_aspect": "lag-structure",
        "expected": [
          "cross-channel",
          "lag-structure",
          "long-range-beyond-window"
        ],
        "localized": true
      },
      {
        "task": "arfima",
        "model": "m0-linear",
        "seed": 0,
        "gap": 0.0412,
        "eligible_for_tightness": false,
        "top_coordinate": "band(P=12)",
        "top_aspect": "periodicity",
        "expected": [
          "lag-structure"
        ],
        "localized": false
      },
      {
        "task": "arfima",
        "model": "m0-linear",
        "seed": 1,
        "gap": 0.0488,
        "eligible_for_tightness": false,
        "top_coordinate": "band(P=31)",
        "top_aspect": "periodicity",
        "expected": [
          "lag-structure"
        ],
        "localized": false
      },
      {
        "task": "arfima",
        "model": "m0-linear",
        "seed": 2,
        "gap": 0.0456,
        "eligible_for_tightness": false,
        "top_coordinate": "band(P=5)",
        "top_aspect": "periodicity",
        "expected": [
          "lag-structure"
        ],
        "localized": false
      },
      {
        "task": "hmm_ar_switch",
        "model": "m0-linear",
        "seed": 0,
        "gap": 0.1789,
        "eligible_for_tightness": true,
        "top_coordinate": "band(P=34)",
        "top_aspect": "periodicity",
        "expected": [
          "lag-structure",
          "periodicity"
        ],
        "localized": true
      },
      {
        "task": "hmm_ar_switch",
        "model": "m0-linear",
        "seed": 1,
        "gap": 0.1439,
        "eligible_for_tightness": true,
        "top_coordinate": "band(P=6)",
        "top_aspect": "periodicity",
        "expected": [
          "lag-structure",
          "periodicity"
        ],
        "localized": true
      },
      {
        "task": "hmm_ar_switch",
        "model": "m0-linear",
        "seed": 2,
        "gap": 0.1494,
        "eligible_for_tightness": true,
        "top_coordinate": "band(P=6)",
        "top_aspect": "periodicity",
        "expected": [
          "lag-structure",
          "periodicity"
        ],
        "localized": true
      },
      {
        "task": "nonlinear_xchannel",
        "model": "m0-linear",
        "seed": 0,
        "gap": 0.0555,
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
        "model": "m0-linear",
        "seed": 1,
        "gap": 0.0592,
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
        "model": "m0-linear",
        "seed": 2,
        "gap": 0.0558,
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
        "lcb": 0.0648,
        "delta": 0.0816,
        "gap": 0.1263
      },
      {
        "lcb": 0.0761,
        "delta": 0.0907,
        "gap": 0.1205
      },
      {
        "lcb": 0.067,
        "delta": 0.0813,
        "gap": 0.1115
      },
      {
        "lcb": 0.1609,
        "delta": 0.2099,
        "gap": 0.241
      },
      {
        "lcb": 0.169,
        "delta": 0.1955,
        "gap": 0.22
      },
      {
        "lcb": 0.1364,
        "delta": 0.1914,
        "gap": 0.2162
      },
      {
        "lcb": 0.0349,
        "delta": 0.0335,
        "gap": 0.0608
      },
      {
        "lcb": 0.0305,
        "delta": 0.0311,
        "gap": 0.0636
      },
      {
        "lcb": 0.0355,
        "delta": 0.0349,
        "gap": 0.0615
      },
      {
        "lcb": 0.0522,
        "delta": 0.053,
        "gap": 0.081
      },
      {
        "lcb": 0.0524,
        "delta": 0.0538,
        "gap": 0.0832
      },
      {
        "lcb": 0.053,
        "delta": 0.0534,
        "gap": 0.0828
      },
      {
        "lcb": 0.0059,
        "delta": 0.0024,
        "gap": 0.0412
      },
      {
        "lcb": 0.0179,
        "delta": 0.0203,
        "gap": 0.0488
      },
      {
        "lcb": 0.0111,
        "delta": 0.0114,
        "gap": 0.0456
      },
      {
        "lcb": 0.0239,
        "delta": 0.0466,
        "gap": 0.1789
      },
      {
        "lcb": -0.0123,
        "delta": -0.0025,
        "gap": 0.1439
      },
      {
        "lcb": -0.0032,
        "delta": 0.0051,
        "gap": 0.1494
      },
      {
        "lcb": 0.0258,
        "delta": 0.0258,
        "gap": 0.0555
      },
      {
        "lcb": 0.0269,
        "delta": 0.0269,
        "gap": 0.0592
      },
      {
        "lcb": 0.024,
        "delta": 0.025,
        "gap": 0.0558
      }
    ],
    "tightness_ratio_median": 0.606,
    "tightness_cells": 18,
    "localization_hits": 17,
    "localization_cells": 21,
    "localization_rate": 0.81
  },
  "pass": true
}
```

## G0-5: PASS
```json
{
  "stats": {
    "cells": 21,
    "saturated": 1,
    "truth_violations": 0,
    "details": [
      {
        "task": "lgssm_multiperiod",
        "model": "m0-linear",
        "seed": 0,
        "saturated": false,
        "gap": 0.1263
      },
      {
        "task": "lgssm_multiperiod",
        "model": "m0-linear",
        "seed": 1,
        "saturated": false,
        "gap": 0.1205
      },
      {
        "task": "lgssm_multiperiod",
        "model": "m0-linear",
        "seed": 2,
        "saturated": false,
        "gap": 0.1115
      },
      {
        "task": "lgssm_multiperiod",
        "model": "m0-short-lookback",
        "seed": 0,
        "saturated": false,
        "gap": 0.241
      },
      {
        "task": "lgssm_multiperiod",
        "model": "m0-short-lookback",
        "seed": 1,
        "saturated": false,
        "gap": 0.22
      },
      {
        "task": "lgssm_multiperiod",
        "model": "m0-short-lookback",
        "seed": 2,
        "saturated": false,
        "gap": 0.2162
      },
      {
        "task": "lgssm_xchannel",
        "model": "m0-linear",
        "seed": 0,
        "saturated": false,
        "gap": 0.0608
      },
      {
        "task": "lgssm_xchannel",
        "model": "m0-linear",
        "seed": 1,
        "saturated": false,
        "gap": 0.0636
      },
      {
        "task": "lgssm_xchannel",
        "model": "m0-linear",
        "seed": 2,
        "saturated": false,
        "gap": 0.0615
      },
      {
        "task": "lgssm_xchannel",
        "model": "m0-short-lookback",
        "seed": 0,
        "saturated": false,
        "gap": 0.081
      },
      {
        "task": "lgssm_xchannel",
        "model": "m0-short-lookback",
        "seed": 1,
        "saturated": false,
        "gap": 0.0832
      },
      {
        "task": "lgssm_xchannel",
        "model": "m0-short-lookback",
        "seed": 2,
        "saturated": false,
        "gap": 0.0828
      },
      {
        "task": "arfima",
        "model": "m0-linear",
        "seed": 0,
        "saturated": true,
        "gap": 0.0412
      },
      {
        "task": "arfima",
        "model": "m0-linear",
        "seed": 1,
        "saturated": false,
        "gap": 0.0488
      },
      {
        "task": "arfima",
        "model": "m0-linear",
        "seed": 2,
        "saturated": false,
        "gap": 0.0456
      },
      {
        "task": "hmm_ar_switch",
        "model": "m0-linear",
        "seed": 0,
        "saturated": false,
        "gap": 0.1789
      },
      {
        "task": "hmm_ar_switch",
        "model": "m0-linear",
        "seed": 1,
        "saturated": false,
        "gap": 0.1439
      },
      {
        "task": "hmm_ar_switch",
        "model": "m0-linear",
        "seed": 2,
        "saturated": false,
        "gap": 0.1494
      },
      {
        "task": "nonlinear_xchannel",
        "model": "m0-linear",
        "seed": 0,
        "saturated": false,
        "gap": 0.0555
      },
      {
        "task": "nonlinear_xchannel",
        "model": "m0-linear",
        "seed": 1,
        "saturated": false,
        "gap": 0.0592
      },
      {
        "task": "nonlinear_xchannel",
        "model": "m0-linear",
        "seed": 2,
        "saturated": false,
        "gap": 0.0558
      }
    ]
  },
  "pass": true
}
```

## G0-6: PASS
```json
{
  "stats": {
    "incumbent_model": "m0-linear",
    "remaining_gap_ratio_by_task": {
      "lgssm_multiperiod": 0.291,
      "lgssm_xchannel": 0.464,
      "arfima": 0.759,
      "hmm_ar_switch": 0.908,
      "nonlinear_xchannel": 0.544
    },
    "qualified_tasks": 4
  },
  "pass": true
}
```

all_pass: True
