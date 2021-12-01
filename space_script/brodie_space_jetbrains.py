from hyperopt import hp

fixed_values = {
    "filter_func": "threshold_trim",
    "coef_gap": 1.0,
    "mismatch_penalty": 0.0,
    "keep_ukn": True,
    "aggregate": "max",
    "rm_dup_stacks": False,
    "freq_by_stacks": True
}

space = {
    "gap_penalty": hp.choice("gap_penalty", (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)),

    "filter_func_k": hp.uniform("filter_func_k", 0.0, 130.0),
    "filter_recursion": hp.choice("filter_recursion", (None, 'modani', 'brodie')),
}
