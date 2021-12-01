from hyperopt import hp

fixed_values= {
    "filter_func": "threshold_trim",
    "keep_ukn": True,
    "aggregate": "max",
    "rm_dup_stacks": False,
    "freq_by_stacks": True
}


space = {
    "filter_func_k": hp.uniform("filter_func_k", 0.0, 130.0),
    "filter_recursion": hp.choice("filter_recursion", (None, 'modani', 'brodie')),
}

