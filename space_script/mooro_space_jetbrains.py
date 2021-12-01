from hyperopt import hp

fixed_values= {
     "filter_func": "threshold_trim",
    "keep_ukn": True,
    "aggregate": "max",
    "rm_dup_stacks": False,
    "freq_by_stacks": True
}


space = {
    "c": hp.uniform("c", 0.0, 30.0),
    "o": hp.uniform("o", 0.0, 30.0),
    "alpha": hp.uniform("alpha", 0.0, 1.0),
    "n_top": hp.choice("n_top", [i for i in range(1, 400)]),


    "filter_func_k": hp.uniform("filter_func_k", 0.0, 130.0),
    "filter_recursion": hp.choice("filter_recursion", (None, 'modani', 'brodie')),
}

