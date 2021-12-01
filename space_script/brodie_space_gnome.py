from hyperopt import hp

fixed_values = {
    "filter_func": "threshold_trim",
    "coef_gap": 1.0,
    "mismatch_penalty": 0.0
}

space = {
    "gap_penalty": hp.choice("gap_penalty", (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)),

    "keep_ukn": hp.choice("keep_ukn", (False, True)),
    "static_df_ukn": hp.choice("static_df_ukn", (False, True)),
    "aggregate": hp.choice("aggregate", ('max', 'avg_query', 'avg_cand', 'avg_short', 'avg_long', 'avg_query_cand')),
    "rm_dup_stacks": hp.choice("rm_dup_stacks", (False, True)),
    "freq_by_stacks": hp.choice("freq_by_stacks", (False, True)),
    "filter_func_k": hp.uniform("filter_func_k", 0.0, 130.0),
    "filter_recursion": hp.choice("filter_recursion", (None, 'modani', 'brodie')),
}
