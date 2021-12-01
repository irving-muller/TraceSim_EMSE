from hyperopt import hp

fixed_values= {
     "match_cost": 1.0,
     "gap_penalty": 1.0,
     "mismatch_penalty": 1.0,
     "const_match": False,
     "const_gap": False,
     "const_mismatch": False,
     "brodie_function": False,
     "no_norm": True,
     "diff_coef": 1.0,
     "sigmoid": False,
     "gamma": 1.0,
     "idf": False,
     "sum": True,
     "reciprocal_func": True,
     "filter_func": "threshold_trim"
}

space = {
    "df_coef": hp.uniform("df_coef", 0.0, 100.0),
    "pos_coef": hp.uniform("pos_coef", 0.0, 30.0),
    "diff_coef": hp.uniform("diff_coef", 0.0, 30.0),

    "keep_ukn": hp.choice("keep_ukn", (False, True)),
    "static_df_ukn": hp.choice("static_df_ukn", (False, True)),
    "aggregate": hp.choice("aggregate", ('max', 'avg_query', 'avg_cand', 'avg_short', 'avg_long', 'avg_query_cand')),
    "rm_dup_stacks": hp.choice("rm_dup_stacks", (False, True)),
    "freq_by_stacks": hp.choice("freq_by_stacks", (False, True)),
    "filter_func_k": hp.uniform("filter_func_k", 0.0, 130.0),
    "filter_recursion": hp.choice("filter_recursion", (None, 'modani', 'brodie')),
}

