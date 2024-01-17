"""
Define here your Ray resources - for hyperparams tuning

e.g. cpu x trial:

{
    'cpu': n,
    'cpu': n,
    ...
}

or cpu and gpu x trial

{
    'gpu': ...,
    'cpu': ...,
    ...
}
"""

TUNING_RES = {
    'cpu': 12/4 # LS: more convenient to define a fraction of CPUs (automatic assignment to each samples)
}