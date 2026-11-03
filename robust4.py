"""robust3.py with the gate-only control and the joint-readout variant added to the models.

python robust4.py <cohort> <target> <seed>
"""
import numpy as np
import robust2, robust3
from train_learned import Model
from train_review import Gate
from train_review2 import GateTopo

robust2.FILTS = ('learned_nc', 'gate_nc', 'gatetopo_nc', 'fixed_tuned_nc', 'sublevel_nc', 'zero_nc')


def make_model(d_in, P, v):
    if v.startswith('gatetopo'):
        return GateTopo(d_in, P)
    if v.startswith('gate'):
        return Gate(d_in, P)
    return Model(d_in, P, v)


robust2.Model = make_model

if __name__ == '__main__':
    robust2.main()
