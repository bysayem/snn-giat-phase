"""First-order energy model. Horowitz, ISSCC 2014 (45 nm, 32-bit float):
   multiply 3.7 pJ, add 0.9 pJ -> MAC = 4.6 pJ, accumulate (AC) = 0.9 pJ.
   Memory access, activation functions and biases are NOT included."""

E_MAC_PJ = 4.6
E_AC_PJ = 0.9


def lstm_cost(n_in, n_h, n_out):
    macs = 4 * (n_in * n_h + n_h * n_h) + n_h * n_out + 3 * n_h   # gates + elementwise
    return {"macs": macs, "energy_pj": macs * E_MAC_PJ}


def snn_cost(n_h, n_out, in_spk, rate1, rate2):
    """in_spk: avg input spikes per step. rate1/rate2: avg spikes/neuron/step."""
    s1, s2 = rate1 * n_h, rate2 * n_h
    acs = in_spk * n_h + s1 * n_h + s1 * n_h + s2 * n_out         # fc1, rec1, fc2, fc3
    upd_macs = 2 * n_h + n_out                                     # membrane decay updates
    syn, upd = acs * E_AC_PJ, upd_macs * E_MAC_PJ
    return {"acs": acs, "update_macs": upd_macs,
            "synaptic_pj": syn, "update_pj": upd, "energy_pj": syn + upd}
