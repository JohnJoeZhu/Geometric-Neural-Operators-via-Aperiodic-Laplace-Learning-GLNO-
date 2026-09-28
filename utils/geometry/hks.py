
def compute_hks(evals, evecs, scales):
    """
    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) values
      - scales: (S) times
    Outputs:
      - (V,S) hks values
    """

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    # TODO could be a matmul
    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # (B,1,S,K)
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)

    out = torch.sum(terms, dim=-1) # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    else:
        return out

def compute_hks_autoscale(evals, evecs, count):
    # these scales roughly approximate those suggested in the hks paper
    scales = torch.logspace(-2, 0., steps=count, device=evals.device, dtype=evals.dtype)
    return compute_hks(evals, evecs, scales)