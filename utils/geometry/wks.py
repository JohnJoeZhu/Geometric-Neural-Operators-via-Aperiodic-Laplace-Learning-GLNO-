
def wks(evals, evecs, energy_list, sigma, scaled=False):
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    indices = (evals > 1e-5)
    evals = evals[indices]
    evecs = evecs[:, indices]

    coefs = torch.exp(-torch.square(energy_list[:, None] - torch.log(torch.abs(evals))[None, :]) / (2 * sigma ** 2))

    weighted_evecs = evecs[None, :, :] * coefs[:, None, :]
    wks = torch.einsum('tnk,nk->nt', weighted_evecs, evecs)

    if scaled:
        inv_scaling = coefs.sum(1)
        return (1 / inv_scaling)[None, :] * wks
    else:
        return wks

def auto_wks(evals, evecs, n_descr, scaled=True):
    abs_ev = torch.sort(evals.abs())[0]
    e_min, e_max = torch.log(abs_ev[1]), torch.log(abs_ev[-1])
    sigma = 7 * (e_max - e_min) / n_descr

    e_min += 2 * sigma
    e_max -= 2 * sigma

    energy_list = torch.linspace(float(e_min), float(e_max), n_descr, device=evals.device, dtype=evals.dtype)

    return wks(abs_ev, evecs, energy_list, sigma, scaled=scaled)

def compute_wks_autoscale(evals, evecs, mass, n_descr=128, subsample_step=1, n_eig=128):
    feats = []
    for b in range(evals.shape[0]):
        feat = auto_wks(evals[b, :n_eig], evecs[b, :, :n_eig], n_descr, scaled=True)
        feat = feat[:, torch.arange(0, feat.shape[1], subsample_step)]
        feat_norm = torch.einsum('np,np->p', feat, mass[b].unsqueeze(1) * feat)
        feat /= torch.sqrt(feat_norm)
        feats += [feat]
    feats = torch.stack(feats, dim=0)
    return feats