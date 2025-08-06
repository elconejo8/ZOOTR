import numpy as np

def choose_preds(preds, pred_thr=0.75, frames_thr=30):
    """
    Chooses which preds are item grabs. Uses breaks to enforce a minimum frame ditance
    between item grabs so as to avoid double counting the same.
    """
    inds = np.where(preds[:,1] > pred_thr)[0]
    breaks = np.where(np.diff(inds) > frames_thr)[0]

    items_inds = inds[breaks]

    return items_inds