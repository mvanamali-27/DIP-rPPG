"""Step 3.2 (shared): one scorecard for every method -- classical or learned.

Every learned model is judged with Leave-One-Subject-Out (LOSO): train on everyone
except one person, test on that person, and repeat for every person. So a model is
never tested on someone it has already seen -- just like a real new user.
"""
import numpy as np
import matplotlib.pyplot as plt


def metrics(true_bpm, pred_bpm):
    """The three headline numbers for a set of predictions."""
    err = np.asarray(pred_bpm) - np.asarray(true_bpm)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "within5": float(np.mean(np.abs(err) <= 5))}


def per_subject_mae(true_bpm, pred_bpm, subjects):
    """MAE for each person separately -- shows who a method struggles on."""
    true_bpm, pred_bpm = np.asarray(true_bpm), np.asarray(pred_bpm)
    return {s: float(np.mean(np.abs(pred_bpm[subjects == s] - true_bpm[subjects == s])))
            for s in np.unique(subjects)}


def loso_splits(subjects):
    """For each person: (that person, rows to train on, rows to test on)."""
    for s in np.unique(subjects):
        test = subjects == s
        yield s, np.where(~test)[0], np.where(test)[0]


def loso_predict(make_model, X, y, subjects, verbose=True):
    """Train and test a model once per person, and return one prediction per window
    -- each made by a model that never saw that window's person.

    make_model: a function that returns a FRESH, untrained model with .fit(X, y)
    and .predict(X). Any scikit-learn model works, and the CNN will get the same
    two methods. A fresh model per person matters: reusing one would let it
    remember the test person from an earlier round.
    """
    pred = np.zeros(len(y))
    people = np.unique(subjects)
    for k, (s, train, test) in enumerate(loso_splits(subjects), 1):
        model = make_model()
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
        if verbose:
            print(f"  [{k}/{len(people)}] tested on {s}: MAE {np.mean(np.abs(pred[test] - y[test])):.2f}")
    return pred


def summary(name, true_bpm, pred_bpm, subjects):
    """Print one line of the scorecard and return its numbers."""
    m = metrics(true_bpm, pred_bpm)
    per_person = list(per_subject_mae(true_bpm, pred_bpm, subjects).values())
    m.update(person_avg=float(np.mean(per_person)), person_median=float(np.median(per_person)),
             person_worst=float(np.max(per_person)))
    print(f"{name:18s} MAE {m['MAE']:5.2f} | RMSE {m['RMSE']:5.2f} | within 5 bpm {m['within5']:4.0%} | "
          f"per person: avg {m['person_avg']:5.2f}, median {m['person_median']:5.2f}, worst {m['person_worst']:5.2f}")
    return m


def plot_agreement(name, true_bpm, pred_bpm):
    """Two standard pictures of how well predictions agree with the truth."""
    true_bpm, pred_bpm = np.asarray(true_bpm), np.asarray(pred_bpm)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: every window as a dot -- perfect predictions sit on the dashed line
    lo = min(true_bpm.min(), pred_bpm.min()) - 5
    hi = max(true_bpm.max(), pred_bpm.max()) + 5
    ax[0].scatter(true_bpm, pred_bpm, s=8, alpha=0.4)
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    ax[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="true bpm", ylabel="predicted bpm",
              title=f"{name}: predicted vs true")
    ax[0].legend()

    # right: Bland-Altman -- the error against the size of the heart rate
    mean, diff = (true_bpm + pred_bpm) / 2, pred_bpm - true_bpm
    bias, spread = diff.mean(), 1.96 * diff.std()
    ax[1].scatter(mean, diff, s=8, alpha=0.4)
    ax[1].axhline(bias, color="k", lw=1)
    ax[1].axhline(bias + spread, color="k", ls="--", lw=1)
    ax[1].axhline(bias - spread, color="k", ls="--", lw=1)
    ax[1].set(xlabel="average of true and predicted (bpm)", ylabel="predicted - true (bpm)",
              title=f"Bland-Altman: bias {bias:+.2f}, 95% of errors in [{bias - spread:+.1f}, {bias + spread:+.1f}]")
    plt.tight_layout()
    plt.show()
