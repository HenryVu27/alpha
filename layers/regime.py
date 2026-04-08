"""Regime Detection: HMM + Bayesian Online Changepoint Detection.

Implements:
- BayesianChangepoint (Adams & MacKay 2007)
- RegimeHMM (wrapper around hmmlearn GaussianHMM)
- select_n_states_bic (BIC-based model selection)
- RegimeDetector (full pipeline combining HMM + BOCPD + PCA)
"""

import os
import pickle
from typing import Optional

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.stats import t as student_t
from sklearn.decomposition import PCA


class BayesianChangepoint:
    """Bayesian Online Changepoint Detection (Adams & MacKay 2007)."""

    def __init__(self, hazard_lambda: int = 250):
        self.hazard_lambda = hazard_lambda
        self.hazard_rate = 1.0 / hazard_lambda

    def _student_t_pdf(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Compute Student-t predictive pdf from Normal-Inverse-Gamma posterior."""
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        return student_t.pdf(x, df=df, loc=mu, scale=scale)

    def run(self, signal: np.ndarray) -> np.ndarray:
        """Run BOCPD on a 1-D signal, returning changepoint probabilities.

        Uses run-length based algorithm with Normal-Inverse-Gamma prior.
        Changepoint probability at time t is derived from the posterior
        run-length distribution: probability mass on short run-lengths
        (r < threshold) indicates a recent changepoint.
        """
        T = len(signal)
        # Run length probability matrix: R[t, r] = P(run_length=r at time t)
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0

        # Sufficient statistics for Normal-Inverse-Gamma prior
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0

        mu = np.array([mu0])
        kappa = np.array([kappa0])
        alpha = np.array([alpha0])
        beta = np.array([beta0])

        changepoint_prob = np.zeros(T)
        # Threshold for "short" run lengths that indicate recent changepoint
        rl_threshold = max(5, self.hazard_lambda // 50)

        for t in range(T):
            x = signal[t]

            # Predictive probabilities under each run length hypothesis
            pred = self._student_t_pdf(x, mu, kappa, alpha, beta)

            # Growth probabilities: R[t+1, r+1] = R[t, r] * pred[r] * (1 - h)
            growth = R[t, : t + 1] * pred * (1 - self.hazard_rate)

            # Changepoint probability: R[t+1, 0] = sum(R[t, r] * pred[r] * h)
            cp = np.sum(R[t, : t + 1] * pred * self.hazard_rate)

            # Update run length distribution
            R[t + 1, 1 : t + 2] = growth
            R[t + 1, 0] = cp

            # Normalize
            evidence = R[t + 1, : t + 2].sum()
            if evidence > 0:
                R[t + 1, : t + 2] /= evidence

            # Changepoint probability: mass on short run-lengths indicates
            # a recent changepoint occurred. When no changepoint has happened,
            # almost all mass is on the longest run length. We only consider
            # this meaningful after enough data has accumulated.
            if t >= rl_threshold:
                short_rl_end = min(rl_threshold, t + 2)
                changepoint_prob[t] = np.sum(R[t + 1, 0:short_rl_end])
            else:
                # Not enough data yet to determine changepoint
                changepoint_prob[t] = R[t + 1, 0]

            # Update sufficient statistics (Normal-Inverse-Gamma)
            mu_new = np.concatenate([[mu0], (kappa * mu + x) / (kappa + 1)])
            kappa_new = np.concatenate([[kappa0], kappa + 1])
            alpha_new = np.concatenate([[alpha0], alpha + 0.5])
            beta_new = np.concatenate([
                [beta0],
                beta + kappa * (x - mu) ** 2 / (2 * (kappa + 1)),
            ])

            mu = mu_new
            kappa = kappa_new
            alpha = alpha_new
            beta = beta_new

        return changepoint_prob


class RegimeHMM:
    """Wrapper around hmmlearn GaussianHMM for regime detection."""

    def __init__(self, n_states: int = 3, covariance_type: str = "diag"):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )

    def fit(self, data: np.ndarray) -> "RegimeHMM":
        """Fit the HMM to data. data shape: (n_samples, n_features)."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.model.fit(data)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.model.predict(data)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Return state probability matrix (n_samples, n_states)."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.model.predict_proba(data)

    def transition_matrix(self) -> np.ndarray:
        """Return the learned transition matrix."""
        return self.model.transmat_

    def score(self, data: np.ndarray) -> float:
        """Return log-likelihood of data under the model."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.model.score(data)

    def bic(self, data: np.ndarray) -> float:
        """Compute BIC for the fitted model on data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_samples, n_features = data.shape
        log_likelihood = self.model.score(data)

        # Number of free parameters
        n_means = self.n_states * n_features
        n_covars = self.n_states * n_features  # diag covariance
        n_trans = self.n_states * (self.n_states - 1)
        n_start = self.n_states - 1
        n_params = n_means + n_covars + n_trans + n_start

        return -2 * log_likelihood + n_params * np.log(n_samples)

    def save(self, path: str) -> None:
        """Pickle the model to path."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load a pickled model from path."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)


def select_n_states_bic(data: np.ndarray, max_states: int = 5) -> int:
    """Select optimal number of HMM states using BIC."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    best_bic = np.inf
    best_n = 2

    for n in range(2, max_states + 1):
        try:
            hmm = RegimeHMM(n_states=n, covariance_type="diag")
            hmm.fit(data)
            bic_val = hmm.bic(data)
            if bic_val < best_bic:
                best_bic = bic_val
                best_n = n
        except Exception:
            continue

    return best_n


class RegimeDetector:
    """Full regime detection pipeline combining HMM + BOCPD + PCA."""

    def __init__(
        self,
        n_states: int = 3,
        hazard_lambda: int = 250,
        state_labels: Optional[dict] = None,
    ):
        self.n_states = n_states
        self.hazard_lambda = hazard_lambda
        self.state_labels = state_labels or {
            0: "risk_on",
            1: "stalemate",
            2: "escalation",
        }
        self.hmm = RegimeHMM(n_states=n_states, covariance_type="diag")
        self.bocpd = BayesianChangepoint(hazard_lambda=hazard_lambda)
        self.pca = PCA(n_components=1)

    def fit(self, features: np.ndarray) -> "RegimeDetector":
        """Fit HMM and PCA on feature data."""
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        self.hmm.fit(features)
        self.pca.fit(features)
        return self

    def detect(self, features: np.ndarray) -> dict:
        """Run full detection pipeline.

        Returns dict with: regime_label, regime_label_idx, regime_probabilities,
        changepoint_probability, transition_matrix, hmm_log_likelihood,
        data_staleness_seconds.
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # HMM predictions
        labels = self.hmm.predict(features)
        proba = self.hmm.predict_proba(features)
        current_regime_idx = int(labels[-1])
        current_regime_label = self.state_labels.get(
            current_regime_idx, f"state_{current_regime_idx}"
        )

        # BOCPD on first principal component (stress index)
        stress_index = self.pca.transform(features).ravel()
        cp_probs = self.bocpd.run(stress_index)

        # Log-likelihood
        log_likelihood = self.hmm.score(features)

        return {
            "regime_label": current_regime_label,
            "regime_label_idx": current_regime_idx,
            "regime_probabilities": proba[-1],
            "changepoint_probability": float(cp_probs[-1]),
            "transition_matrix": self.hmm.transition_matrix(),
            "hmm_log_likelihood": float(log_likelihood),
            "data_staleness_seconds": 0.0,
        }

    def save(self, directory: str) -> None:
        """Save HMM and PCA models to directory."""
        os.makedirs(directory, exist_ok=True)
        self.hmm.save(os.path.join(directory, "hmm.pkl"))
        with open(os.path.join(directory, "pca.pkl"), "wb") as f:
            pickle.dump(self.pca, f)

    def load(self, directory: str) -> None:
        """Load HMM and PCA models from directory."""
        self.hmm.load(os.path.join(directory, "hmm.pkl"))
        with open(os.path.join(directory, "pca.pkl"), "rb") as f:
            self.pca = pickle.load(f)
