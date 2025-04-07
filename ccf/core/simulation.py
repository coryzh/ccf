from ccf.core.cross_correlation_functions import NormalizedCCF
from scipy.stats import sigmaclip
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SimulationResults:
    simulation_type: str
    simulated_results: np.ndarray

    @property
    def point_estimate(self, est_type: str = "median") -> float:
        valid_options = ["median", "mean"]

        if est_type == "median":
            return np.median(self.simulated_results)

        elif est_type == "mean":
            return np.mean(self.simulated_results)

        else:
            raise ValueError(
                f"{type} is not a valid option; "
                f"please choose from {valid_options}"
            )

    @property
    def bounds(self, conf_level: float = 0.68) -> Tuple[float, float]:
        lo_percentile = (1 - conf_level) * 100 / 2
        hi_percentile = 100 - lo_percentile

        return (
            np.percentile(
                self.simulated_results, q=(lo_percentile, hi_percentile)
            )
        )

    @property
    def err(
        self, est_type: str = "median", conf_level: float = 0.68
    ) -> Tuple[float, float]:

        point_est = self.point_estimate(est_type=est_type)
        lo, hi = self.bounds(conf_level=conf_level)
        return (
            point_est - lo, hi - point_est
        )


class MonteCarloSimulation:
    def __init__(self, n_sim: int, ccf: NormalizedCCF,
                 n_sigma_clip: float = 3, n_simulations: int = 1000):
        self.n_sim = n_sim
        self.ccf = ccf
        self.n_sigma_clip = n_sigma_clip
        self.n_simulations = n_simulations

    @property
    def data_clipped(self) -> np.ndarray:
        results = sigmaclip(
            self.ccf.data_obs,
            low=self.n_sigma_clip, high=self.n_sigma_clip
        )
        return results[0]

    @property
    def data_uncertainty(self) -> float:
        return np.std(self.data_clipped)

    def get_simulated_data(self) -> np.ndarray:
        data_simulated = np.zeros(
            shape=(self.n_simulations, self.ccf.series_length)
        )

        data_simulated = (
            self.ccf.data_obs[np.newaxis, :]
            + self.data_uncertainty * np.random.randn(
                self.n_simulations, self.ccf.series_length
            )
        )

        return data_simulated

    def run(self) -> SimulationResults:
        rv_rand = np.zeros(self.n_simulations)

        for i, data_rand in enumerate(self.get_simulated_data()):
            ccf_rand = NormalizedCCF(
                data_obs=data_rand,
                data_temp=self.ccf.data_temp,
                bins=self.ccf.bins
            )

            rv_rand[i] = ccf_rand.rv

        results = SimulationResults(
            simulation_type="Monte Carlo",
            simulated_results=rv_rand
        )

        return results
