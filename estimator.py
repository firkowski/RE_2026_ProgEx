from const import EstimatorConstant
import numpy as np


class EKF:
    """
    Extended Kalman Filter for the whale tracking problem under Gaussian noise.

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator (initial state bounds, process 
            noise paramaters, etc...)
    """

    def __init__(
            self,
            estimator_constant: EstimatorConstant,
    ):
        self.constant = estimator_constant

        # Posterior mean of the state estimate
        self.xm = None
         # Posterior covariance of the state estimate
        self.Pm = None

    def initialize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator. Set up any internal state required by the
        filter and return the initial posterior mean and covariance.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The initial posterior state mean. The order of states is
                x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
            Pm : np.ndarray, dim: (num_states, num_states)
                The initial posterior state covariance.
        """
        xm = None
        Pm = None

        return xm, Pm

    def estimate(
            self,
            measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update the estimator with a new measurement and return the posterior
        mean and covariance at the current time step.

        Args:
            measurement : np.ndarray, dim: (num_measurements,)
                Sensor measurements from time step k, z(k). The order of
                measurements is z = [z_Ux, z_Uy, z_Sx, z_Sy, z_Sz], where
                z_Ux, z_Uy are the USBL measurements (np.nan during dropout)
                and z_Sx, z_Sy, z_Sz are the sonar measurements.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The posterior state mean at time step k. The order of
                states is x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
            Pm : np.ndarray, dim: (num_states, num_states)
                The posterior state covariance at time step k.
        """
        xm = None
        Pm = None

        return xm, Pm


class NonGaussianEstimator:
    """
    Estimator for the whale tracking problem under non-Gaussian noise.

    You are free to choose any recursive estimation strategy you like for this
    class. The public interface only exposes a single state estimate; the
    internal representation of your filter is up to you.

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
    """

    def __init__(
            self,
            estimator_constant: EstimatorConstant,
    ):
        self.constant = estimator_constant

    def initialize(self) -> np.ndarray:
        """
        Initialize the estimator. Set up any internal state required by the
        filter and return the initial state estimate.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The initial state estimate. The order of states is
                x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
        """
        xm = None

        return xm

    def estimate(
            self,
            measurement: np.ndarray,
    ) -> np.ndarray:
        """
        Update the estimator with a new measurement and return the posterior
        state estimate at the current time step.

        Args:
            measurement : np.ndarray, dim: (num_measurements,)
                Sensor measurements from time step k, z(k). The order of
                measurements is z = [z_Ux, z_Uy, z_Sx, z_Sy, z_Sz], where
                z_Ux, z_Uy are the USBL measurements (np.nan during dropout)
                and z_Sx, z_Sy, z_Sz are the sonar measurements.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The posterior state estimate at time step k. The order of
                states is x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
        """
        xm = None

        return xm
