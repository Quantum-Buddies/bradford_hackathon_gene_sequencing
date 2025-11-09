from __future__ import annotations

from typing import Callable, Tuple, Optional, List
import numpy as np
import warnings


class PositionOperator:
    """Position operator x̂ in binary representation.

    Matches the convention used in the QD‑HMC reference implementation.
    """

    def __init__(self, qubits: int):
        self.d = qubits
        self.N = 2 ** qubits

    def eigenvalue(self, k: int) -> float:
        return np.sqrt(2 * np.pi / self.N) * (k - self.N / 2)

    def domain(self) -> np.ndarray:
        return np.array([self.eigenvalue(k) for k in range(self.N)])

    def encode_state(self, x: float) -> int:
        domain = self.domain()
        return int(np.argmin(np.abs(domain - x)))


class MomentumOperator:
    """Momentum operator p̂ via centered QFT relation: p̂ = F†_c x̂ F_c"""

    def __init__(self, qubits: int):
        self.pos_op = PositionOperator(qubits)

    def eigenvalue(self, k: int) -> float:
        return self.pos_op.eigenvalue(k)

    def domain(self) -> np.ndarray:
        return self.pos_op.domain()


def _binary_indices(k: int, n_qubits: int) -> List[int]:
    """Indices of qubits that are 0 in the binary representation of k.

    Used to apply X gates before multi-controlled operations.
    """
    binary = format(k, f"0{n_qubits}b")
    return [i for i, bit in enumerate(binary) if bit == "0"]


class _QDKernel:
    """Minimal QD‑HMC kernel with numeric phases (no parameter binding).

    Requires Qiskit. Builds a Trotterized circuit and samples proposals via
    statevector probabilities.
    """

    def __init__(
        self,
        energy_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        precision: int,
        t: float,
        r: int,
        num_vars: int,
        eta_mu: float | None = None,
        eta_sigma: float | None = None,
        lambda_mu: float | None = None,
        lambda_sigma: float | None = None,
    ) -> None:
        self.energy_fn = energy_fn
        self.gradient_fn = gradient_fn
        self.precision = precision
        self.t = t
        self.r = r
        self.num_vars = num_vars

        if precision > 3:
            warnings.warn(
                f"precision={precision} leads to 2^{precision} basis states per var; "
                "simulation may be infeasible.",
                RuntimeWarning,
            )

        # Try to import Qiskit components lazily
        try:
            from qiskit import QuantumCircuit, QuantumRegister
            from qiskit.circuit.library import QFT
            from qiskit.quantum_info import Statevector
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Qiskit is required for QMCMC. Install with `pip install qiskit`."
            ) from e
        self._QuantumCircuit = QuantumCircuit
        self._QuantumRegister = QuantumRegister
        self._QFT = QFT
        self._Statevector = Statevector

        # Operators per variable
        self.pos_ops = [PositionOperator(precision) for _ in range(num_vars)]
        self.mom_ops = [MomentumOperator(precision) for _ in range(num_vars)]

        # Randomization hyperparameters (as in QD‑HMC)
        self.eta_mu = 0.5 if eta_mu is None else float(eta_mu)
        self.eta_sigma = 0.4 if eta_sigma is None else float(eta_sigma)
        self.lambda_mu = 0.5 if lambda_mu is None else float(lambda_mu)
        self.lambda_sigma = 0.4 if lambda_sigma is None else float(lambda_sigma)

    # ---- Qiskit helpers ----
    def _centered_qft(self, qc, qubits: List, inverse: bool = False):
        qc.x(qubits[0])
        qc.append(self._QFT(len(qubits), inverse=inverse, do_swaps=True), qubits)
        qc.x(qubits[0])

    def _encode_gradient_oracle(self, qc, qubits: List, grad_fn_1d: Callable[[float], float], lam_val: float):
        n_qubits = len(qubits)
        domain = self.pos_ops[0].domain()  # all vars share same precision grid
        for k in range(len(domain)):
            x_k = domain[k]
            grad_k = float(grad_fn_1d(x_k))
            phase = lam_val * grad_k

            if k == 0:
                for q in qubits:
                    qc.x(q)
                qc.p(phase, qubits[-1])
                for q in qubits:
                    qc.x(q)
            else:
                for idx in _binary_indices(k, n_qubits):
                    qc.x(qubits[idx])
                qc.mcp(phase, qubits[:-1], qubits[-1])
                for idx in _binary_indices(k, n_qubits):
                    qc.x(qubits[idx])

    def _kinetic_evolution(self, qc, qubits: List, eta_val: float):
        # Transform to momentum basis
        self._centered_qft(qc, qubits, inverse=False)
        # Apply diagonal phase in momentum basis
        momenta = self.mom_ops[0].domain()
        n_qubits = len(qubits)
        for k in range(len(momenta)):
            p_k = momenta[k]
            phase = -eta_val * (p_k ** 2) / 2.0
            for idx in _binary_indices(k, n_qubits):
                qc.x(qubits[idx])
            qc.mcp(phase, qubits[:-1], qubits[-1])
            for idx in _binary_indices(k, n_qubits):
                qc.x(qubits[idx])
        # Back to position basis
        self._centered_qft(qc, qubits, inverse=True)

    def _potential_evolution(self, qc, qubits: List, grad_fn_1d: Callable[[float], float], lam_val: float):
        self._encode_gradient_oracle(qc, qubits, grad_fn_1d, lam_val)

    def _make_grad_fn_for_var(self, var_idx: int) -> Callable[[float], float]:
        def grad_1d(x_val: float) -> float:
            full = np.zeros(self.num_vars, dtype=np.float64)
            full[var_idx] = x_val
            g = self.gradient_fn(full)
            return float(g[var_idx])
        return grad_1d

    def _build_evolution_circuit(self) -> object:
        """Create a Qiskit circuit implementing r Trotter steps."""
        qc = self._QuantumCircuit(*[self._QuantumRegister(self.precision, name=f"x{i}") for i in range(self.num_vars)])

        dt = self.t / self.r
        eta = np.clip(np.random.normal(self.eta_mu, self.eta_sigma), 0.01, 2.0) * dt
        lam = np.clip(np.random.normal(self.lambda_mu, self.lambda_sigma), 0.01, 2.0) * dt

        for _ in range(self.r):
            for var_idx in range(self.num_vars):
                qubits = list(qc.qregs[var_idx])
                self._kinetic_evolution(qc, qubits, eta)
                self._potential_evolution(qc, qubits, self._make_grad_fn_for_var(var_idx), lam)

        # Momentum flip M = F†_c X_0 F_c per variable
        for var_idx in range(self.num_vars):
            qubits = list(qc.qregs[var_idx])
            self._centered_qft(qc, qubits, inverse=False)
            qc.x(qubits[0])
            self._centered_qft(qc, qubits, inverse=True)

        return qc

    def _prepare_initial_state(self, qc, current_state: np.ndarray):
        # Initialize each variable register to nearest position basis state
        for var_idx in range(self.num_vars):
            k = self.pos_ops[var_idx].encode_state(float(current_state[var_idx]))
            binary = format(k, f"0{self.precision}b")
            for bit_idx, bit in enumerate(binary):
                if bit == "1":
                    qc.x(qc.qregs[var_idx][bit_idx])

    def _decode_bitstring(self, bitstring: str) -> np.ndarray:
        bitstring = bitstring[::-1]  # Qiskit little-endian handling
        out = []
        for i in range(self.num_vars):
            start = i * self.precision
            end = start + self.precision
            k = int(bitstring[start:end], 2)
            x = self.pos_ops[i].eigenvalue(k)
            out.append(x)
        return np.array(out, dtype=np.float64)

    def one_step(
        self,
        current_state: np.ndarray,
        seed: Optional[int] = None,
        save_circuit_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        if seed is not None:
            np.random.seed(seed)
        qc = self._build_evolution_circuit()

        # Compose with initial state preparation
        init_qc = self._QuantumCircuit(*qc.qregs)
        self._prepare_initial_state(init_qc, current_state)
        full_qc = init_qc.compose(qc)

        # Optional: save circuit image
        if save_circuit_path:
            try:
                from qiskit.visualization import circuit_drawer  # type: ignore
                fig = circuit_drawer(full_qc, output='mpl')
                import os
                os.makedirs(os.path.dirname(str(save_circuit_path)), exist_ok=True)
                fig.savefig(str(save_circuit_path), dpi=150, bbox_inches='tight')
            except Exception as _e:
                pass

        # Sample from statevector probabilities
        sv = self._Statevector.from_instruction(full_qc)
        probs = np.abs(sv.data) ** 2
        idx = np.random.choice(len(probs), p=probs)
        measured = format(idx, f"0{self.num_vars * self.precision}b")
        proposal = self._decode_bitstring(measured)
        wf = probs.copy() if (self.num_vars * self.precision) <= 16 else None

        # Return proposal and its energy (no acceptance here)
        e_prop = self.energy_fn(proposal)
        return proposal, float(e_prop), wf


class QDHMCSampler:
    """Self-contained QMCMC sampler with a QD‑HMC‑style proposal (Qiskit only)."""

    def __init__(
        self,
        energy_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        num_vars: int,
        precision: int = 2,
        t: float = 1.0,
        r: int = 3,
        eta_mu: float | None = None,
        eta_sigma: float | None = None,
        lambda_mu: float | None = None,
        lambda_sigma: float | None = None,
    ) -> None:
        self.kernel = _QDKernel(
            energy_fn=energy_fn,
            gradient_fn=grad_fn,
            precision=precision,
            t=t,
            r=r,
            num_vars=num_vars,
            eta_mu=eta_mu,
            eta_sigma=eta_sigma,
            lambda_mu=lambda_mu,
            lambda_sigma=lambda_sigma,
        )
        self.num_vars = num_vars
        self.precision = precision

    def sample(
        self,
        burnin: int,
        samples: int,
        init: np.ndarray | None = None,
        seed: int | None = 42,
        progress: bool = False,
        log_every: int = 10,
        save_circuit_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, dict]:
        if init is None:
            init = np.zeros(self.num_vars, dtype=np.float64)
        np.random.seed(seed)

        draws = []
        energies = []
        wavefunctions = []
        accepted = 0

        x = init.astype(np.float64).copy()
        e = self.kernel.energy_fn(x)
        track_wf = (self.num_vars * self.precision) <= 16

        total = burnin + samples
        # Optional progress bar
        pbar = None
        if progress:
            try:
                from tqdm.auto import tqdm as _tqdm  # type: ignore
                pbar = _tqdm(total=total, desc="QMCMC", leave=True)
            except Exception:
                pbar = None

        circuit_saved = False
        for i in range(total):
            x_prop, e_prop, wf = self.kernel.one_step(
                x,
                save_circuit_path=(save_circuit_path if not circuit_saved else None),
            )
            circuit_saved = True or circuit_saved
            log_accept = e - e_prop
            if np.log(np.random.rand()) < log_accept:
                x = x_prop
                e = e_prop
                if i >= burnin:
                    accepted += 1
            if i >= burnin:
                draws.append(x.copy())
                energies.append(e)
                if track_wf and wf is not None:
                    wavefunctions.append(wf)

            # Update progress
            if pbar is not None:
                pbar.update(1)
                if i >= burnin:
                    cur_rate = accepted / max(1, (i + 1 - burnin))
                    pbar.set_postfix({"acc": f"{100*cur_rate:.1f}%"})
            elif progress and ((i + 1) % max(1, log_every) == 0):
                phase = "Burn-in" if i < burnin else "Sampling"
                cur_rate = accepted / max(1, (i + 1 - burnin)) if i >= burnin else 0.0
                print(f"[QMCMC] {phase}: {i+1}/{total} | Acc {100*cur_rate:.1f}% | E {e:.4f}", flush=True)

        draws = np.asarray(draws, dtype=np.float64)
        if pbar is not None:
            pbar.close()
        acc_rate = accepted / max(1, samples)
        diagnostics = {
            "energies": np.asarray(energies, dtype=np.float64),
            "acceptance_rate": float(acc_rate),
            "final_state": x.copy(),
            "wavefunctions": wavefunctions if (track_wf and wavefunctions) else None,
        }
        return draws, float(acc_rate), diagnostics

# ----------------------------------------------------------------------------
# CONFIGURATIONS FOR QMCMC (QD‑HMC style)
# ----------------------------------------------------------------------------

# These defaults mirror the reference configuration used in QD‑HMC demos and
# are provided here for discoverability. They are not applied automatically;
# pass explicit values from your entry point (run_qnlp_qmcmc.py).

DEFAULT_QMCMC_CONFIG = {
    "precision": 2,
    "t": 1.0,
    "r": 3,
    "eta_mu": 0.5,
    "eta_sigma": 0.4,
    "lambda_mu": 0.5,
    "lambda_sigma": 0.4,
}

if __name__ == "__main__":  # pragma: no cover
    import json
    print("QMCMC (QD‑HMC) default configuration:")
    print(json.dumps(DEFAULT_QMCMC_CONFIG, indent=2))
