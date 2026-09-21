# -*- coding: utf-8 -*-
"""
Integradores do núcleo Double Wiebe.

* ``scipy_integrator`` — implementação de referência (solve_ivp adaptativo,
  DOP853 por padrão); usada pelo backend ``serial`` e pelo re-run final de
  validação.
* ``rk4_numpy`` / ``rk4_numba`` — RK4 de passo fixo EM LOTE: integra N
  candidatos simultaneamente (vetorizado sobre a população do DE/PSO).
  Equações idênticas às de ``thermodynamics.py``; muda apenas o integrador.

* ``rk4_cuda`` — o mesmo RK4 em lote na GPU (CuPy, extra ``.[cuda]``;
  backend ``cuda``). Um thread CUDA por candidato.

OpenCL: NÃO existe como integrador — o extra ``.[opencl]`` serve apenas à
detecção de hardware (``backends.detection``, ver docs/hpc.md §2).
"""
from .scipy_integrator import integrate_ode, safe_integrate  # noqa: F401
from .rk4_numpy import batch_integrate, batch_integrate_np    # noqa: F401


def get_integrator(nome: str):
    """Retorna a função batch_integrate do integrador pedido.

    nome: "scipy" (referência, não-batch), "rk4_numpy", "rk4_numba",
    "rk4_cuda" (requer CuPy + GPU). Para "rk4_opencl" lança BackendError
    com mensagem clara (ver docs/hpc.md).
    """
    if nome == "scipy":
        return None                     # referência: usa solve_ivp
    if nome == "rk4_numpy":
        return batch_integrate_np
    if nome == "rk4_numba":
        from .rk4_numba import batch_integrate_numba
        return batch_integrate_numba
    if nome == "rk4_cuda":
        from .rk4_cuda import batch_integrate_cuda, cuda_available
        if not cuda_available():
            from ..backends.base import BackendNotAvailableError
            raise BackendNotAvailableError(
                "Integrador 'rk4_cuda' indisponível: requer CuPy e uma GPU "
                "NVIDIA (pip install -e \".[cuda]\").")
        return batch_integrate_cuda
    if nome == "rk4_opencl":
        from ..backends.base import BackendError
        raise BackendError(
            "Integrador 'rk4_opencl' não existe: OpenCL não é backend de "
            "execução do Double Wiebe (apenas detecção de hardware; "
            "justificativa em docs/hpc.md §2).")
    raise ValueError(f"integrador desconhecido: '{nome}'")