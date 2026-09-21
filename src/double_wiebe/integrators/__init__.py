# -*- coding: utf-8 -*-
"""
Integradores do núcleo Double Wiebe.

* ``scipy_integrator`` — implementação de referência (solve_ivp adaptativo,
  DOP853 por padrão); usada pelo backend ``serial`` e pelo re-run final de
  validação.
* ``rk4_numpy`` / ``rk4_numba`` — RK4 de passo fixo EM LOTE: integra N
  candidatos simultaneamente (vetorizado sobre a população do DE/PSO).
  Equações idênticas às de ``thermodynamics.py``; muda apenas o integrador.

CUDA/OpenCL: NÃO existem como integradores/backends de execução — decisão
documentada em docs/hpc.md (§2). Os extras ``.[cuda]``/``.[opencl]``
instalam cupy/pyopencl APENAS para a detecção de hardware
(``backends.detection``, comando ``devices`` e painel da GUI).
"""
from .scipy_integrator import integrate_ode, safe_integrate  # noqa: F401
from .rk4_numpy import batch_integrate, batch_integrate_np    # noqa: F401


def get_integrator(nome: str):
    """Retorna a função batch_integrate do integrador pedido.

    nome: "scipy" (referência, não-batch), "rk4_numpy", "rk4_numba".
    Para "rk4_cuda"/"rk4_opencl" lança BackendError com mensagem clara —
    GPU não é caminho de execução do pacote (ver docs/hpc.md).
    """
    if nome == "scipy":
        return None                     # referência: usa solve_ivp
    if nome == "rk4_numpy":
        return batch_integrate_np
    if nome == "rk4_numba":
        from .rk4_numba import batch_integrate_numba
        return batch_integrate_numba
    if nome in ("rk4_cuda", "rk4_opencl"):
        from ..backends.base import BackendError
        raise BackendError(
            f"Integrador '{nome}' não existe: CUDA/OpenCL não são "
            "backends de execução do Double Wiebe (apenas detecção de "
            "hardware; justificativa em docs/hpc.md §2).")
    raise ValueError(f"integrador desconhecido: '{nome}'")