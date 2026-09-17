# -*- coding: utf-8 -*-
"""
Integradores do núcleo Double Wiebe.

* ``scipy_integrator`` — implementação de referência (solve_ivp adaptativo,
  DOP853 por padrão); usada pelo backend ``serial`` e pelo re-run final de
  validação.
* ``rk4_numpy`` / ``rk4_numba`` — RK4 de passo fixo EM LOTE: integra N
  candidatos simultaneamente (vetorizado sobre a população do DE/PSO).
  Equações idênticas às de ``thermodynamics.py``; muda apenas o integrador.
* ``rk4_cuda`` / ``rk4_opencl`` — portas opcionais para GPU (import
  protegido; exigem hardware e bibliotecas instaladas).
"""
from .scipy_integrator import integrate_ode, safe_integrate  # noqa: F401
from .rk4_numpy import batch_integrate, batch_integrate_np    # noqa: F401


def get_integrator(nome: str):
    """Retorna a função batch_integrate do integrador pedido.

    nome: "scipy" (referência, não-batch), "rk4_numpy", "rk4_numba",
    "rk4_cuda", "rk4_opencl".  Lança BackendError (do pacote backends via
    import tardio para evitar ciclo) quando indisponível.
    """
    if nome == "scipy":
        return None                     # referência: usa solve_ivp
    if nome == "rk4_numpy":
        return batch_integrate_np
    if nome == "rk4_numba":
        from .rk4_numba import batch_integrate_numba
        return batch_integrate_numba
    if nome == "rk4_cuda":
        from .rk4_cuda import batch_integrate_cuda
        return batch_integrate_cuda
    if nome == "rk4_opencl":
        from .rk4_opencl import batch_integrate_opencl
        return batch_integrate_opencl
    raise ValueError(f"integrador desconhecido: '{nome}'")