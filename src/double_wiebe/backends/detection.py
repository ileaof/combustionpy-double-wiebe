# -*- coding: utf-8 -*-
"""
detection.py — Detecção de hardware e backends disponíveis.

Usado por `double-wiebe devices`, pela GUI (aba Desempenho) e pela seleção
automática (`select_backend("auto")`). Bibliotecas opcionais (cupy,
pyopencl) são importadas APENAS aqui — nunca no núcleo científico —
cumprindo a regra de dependências opcionais.
"""
from __future__ import annotations

import os
import platform


def detect_cpu() -> dict:
    """Informações de CPU (núcleos, modelo)."""
    nome = platform.processor() or "desconhecido"
    return {
        "cpu": nome,
        "n_cores": os.cpu_count() or 1,
        "python": platform.python_version(),
        "plataforma": platform.platform(),
    }


def detect_numba() -> dict:
    """numba (extra [cpu]): versão e disponibilidade."""
    try:
        import numba
        return {"disponivel": True, "versao": numba.__version__}
    except ImportError:
        return {"disponivel": False, "versao": None}


def detect_cuda() -> dict:
    """GPUs CUDA via CuPy (opcional). Nunca lança exceção."""
    try:
        import cupy                                    # opcional [cuda]
        n = int(cupy.cuda.runtime.getDeviceCount())
        if n <= 0:
            return {"disponivel": False, "gpus": []}
        gpus = []
        for i in range(n):
            props = cupy.cuda.runtime.getDeviceProperties(i)
            nome = props["name"].decode("utf-8", "replace") \
                if isinstance(props["name"], bytes) else str(props["name"])
            gpus.append(nome)
        return {"disponivel": True, "gpus": gpus}
    except ImportError:
        return {"disponivel": False, "gpus": [],
                "motivo": "cupy não instalado (pip install "
                          "double-wiebe[cuda])"}
    except Exception as e:                             # driver ausente etc.
        return {"disponivel": False, "gpus": [],
                "motivo": f"{type(e).__name__}: {e}"}


def detect_opencl() -> dict:
    """Dispositivos OpenCL via PyOpenCL (opcional). Detecta suporte FP64."""
    try:
        import pyopencl as cl                          # opcional [opencl]
        devs = []
        for p in cl.get_platforms():
            for d in p.get_devices():
                fp64 = "cl_khr_fp64" in d.extensions or \
                    "cl_amd_fp64" in d.extensions
                devs.append({
                    "nome": d.name.strip(),
                    "tipo": str(cl.device_type.to_string(d.type)),
                    "fp64": fp64,
                })
        return {"disponivel": len(devs) > 0, "dispositivos": devs}
    except ImportError:
        return {"disponivel": False, "dispositivos": [],
                "motivo": "pyopencl não instalado (pip install "
                          "double-wiebe[opencl])"}
    except Exception as e:                             # driver/plataforma
        return {"disponivel": False, "dispositivos": [],
                "motivo": f"{type(e).__name__}: {e}"}


def detect_hardware() -> dict:
    """Retrato completo do hardware desta máquina."""
    return {
        "cpu": detect_cpu(),
        "numba": detect_numba(),
        "cuda": detect_cuda(),
        "opencl": detect_opencl(),
    }


def hardware_report() -> str:
    """Relatório de hardware em texto (CLI `devices` e GUI)."""
    h = detect_hardware()
    linhas = [
        f"CPU: {h['cpu']['cpu']}",
        f"Núcleos lógicos: {h['cpu']['n_cores']}",
        f"Python: {h['cpu']['python']}",
        f"Plataforma: {h['cpu']['plataforma']}",
        "",
        f"numba (extra [cpu]): "
        f"{h['numba']['versao'] if h['numba']['disponivel'] else 'nao instalado — backends acelerados usam so NumPy'}",
        "",
        "CUDA: " + (
            " | ".join(h["cuda"]["gpus"])
            if h["cuda"]["disponivel"]
            else "indisponível"
                 + (f" ({h['cuda'].get('motivo')})"
                    if h["cuda"].get("motivo") else " (sem GPU detectada)")),
        "OpenCL: " + (
            "; ".join(f"{d['nome']} (FP64: {'sim' if d['fp64'] else 'NÃO'})"
                      for d in h["opencl"]["dispositivos"])
            if h["opencl"]["disponivel"]
            else "indisponível"
                 + (f" ({h['opencl'].get('motivo')})"
                    if h["opencl"].get("motivo") else " (sem dispositivo)")),
    ]
    return "\n".join(linhas)