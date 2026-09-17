# -*- coding: utf-8 -*-
"""AppTest: GUI carrega sem exceções e a seção Desempenho existe."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


def main():
    at = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "src" / "double_wiebe"
            / "gui.py"), default_timeout=120)
    at.run()
    erros = [e.value for e in at.exception]
    assert not erros, f"exceções na GUI: {erros}"
    partes = ([x.value for x in at.markdown]
              + [x.label for x in at.select_slider]
              + [x.label for x in at.selectbox]
              + [x.label for x in at.expander]
              + [x.label for x in at.button])
    texto = "\n".join(str(p) for p in partes)
    assert "Desempenho" in texto, "secao Desempenho ausente na GUI"
    assert "Hardware detectado" in texto, "painel de hardware ausente"
    print("GUI OK: sem excecoes; secao Desempenho + Hardware presentes")
    return 0


if __name__ == "__main__":
    sys.exit(main())