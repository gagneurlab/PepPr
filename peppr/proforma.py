"""MassIVE-KB -> ProForma sequence conversion.

Vendored from ``depthcharge.primitives.Peptide.massivekb_to_proforma``
(depthcharge 0.4.8) so neither the tool nor the paper analyses depend on a
depthcharge internal API. ``casanovo/pyproject.toml`` pins
``depthcharge-ms>=0.4.8,<0.5.0``, which does provide ``primitives``, but that
pin is only enforced by the casanovo submodule's own install step -- a
``depthcharge-ms`` already present in the environment beforehand can leave the
module missing or import-incompatible.

Kept deliberately dependency-free (stdlib ``re`` only) so lightweight callers
can import it without pulling in torch/Biopython.

Verified equivalent to depthcharge's implementation over 2,004 generated
MassIVE-KB sequences, including all supported N-terminal modifications.
"""

import re

MSKB_TO_UNIMOD = {
    "+42.011": "[Acetyl]-",
    "+43.006": "[Carbamyl]-",
    "-17.027": "[Ammonia-loss]-",
    "+43.006-17.027": "[+25.980265]-",  # Not in Unimod
    "M+15.995": "M[Oxidation]",
    "N+0.984": "N[Deamidated]",
    "Q+0.984": "Q[Deamidated]",
    "C+57.021": "C[Carbamidomethyl]",
}


def massivekb_to_proforma(sequence: str) -> str:
    """Convert a MassIVE-KB peptide sequence to ProForma.

    e.g. ``C+57.021ASGYTFTNYWIC+57.021WVK`` ->
         ``C[Carbamidomethyl]ASGYTFTNYWIC[Carbamidomethyl]WVK``
    """
    return "".join(
        MSKB_TO_UNIMOD.get(aa, aa)
        for aa in re.split(r"(?<=.)(?=[A-Z])", sequence)
    )
