# ALPS

The mAb assembly pipeline (`dnps_hybrid/assembly.py`) shells out to **ALPS**, a
third-party peptide de novo sequencing assembler, via `java -jar ALPS.jar <csv> <k> <c>`.

ALPS is **not redistributed with this repository**. To run the assembly step,
obtain `ALPS.jar` from its authors/distributors and either:

- place it at `tools/ALPS/ALPS.jar` (the default location), or
- set the `DNPS_ALPS_JAR` environment variable to its path.

A Java runtime (`java`) must be on `PATH`. If the jar is missing, `assembly.py`
raises a `FileNotFoundError` pointing back here.
