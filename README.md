# Static Analysis for Differentiability

### Overview

The static analyser implemented in `diffai/` performs the following tasks:

- Verify the differentiability, and the local Lipschitz continuity, of a given Pyro/Python program.
- Based on the verification result, transform the analysed program automatically to an optimised program that better leverages Pyro's inference engine.

To evaluate the static analyser, we applied it to the eight Pyro examples in `srepar/srepar/examples/`.
You can reproduce this experiment by running `cd diffai; make run-diff-examples`.
