# Static Analysis for Differentiability

The static analyser implemented in `diffai/` achieves the following goals:

- Verify the differentiability, and the local Lipschitz continuity, of a given Pyro/Python program.
- Based on the verification result, transform the analysed program automatically to an optimised program that better leverages Pyro's inference engine.

We have applied the analyser to the eight Pyro examples in `srepar/srepar/examples`.
You can reproduce this experiment by running `cd diffai; make run-diff-examples`.
