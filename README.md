# realtime-rdk

Realtime Neurofeedback with Random Dot Kinematograms

## Gaussian-masked RDK stimulus

This repo includes a custom Gaussian-masked RDK stimulus in `src/rdk.py` used by `mib_training_v4.py`.

Key properties:

- Coherent signal dots move in the same direction; noise dots take a new random direction every frame.
- A radial Gaussian mask reduces dot opacity toward the edge of the field, making border dots less visible.
- Efficient drawing via PsychoPy `ElementArrayStim`.

Config mapping (pixels converted from degrees when read):

- `RDK.n_dots`, `RDK.dot_size`, `RDK.field_size`, `RDK.dot_life`
- Trial-defined `speed` and `direction` (deg)
- `coherence` per trial
- Optional `RDK.gauss_sigma` to override default sigma (defaults to radius/2)

The same class is used for the noise field with `coherence: 0.0`.

## TODO

[x] RNG for same direction sequences
[x] DVAs on the plots
[x] RTs for different speeds/coherences
[ ] BLINK detection
[ ] Equalize stimulus and background luminance
