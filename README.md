# agent-org-forms

A simulation of how to initialize a fixed group of 48 agents. The mix is specialists, generalists, coordinators, and communicators. The score is either discovery or revenue. Promotion rules and two structures, a flat group and modules, are part of the same design.

This is a simulation of a design. It is not an estimate of a real firm's optimal manager fraction.

The question is which role mix to start with, who coordinates, who makes, who communicates, and how promotion works, when the score switches from revenue before a shock to discovery after it. Two failure modes are in the score: too many cooks (a coordination gap that grows with pairwise links) and a mass of people idle. No opened source supplies an 80% idle law, and Pareto's income curve is not the claim that 20% of employees do 80% of the work.

Code: <https://github.com/dylanler/agent-org-forms>.

## How to run

Dependencies are numpy and matplotlib only.

```bash
python simulate.py
```

`simulate.py` is the entrypoint. With no argument, or with any argument other than `smoke`, it runs `smoke()` and then `full()`. The smoke path checks role counts, the promotion update, and a short 30-period run.

```bash
python simulate.py smoke
```

That runs only `smoke()`.

`full()` writes `results.md`, `results.json`, and `plots/`. Those paths are hardcoded to `/workspace/org-wisdom`. The main grid uses seeds 0 through 29. The alpha, gamma, beta1, and Graicunas runs use seeds 0 through 9. Reported ± values are the sample standard deviation across seeds with `ddof=1`.

`full()` calls `write_readme()` first and replaces this file with a generated design note. Copy this README aside if you want to keep it.

## Design parameters that are not fitted

These are set in the script. They are not estimated from data.

| Symbol | Code name | Value on the main run | Role |
| --- | --- | --- | --- |
| alpha | `ALPHA_MAIN` | 1 | Weight on the coordination gap |
| gamma | `GAMMA_MAIN` | -0.14 | Exponent on team size in effort |
| sigma | `SIGMA` | 0.5 | Noise on role traits |
| beta1 | `BETA1_MAIN` | -1 | Weight on depth when promotion sets coordination skill |
| iota_0 | `IOTA0` | 0.5 | Idle threshold inside `Y_net` only |

gamma = -0.14 is a Latané shouting-task anchor, not a universal constant. Discovery and revenue use output `Y`, not `Y_net`. iota_0 = 0.5 is a scenario knob. A median idle share of 0.8 was only a failure check, not a target.

## Results

Means below are from the main grid, 30 seeds, ± the seed sample standard deviation (`ddof=1`). Full tables are in [results.md](results.md) and [results.json](results.json). Figures are in [plots/](plots/).

On modules, the highest mean cumulative revenue was coordinator share 0.5, specialist share of the maker pool 0.75, and communicator share 0.1. That cell is 14 specialists, 5 generalists, 24 coordinators, and 5 communicators: cumulative revenue 113.8151 ± 12.0336, cumulative discovery 8.8004 ± 3.8421.

The highest mean cumulative discovery on modules was coordinator share 0.25, specialist share 0.75, and communicator share 0: cumulative discovery 36.7658 ± 9.7684, cumulative revenue 30.2240 ± 9.9102.

On the plotted slice (modules, specialist share 0.5, communicator share 0.1), cumulative revenue did not make an inverse-U in coordinator share. The means were 97.8542 ± 5.7335, 100.4960 ± 6.4454, 96.6813 ± 8.1965, 104.4592 ± 6.8621, and 108.9716 ± 10.2031, with the peak at coordinator share 0.5.

The within-run median idle share reached 0.8 in 0 of 2700 cell-seed runs. The maximum within-run median was 0.2500.

Changing alpha and gamma flipped the ranking of coordinator share by cumulative revenue: 8 of 9 (alpha, gamma) orders differed from the default order. See the sweep table in results.md.

Again: these numbers describe this simulation. They are not an estimate of the right fraction of managers in a real firm.
