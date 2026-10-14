# DRL4Sepsis: Offline Reinforcement Learning for Sepsis Treatment

This repository accompanies the following publication:

> **Architecture and evaluation design shape offline reinforcement learning for sepsis treatment across two ICU cohorts**\
> Mohammad Mahdi Azarbeik\*, Lorenz Kapral\*, Razvan Bologheanu, Richard Weiss, Emanuel Sallinger, Stefan J. Schaller, Oliver Kimberger†, Clemens Heitzinger†\
> *Submitted to npj Digital Medicine*\
> \* These authors contributed equally as first authors.\
> † These authors contributed equally as last authors.

---

## Overview

Reinforcement learning (RL) has been studied retrospectively for haemodynamic management in sepsis, but it is unclear how the learning architecture and the evaluation design affect reported policy performance. We reimplemented the widely used formulation of Komorowski et al. (2018) on MIMIC-IV, held the cohort definition, action space, reward, discount factor and evaluation framework fixed, and compared four offline RL agents with the clinician policy. All policies were trained on MIMIC-IV and applied without refitting to ViennaAIdb, an independent European ICU cohort.

The repository covers:
- Sepsis-3 cohort extraction and pre-processing for MIMIC-IV v3.1 and ViennaAIdb
- Three 900-state $K$-means state spaces (action-weighted, unweighted and mortality-weighted)
- A sparse autoencoder that provides the continuous state representation for the deep agents
- The four agents: DDP-DQN, DQN, action-weighted $Q$-learning and unweighted $Q$-learning
- An incremental analysis of the Rainbow DQN components and the two offline corrections (CQL, BCQ)
- Off-policy evaluation with FQE, DualDICE and GradientDICE in a state space independent of all policies
- Behavioural analyses that compare each policy's treatment decisions with the clinicians'

---

## Study design

**Cohorts.**
| Cohort | Role | ICU stays | Timesteps | 90-day mortality |
|---|---|---|---|---|
| MIMIC-IV v3.1, development split | Training and validation | 21,653 | 259,749 | 26.1% |
| MIMIC-IV v3.1, test split | Internal validation | 14,433 | 174,352 | 26.4% |
| ViennaAIdb | External validation | 8,368 | 110,585 | 16.6% |

**Markov decision process.**
- **Timestep:** 4 hours, from 25 h before to 49 h after sepsis onset (at most 20 timesteps per stay)
- **State:** 47 variables (demographics, vital signs, laboratory values, ventilation status and cumulative fluid quantities)
- **Action:** a 5×5 grid of IV fluid and vasopressor (norepinephrine-equivalent) dose bins, fitted on the development split and applied unchanged to ViennaAIdb
- **Reward:** +100 for 90-day survival and −100 for death, assigned at the last timestep; all intermediate rewards are zero
- **Discount factor:** γ = 0.99

**Agents.**
| Agent | Role | State space | Description |
|---|---|---|---|
| **DDP-DQN** | Proposed (deep) | 256-d sparse-autoencoder embedding | Dueling heads, double $Q$-learning and prioritised experience replay, with conservative $Q$-learning (CQL) and a batch-constrained (BCQ) action mask |
| DQN | Baseline (deep) | 256-d sparse-autoencoder embedding | Plain DQN without the optimisation components or the offline corrections |
| **Action-weighted $Q$-learning** | Proposed (tabular) | 900-state $K$-means, weighted by a random forest that predicts the clinician's action | Removes state–action pairs observed 15 times or fewer |
| Unweighted $Q$-learning | Baseline (tabular) | 900-state unweighted $K$-means | The original Komorowski formulation; removes pairs observed 5 times or fewer |

**Incremental component analysis.** Fourteen deep variants were each trained with the same 8 random seeds. Starting from a plain DQN, the Rainbow components (double $Q$-learning, dueling heads, prioritised experience replay, multi-step returns and distributional returns) were added individually and in combination, followed by CQL and BCQ. Each run was scored by its peak FQE value, the number of training trajectories needed to reach that peak, and its peak agreement with clinician actions.

**Off-policy evaluation.** The four policies and the clinician policy are evaluated in a third, mortality-weighted 900-state $K$-means space. This space is derived from a random forest that predicts 90-day mortality and does not belong to any of the policies being compared. Three estimators are used:
- Fitted $Q$-evaluation (FQE), cross-fitted over 5 folds split by ICU stay
- DualDICE
- GradientDICE with λ = 1

DualDICE and GradientDICE are solved exactly in the discrete state space. For each policy we also report its *extrapolated share*, the fraction of its projected treatment trajectory that lies at or after the first decision no clinician made in that state. The gap between the DualDICE and GradientDICE estimates (the *coverage bracket*) shows how much an estimate depends on how these unsupported decisions are valued. The clinician policy's on-policy return serves as a reference. Confidence intervals come from 500 ICU-stay bootstrap resamples, with the estimator refitted in each one. Weighted importance sampling is not reported, because its weights concentrate on a few trajectories at these trajectory lengths.

**Behavioural analyses.** We compare each policy with the clinicians using its action distribution, its agreement on treatment initiation (Cohen's κ, at stay and timestep level), its dose-bin differences per timestep, and its treatment rates conditional on SOFA score, mean arterial pressure and urine output.

---

## Main findings

- Among the retained Rainbow components, architecture mainly affected optimisation speed rather than peak estimated value. Dueling heads reached peak value 2.5-fold faster than a plain DQN, and DDP-DQN reached it 3.1-fold faster than a plain DQN with the same offline corrections.
- The offline corrections (CQL and BCQ) changed learnt behaviour more than the architectural components. They increased agreement with clinician actions by 7.7–8.6 percentage points.
- Only the two deep agents exceeded the clinician policy with non-overlapping 95% intervals under all three estimators in both cohorts. For action-weighted $Q$-learning on ViennaAIdb, the result depended on how unsupported decisions were valued.
- DDP-DQN best preserved the structure of clinicians' fluid and vasopressor decisions across both cohorts, and kept the smallest share of its projected trajectory outside the observed action support (1.2% on MIMIC-IV, 2.5% on ViennaAIdb).

---

## Code availability

The cleaned code for cohort extraction, model training, off-policy evaluation and analysis is being added to this repository. The repository will also include the derived cohort definitions, the action bin edges and the aggregate results underlying every figure and table in the paper.

The extraction and pre-processing pipeline was adapted from the Python reimplementation of Komorowski et al., available at: [https://github.com/cmudig/AI-Clinician-MIMICIV](https://github.com/cmudig/AI-Clinician-MIMICIV)

Related repository on DICE estimators for sepsis: [https://github.com/Reinforcement-Learning-TU-Vienna/dice_rl_sepsis](https://github.com/Reinforcement-Learning-TU-Vienna/dice_rl_sepsis)

The analyses used Python 3.10.16, PyTorch 2.5.1, scikit-learn 1.7.2 and NumPy 2.2.6.

---

## Data

- **MIMIC-IV v3.1** is available to credentialed users on PhysioNet: [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)
- **ViennaAIdb** contains de-identified clinical data from a single centre (Medical University of Vienna). Because of data protection regulations, it cannot be made publicly available.

---

## Contact

For questions regarding this work, please contact:

**Mohammad Mahdi Azarbeik**\
Medical University of Vienna, Department of Anaesthesia, Intensive Care Medicine and Pain Medicine\
TU Wien, Institute for Logic and Computation\
Ludwig Boltzmann Institute for Digital Health and Patient Safety\
[mohammad.azarbeik@tuwien.ac.at](mailto:mohammad.azarbeik@tuwien.ac.at)

---

## Citation

If you use this work, please cite:

```
Azarbeik, M.M.*, Kapral, L.*, Bologheanu, R., Weiss, R., Sallinger, E., Schaller, S.J., Kimberger, O.†, & Heitzinger, C.†
Architecture and evaluation design shape offline reinforcement learning for sepsis treatment across two ICU cohorts.
Submitted to npj Digital Medicine.
```

---

## License

This project is released under the MIT License; see [LICENSE](LICENSE).
