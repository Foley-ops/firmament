# Bug court — 2026-09-25 (HEAD fc90e3d)

Method: 4 prosecutors (life kernel, environment physics, history/persistence, observation/interfaces) had to reproduce each accusation with a script on the current code, CPU-only. An independent defense agent re-ran every repro and tried to acquit (intended behavior, flawed repro, unreachable state). 12 accusations, 12 guilty (10 distinct bugs: meteor and 'failed command poisons the log' were each found twice). Repro scripts are in `evidence/` (paths below are relative to it). Run from the repo root with `CUDA_VISIBLE_DEVICES="" uv run python <script>`.

## 1. A binder partner link is never cleared: after the other polymer dies or is reused, the survivor stays FREE with a stale p_partner and can never be copied or bind again

- **Verdict:** guilty — severity major → major; reachable in normal runs: True
- **Where:** `firmament/core/polymers.py:147 (the hydrolysis branch releases a partner only when it is P_BOUND); also polymers.py:389 (copy start overwrites p_partner[p] without clearing the binder partner written at 194-195)`
- **Expected:** When one member of a binder aggregate dies, or leaves the pair by starting a copy or being bound as a template, the other member's partner link should be released. The kernel already does this for copier/template pairs (hydrolysis lines 146-150; the abort path is commented 'release everyone'). A FREE polymer should never hold p_partner pointing at a dead slot, or at a recycled slot that now holds an unrelated polymer. It should also stay eligible as a copy template: the templating rule in spec §4.4 says a replicase 'can bind an adjacent polymer', and only P_BOUND/P_COPYING mean 'busy'. *(source: k_cell_pass's own partner-release logic in hydrolysis (lines 146-150) and the abort comment 'release everyone'; milestones._binder_pairs defines a binder pair as MUTUAL partners; FIRMAMENT_Engineering_Writeup §4.4 (templated copying of any adjacent polymer; binder = aggregate); the task's 'states that can never be freed' criterion)*
- **Actual:** The hydrolysis release checks p_state[tm]==P_BOUND. A binder partner is P_FREE, so it is never released. Copy start sets p_partner[p]=tmpl and leaves the partner pointing at p. Template selection accepts only p_partner<0, and binding needs p_partner<0. So the orphaned polymer is permanently removed from heredity. Its pointer later aims at whatever recycled polymer ends up in that slot (ids 3 and 810 in the repro). Natural run with no state injection (shipped mutation rate 0.005, seed = replicase+binder+pad): 5 of 5 orphans observed for 100 ticks or more were never templated again, against 4 of 5 in the p_partner==-1 control. Controlled run: X was never templated in the 1486 ticks after its partner Y died. In the counterfactual arm (X.p_partner set to -1 at Y's death), X was copied, paired with its own copy (id 5), and was orphaned again when that copy died (slot 5 was later reused by id 828). Element totals stayed exact in both arms, so this is a state-machine bug, not a matter leak.
- **Evidence:** `life/repro_binder_dead_partner.py`

```
AS SHIPPED (no release): X templated 0 times after Y died
  tick 14: Y hydrolysed (slot 2 state=0); X state=1 X.p_partner=slot 2
  tick  314: X state=1 X.p_partner=slot 2 (slot 2 now holds id 3, state 3, its p_partner 42) | X bound as template 0x since Y died | live=137
  tick  914: X state=1 X.p_partner=slot 2 (slot 2 now holds id 810, state 5, its p_partner -1) | X bound as template 0x since Y died | live=149
  tick 1500: X state=1 X.p_partner=slot 2 (slot 2 now holds id 810, state 2, its p_partner 255) | X bound as template 0x since Y died | live=144
  element totals exact: True
COUNTERFACTUAL (X.p_partner released at Y's death): X templated 1 times after Y died
  tick 1500: X state=1 X.p_partner=slot 5 (slot 5 now holds id 828, state 2, its p_partner 247) | X bound as template 1x since Y died
--- repro_binder_orphan.py (natural, mutation 0.005, no injection):
ORPHANED (p_partner points to a polymer that does not point back): 5 non-replicase FREE polymers observed >= 100 ticks; 0 of them were later bound as a copy template (0%)
CONTROL  (p_partner == -1): 5 non-replicase FREE polymers observed >= 100 ticks; 4 of them were later bound as a copy template (80%)
  id=    5 orphaned@tick 149  now state=1 p_partner=slot 3 (that slot: state=3, its p_partner=569, points back: False)  templated since: False
```

**Defense (re-ran, tried to acquit):** Guilty. I tried to acquit and could not.

1. It reproduces. I re-ran both of the prosecutor's scripts with the exact CPU-only command. The output matches the claim: X keeps X.p_partner=slot 2 from tick 14 to tick 1500. That slot is first dead, then reused by id 3 and later id 810. X is templated 0 times, against 1 time in the counterfactual arm. Element totals stay exact.

2. The code confirms the mechanism (firmament/core/polymers.py):
- Lines 188-196: the binder effect writes a mutual link, p_partner[p]=p2 and p_partner[p2]=p, and leaves both polymers P_FREE.
- Lines 146-150: hydrolysis releases a partner only if `p_state[tm]==P_BOUND and p_partner[tm]==p`. A FREE binder partner is never released.
- Lines 368-389: copy start checks `(motifs & repl_bit)` but never checks p_partner[p]<0. A paired replicase+binder can therefore start copying, and line 389 overwrites p_partner[p] without clearing the partner's back-link.
- Lines 372/380 (template choice) and 189/192 (binding) both require p_partner<0.
- Nothing ever clears a FREE polymer's p_partner except its own death (line 150). The completion and abort paths clear only the copier and the template.
So a FREE non-replicase with a stale link can never be templated or bound again. A replicase escapes because starting a copy overwrites its link, which is why the effect is specific to non-replicases.

3. Defense attempts that failed:
(a) Intended "sticky aggregate"? DECISIONS.md, RUNBOOK, the writeup §4.4, docstrings and the git log contain nothing saying a binder link should outlive its partner. The writeup's state enum is "free · bound-to-template · being-copied · membrane-attached", with no permanently-stuck-free state. milestones._binder_pairs defines a pair as MUTUAL partners. ecology counts (p_partner>=0)//2, which assumes links are symmetric. A pointer into a dead slot, or into a reused slot now holding an unrelated polymer, has no physical meaning. My run also saw 3 samples at 288 K and 6 at 312 K where the stale link's target was mutually paired with a third polymer.
(b) Flawed repro? The controlled repro injects the X<->Y pair and raises hydrolysis to 3e-6. So I wrote my own runs (defense_check2.py) with no injection: the seed was placed via place_seed, which is the CLI seed path, with shipped hydrolysis 1e-7 and shipped mutation 0.005.
- At 288 K: 17 polymers gained a one-way stale link when their replicase partner started copying. Of the 13 followed for at least 50 ticks, 0 were templated and 0 had the link reset. In the control group, 19 of 20 were templated.
- At 312 K: stale links to DEAD slots (11 polymers) and to RECYCLED slots (6 polymers) occurred naturally. Of 32 followed, 0 were templated and 0 were reset. In the control group, 6 of 7 were templated.
(c) Reachability: the shipped SEED_PROPOSAL seed has no binder motif, so the bug needs either a binder in the seed or a binder that evolves. Two things make it reachable anyway. The seed command accepts any sequence. The binder is part of the frozen genetic_code_v0, and two milestone detectors (candidate_aggregate, candidate_predation) exist to catch binder evolution. Once binders exist, the bug fires within a few hundred ticks.
(d) Not on the excluded list: it is not thermodynamics, not a dt, cell_meters or GUI issue, not style and not performance.

Severity: major. It adds a rule that is not in the spec. Any non-replicase binder carrier whose partner starts copying or dies is permanently removed from heredity, which biases selection against the binder motif and against templating any adjacent polymer (§4.4). It also leaves dangling slot references and skews ecology.bound_pairs. No crash and no conservation leak, which is why it is not critical.

Evidence files: /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-life-binder-partner-never-released/defense_check2.py and defense_check.py.

## 2. place_seed writes the seed to the causal log before checking its length: a seed over max_length permanently bricks the run, and a 0- or 1-monomer seed recreates the sub-MIN_LEN 'polymer' that v0.2 removed

- **Verdict:** guilty — severity major → minor; reachable in normal runs: True
- **Where:** `firmament/operator/console.py:171 (commit_now runs before any length check; place_seed lines 158-173 check only monomer counts and slot 0); crash at console.py:195 (sq[slot, :len(seq)] = seq); id hard-coded at console.py:190`
- **Expected:** place_seed says it is 'Validated first, then committed to the causal log'. A seed must satisfy MIN_LEN (2) <= len <= polymers.max_length before it is committed. A rejected seed must leave events.jsonl untouched and the run resumable. RUNBOOK says a resume is 'indistinguishable from never having stopped'. *(source: console.place_seed docstring; DECISIONS 2026-09-25 'MIN_LEN = 2. Fewer than two units ... is not a polymer ... (v0.1 allowed zero-atom, immortal, free "polymers")'; causal.py docstring (commands are committed and then applied); RUNBOOK Resume section; the 'seed must be the first polymer' guard at console.py:169)*
- **Actual:** (1) Running `firmament seed` with a 302-monomer sequence (max_length 256) writes seed_placed n=0 to events.jsonl, then apply_seed raises ValueError. The corrected 30-mer seed is then refused ('cannot commit a new command while logged history is replaying'), and every `firmament resume` re-applies the bad record and crashes the same way. The run cannot be recovered without hand-editing the append-only causal log. (2) A seed of '' is accepted and becomes a FREE p_len=0 zero-atom polymer that is still alive after 2000 ticks, because hydrolysis pdie = 1-exp(-kh*0*dt) = 0. That is exactly the v0.1 artifact, and the instruments count it as a completed individual. A seed of 'M1' gives a p_len=1 polymer. (3) Related gap in the same guard: after the first lineage dies out, a second, different seed is accepted and gets p_id 1 again, while lineage.sqlite already records children 2 and 3 of the original id 1.
- **Evidence:** `life/repro_seed_brick_cli.sh`

```
=== 1) firmament seed, 302-monomer sequence
ValueError: could not broadcast input array from shape (302,) into shape (256,)
events.jsonl: [('seed_placed', 0, 3)]
=== 2) corrected 30-mer seed
RuntimeError: cannot commit a new command while logged history is replaying
=== 3) resume --ticks 2
ValueError: could not broadcast input array from shape (302,) into shape (256,)
=== 4) resume again
ValueError: could not broadcast input array from shape (302,) into shape (256,)
crash.json: ValueError('could not broadcast input array from shape (302,) into shape (256,)') at tick 3
--- repro_seed_validation.py:
(a) zero-length seed '': accepted; after 2000 ticks slot0 state=1 (1=FREE) p_len=0 id=1 ; counted as completed polymer: True
(b) length-1 seed 'M1': accepted; after 2000 ticks slot0 state=1 (1=FREE) p_len=1 id=1 ; counted as completed polymer: True
   events.jsonl records: [('seed_placed', 0, 302)] ; state.causal_n = 0
   tick 65: every polymer dead; lineage has 3 copies, children of id 1: [2, 3] ; next_poly_id = 10
   second place_seed accepted; seed_placed records: 2 ; new seed p_id = 1 (same id as the first seed); next_poly_id = 10
```

**Defense (re-ran, tried to acquit):** I re-ran both of the prosecutor's scripts myself, CPU-only, and every claim reproduced exactly.

Code facts (firmament/operator/console.py):
- place_seed (lines 158-173) says "Validated first, then committed to the causal log".
- The only checks are the per-monomer availability loop and `p_state[0] != 0`. There is no check against MIN_LEN or against polymers.max_length.
- causal.commit_now then calls events.append, which writes to events.jsonl and flushes, before _apply -> apply_seed.
- apply_seed crashes at `sq[slot, :len(seq)] = seq` when len > max_length.
- After that, load_replay requeues the unapplied record (n >= causal_n) on every resume. commit_now refuses new commands while replay_queue is non-empty. So that run_id can never be resumed or re-seeded.

Defense arguments I tried:
1. Intended or documented? No. The docstring promises validation before commit. DECISIONS 2026-09-25 says "MIN_LEN = 2 ... (v0.1 allowed zero-atom, immortal, free 'polymers')". M1_ATTEMPTS.md names zero-length polymers a "physics hole" that invalidated M1 attempt 1. The v0.2 fix enforces MIN_LEN only when a copy completes (polymers.py:336), never at seeding.
2. Only reachable through odd input? It is operator input, but config.py allows `max_length: int = Field(256, ge=2, le=256)`. My probe (B) used a valid config with max_length 48 and the APPROVED 60-mer seed from docs/SEED_PROPOSAL.md. It bricks the run the same way, with no typo involved. My probe (C) shows `firmament seed --sequence ''` is accepted by the real CLI. The resulting polymer is immortal (pdie = 1 - exp(-kh*0*dt) = 0) and the population instrument counts it (n_polymers=1, len_max=0 at ticks 100/200/300). That alone would satisfy the M1 criterion "population never zero".
3. Is "permanently bricked" overstated? Partly. fork_run cuts history at n < causal_n, so `firmament fork` of the bricked run gives a clean run with 0 event lines. My probe (A) placed the corrected seed in the fork and resumed it without a crash. No hand-editing of the log is needed, and the M1 protocol already forks before seeding. The world is recoverable, but the run_id stays dead and RUNBOOK's resume guarantee fails for it.
4. Part (d), re-seeding after extinction, goes against the build guide's "If the seed dies: new run" protocol. It needs operator misuse, but the duplicate p_id=1 reproduced.

Conclusion: the documented contract ("validated first, then committed") and the v0.2 MIN_LEN invariant are both broken, reachable through the public CLI and a valid config, and shown with verbatim output. Guilty.

I lowered the severity from major to minor:
- It needs operator-supplied input on a one-time, approval-gated command.
- The approved seed is valid under every shipped config (all use max_length 256).
- The overlong case fails loudly, and the world is recoverable with the documented fork tool.
- Normal runs with the approved seed are unaffected.

## 3. The photoactive (light) and sense[A] gates only affect the catalyst effect; membrane, emit (and, by the same code path, motor and binder) run at full strength in darkness and with [A]=0

- **Verdict:** guilty — severity minor → minor; reachable in normal runs: True
- **Where:** `firmament/core/polymers.py:186 (gate is applied only in the catalyst branch); ungated: membrane 198, emit 224, motor 203-221, binder 188-196`
- **Expected:** A polymer carrying photoactive should have its other motif effects scaled by local light, and one carrying sense_A should have them scaled by [A]/([A]+1000). In total darkness with [A]=0, a photoactive+membrane or sense_A+membrane polymer should accrete no L, and photoactive+emit_A should emit no A, just as photoactive+catalyst is silenced. *(source: configs/genetic_code_v0.yaml: photoactive 'photosystem (gates other motifs by light)', sense_A 'signal receptor (gates other motifs by [A])'; k_cell_pass motif-effect header comment 'gates: photoactive x light, sense x [S]'; Engineering Writeup §4.4 table: photoactive = 'Motif activity modulated by local light', sense[S] = 'activity modulated by that species')*
- **Actual:** Polymers-only sim with light_water = 0 everywhere and [A] = 0, each probe alone in its own cell with identical resources, 10 ticks: photoactive and sense_A both reduce the formose catalyst to 0.0 (vs 4.0 ungated). photoactive+membrane and sense_A+membrane accrete 39 L, the same as a plain membrane polymer. photoactive+emit_A emits 20 A, the same as plain emit_A. The gate variable reaches only the catalyst's add. Also, the gated-off catalyst still pays its full 10 P~P for zero effect.
- **Evidence:** `life/repro_gate_only_catalyst.py`

```
max light_water anywhere = 0.0 W/m^2 ;  [A] in every probe cell at start = 0
  catalyst_formose                 motifs=['catalyst_formose']
      P~P spent= 10  L accreted=  0  (membrane_store=0)  A emitted=  0  formose catalyst (last substep)=4.0
  photoactive + catalyst_formose   motifs=['catalyst_formose', 'photoactive']
      P~P spent= 10  L accreted=  0  (membrane_store=0)  A emitted=  0  formose catalyst (last substep)=0.0
  sense_A + catalyst_formose       motifs=['catalyst_formose', 'sense_A']
      P~P spent= 10  ...  formose catalyst (last substep)=0.0
  membrane                         motifs=['membrane']
      P~P spent= 20  L accreted= 39  (membrane_store=39)
  photoactive + membrane           motifs=['membrane', 'photoactive']
      P~P spent= 20  L accreted= 39  (membrane_store=39)
  sense_A + membrane               motifs=['membrane', 'sense_A']
      P~P spent= 20  L accreted= 39  (membrane_store=39)
  emit_A                           motifs=['emit_A']
      P~P spent= 10  ...  A emitted= 20
  photoactive + emit_A             motifs=['photoactive', 'emit_A']
      P~P spent= 10  ...  A emitted= 20
```

**Defense (re-ran, tried to acquit):** I could not find a defense that holds up, so I find the code guilty with minor severity.

1. **It reproduces.** The prosecutor's script produced exactly the claimed numbers when I re-ran it on CPU.

2. **The prosecutor's repro could have been flawed; independent tests rule that out.** I wrote two repros of my own. Neither does the manual slot and array injection the prosecutor used; both place polymers through the real `place_seed` → `apply_seed` console path.
   - `defense_gate_ab.py` runs each probe twice, as a self-controlled A/B: gate ≈ 1 (light_water = 300, [A] = 1e6) and gate = 0 (light_water = 0, [A] = 0).
     - The catalyst responds: 4.000 or 3.996 with the gate on, 0.000 with it off. So the gate is computed correctly and works.
     - Membrane (20 vs 20), emit_A (10 vs 10) and the motor's staged move (target cell 28 vs 28) are identical in both conditions.
   - `defense_natural_night.py` uses the full module stack with the real radiation module. `sun_cos_zenith` puts tick 0 at local midnight (cosz = 0), so light_water = 0 comes from the physics, with nothing injected.
     - photoactive+membrane accretes 79 L, the same as a plain membrane.
     - photoactive+catalyst gives a catalyst field of 0.0, against 80.0 for a plain catalyst.

3. **Code.** In `firmament/core/polymers.py`, lines 160-171 build a single gate per polymer as the product over all its photoactive and sense motifs. Line 159's header says "gates: photoactive x light, sense x [S]" for the whole effect block. But `gate` is read only on line 186, in the `eff == 1` catalyst branch. None of the other branches use it: binder (188-196), membrane (197-202), motor (203-221) or emit (222-242).

4. **Intent and documentation.** The documents contradict the implementation:
   - `configs/genetic_code_v0.yaml` says photoactive "gates other motifs by light" and sense_A "gates other motifs by [A]".
   - Engineering Writeup §4.4 says "Motif activity modulated by local light" and "activity modulated by that species".
   - Photoactive and sense have no effect of their own (line 176 skips them), so "motif activity" can only mean the activity of the polymer's other motifs.
   - Nothing narrows this to catalysts. DECISIONS.md (including the 2026-09-25 sections), RUNBOOK, PROGRESS, SEED_PROPOSAL, docstrings and tests contain no such statement, and no test covers gating at all.
   - It is not in the "NOT bugs" list. The thermodynamic-gating exclusion concerns reaction ΔG gating, a different mechanism, and photoactive and sense are v0 catalog motifs, not future work.

5. **Reachability.** Darkness happens naturally at every night, including tick 0, and [A] = 0 is the normal background. Polymers carrying photoactive or sense_A alongside membrane, emit, motor or binder are not in the approved seed, which has only replicase. They can arise through mutation, though: ε = 0.005 per monomer, and `scan_motifs` runs on every completed child. So the defect is latent but reachable in normal evolving runs. It never crashes and never breaks conservation; it only makes these motifs light- and signal-independent when they should not be. That is why I rate it minor, not major.

6. **Partial defense.** The side claim that a gated-off catalyst "still pays full P~P for zero effect" is arguable. The spec says "Every motif costs energy to express", so paying regardless of the gate could be read as an intended expression cost. I do not count that part as proven. The central claim does not depend on it, and that claim is proven.

## 4. Shallow-water momentum treats a dry bank's bed height as a water surface, so a still lake starts flowing on its own at tick 1 (m/s currents, distorted surface, erosion)

- **Verdict:** guilty — severity critical → critical; reachable in normal runs: True
- **Where:** `firmament/core/fluid.py:87-88 (k_momentum: ex/ey use elev+sed+hw of DRY neighbours in a centred difference)`
- **Expected:** A lake at rest (flat free surface, no forcing) stays at rest: zero velocity, flat surface, no sediment moved. terrain.generate sets up exactly this state (water = max(0, level - elev)). *(source: Basic hydrostatics: for the shallow-water equations, a flat free surface with u=0 is an exact steady state. Spec (Engineering Writeup section 3, Fluid): 'Shallow-water equations ... Runoff follows the shallow-water solve.' Control run: the same pond with no dry bank stays exactly at rest.)*
- **Actual:** The pressure-gradient term uses eta = elev+sed+hw of a dry bank cell whose bed sits above the water. That gives a large false slope at every shoreline cell. Together with collocated centred differences this drives a lasting odd-even sawtooth. Minimal pond (8x8, 1 m deep, 2 m bank): velocities of ±7.55 m/s in every row and a depth sawtooth of 0.575-1.619 m that never decays. Control without a bank: 0.0000 m/s. M1 world terrain (256x256, seed 7, vents from world_m1.yaml): after 3 ticks, max 5.6 m/s, 9069 cells above 0.5 m/s, surface spread 0.26 m. Each minute 523,515 m^3 crosses faces, about 9x the 57,688 m^3 lake. Sediment erodes by up to 3.9 mm in 3 ticks. The 32x32 test world reaches 40 m/s and 16x16 reaches 87 m/s, which also pins the CFL cap. The audits stay silent because mass and energy are still conserved.
- **Evidence:** `env/r2_step_bank.py (also r1_lake_at_rest.py [16|32], r3_m1_world_rest.py)`

```
--- bank=False: ...
tick  30: max|u|   0.0000 m/s  row0 u[0,:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
--- bank=True: water surface eta = 1.0 m everywhere wet; bank top = 2.0 m
tick   1: max|u|   7.5513 m/s  row0 u[0,:] = [-7.547, 7.549, -7.545, 7.55, -7.547, 7.551, -7.551, 0.0]
tick  30: max|u|   7.5486 m/s ...
          row0 depth = [0.967, 1.062, 0.798, 1.308, 0.671, 1.619, 0.575, 0.0]  eta spread(wet) 1.0436 m
[r3, M1 terrain] t=0 wet cells 26214, lake volume 57688 m^3, max depth 7.90 m, eta spread over wet 0.00e+00 m, speed 0
tick 3: n_sub 704  max speed 5.610 m/s  cells >0.5 m/s: 9069  mean speed(wet) 0.5081  eta spread(wet) 0.2625 m  water volume through faces this tick 523515.4 m^3  max|dsed| 3.88e-03 m
```

**Defense (re-ran, tried to acquit):** Guilty. (1) Reproduced: I re-ran r2, r3 and r1 (16/32) on CPU with the exact commands. The output matches the accusation digit-for-digit. Without a bank the pond stays at 0.0000 m/s. With a 2 m dry bank it runs at ±7.55 m/s, in a steady row-wise alternating pattern with an uneven depth sawtooth, and never decays. M1 terrain reaches 5.61 m/s with 9069 cells above 0.5 m/s after 3 ticks. (2) Causation proven: I monkeypatched k_momentum in memory only (no repo edit) so that a dry neighbour whose bed is above the cell's surface is used as a wall (eta_nb = eta_c). Everything else is repo code. With that change the 8x8 bank pond and the 256x256 M1 terrain stay exactly at rest: 0.00000 m/s, eta spread 0, dh about 1e-17, no sediment moved. So fluid.py:87-88 using elev+sed+hw of dry neighbours in the centred difference is the proximate cause. (3) Not intended: DECISIONS.md, RUNBOOK and the spec say nothing about wet/dry, shoreline or lake-at-rest handling. The spec asks for shallow-water equations and says erosion is 'slow'. A flat surface at rest is an exact SWE steady state (the standard well-balancedness test). It is not in the NOT-bugs list. (4) Reachable in normal runs: terrain.generate sets water=max(0,level-elev), so every world starts in this state. k_momentum and terrain.py are byte-identical between genesis-v0.1-historical and HEAD. The owner's real GPU M1 genesis run snapshots (read-only) show 5.585 m/s and 7840 cells above 0.5 m/s at tick 1, matching CPU. At tick 518400 (360 days) there are still 1941 cells above 0.5 m/s, and 7418 of 9047 adjacent fast pairs have opposite-sign u (checkerboard). All 679 audit entries in that run say 'audit ok'. Defense mitigation, which does not acquit: there is a second, independent mechanism. Once the dry-bank term is fixed, eta roundoff of about 2e-15 still grows exponentially (1.5e-15 to 3.2 m/s within one tick on 32x32). The patched 32x32 still reaches 13.8 m/s against 39.4 unpatched, so the prosecutor's 16/32 magnitudes are not wholly due to lines 87-88. On M1 terrain this second mechanism appears around tick 10 at about 1% of the scale: roughly 120 cells above 0.5 m/s against roughly 10,300 unpatched. The accused defect still alone breaks lake-at-rest at tick 1 and dominates the production world. Severity stays critical: the core water physics is silently wrong in every run, persistently. Net flux across cell faces is about 9x the lake volume per tick, and dissolved species advect with it. Erosion is about 100x the fixed-kernel control. Audits cannot detect it. Scripts: /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-lake-at-rest-phantom-currents/d1_read_real_snapshots.py, d2_causal_patch.py, d4_trace_substeps.py, d5_m1_long.py.

## 5. Species diffusion moves solute down the COUNT gradient, not the concentration gradient, so a uniform solution in still water un-mixes (up to 4x) toward the shallow cells

- **Verdict:** guilty — severity major → major; reachable in normal runs: True
- **Where:** `firmament/core/fluid.py:356-360 (diff_out: d = n - spec[s, ni, nj]; amt = stoch_round(d * diff_frac))`
- **Expected:** A solution of uniform concentration in still water is at diffusive equilibrium: no net transfer and no concentration change. DIFF_COEF is documented in m^2/s, the units of a Fickian coefficient that acts on concentration. Counts are defined as µmol in the cell's water column, so concentration = count / depth. *(source: Fick's first law (flux = -D grad c) and the second law: a uniform solution cannot spontaneously form a gradient. fluid.py:30 'DIFF_COEF = 0.01 / 60  # m^2/s'. DECISIONS 2026-09-25: 'One count = 1 µmol in the cell's 1 m^2 water column -> molar concentrations per cell'. chemistry and polymers compute conc = count_mol/(depth*1000).)*
- **Actual:** The kernel equalizes counts per cell, whatever the water volume. Setup: still pond with a flat surface and all cells wet (no advection, no evaporation or rain; max|u| and max|dh| stay 0). Deep half is 2 m, shallow half 0.5 m, both at 1 mM Pi. Over 3000 ticks (about 2 sim-days) the shallow half rises to 2.49 mM and the deep half falls to 0.63 mM, a 3.96x gradient that approaches the 4x depth ratio. Pi is conserved exactly. In normal runs init_wet gives equal counts per wet cell, and this diffusion holds that state. Shallow margins therefore stay depth-ratio more concentrated forever. That biases the thermodynamic gate (RT ln[conc]) and copy energetics.
- **Evidence:** `env/r7_diffusion_counts.py`

```
diff_frac per tick = 0.01
tick     0: [Pi] deep half 1.0000 mM, shallow half 1.0000 mM, ratio 1.000; max|u| 0.0e+00  max|dh| 0.0e+00  total Pi 80000000
tick   100: [Pi] deep half 0.9016 mM, shallow half 1.3935 mM, ratio 1.546; max|u| 0.0e+00  max|dh| 0.0e+00  total Pi 80000000
tick  1000: [Pi] deep half 0.6921 mM, shallow half 2.2316 mM, ratio 3.224; max|u| 0.0e+00  max|dh| 0.0e+00  total Pi 80000000
tick  3000: [Pi] deep half 0.6282 mM, shallow half 2.4873 mM, ratio 3.959; max|u| 0.0e+00  max|dh| 0.0e+00  total Pi 80000000
```

**Defense (re-ran, tried to acquit):** GUILTY. (1) The prosecutor's repro reproduces bit for bit on CPU. The state stays still (max|u|=0, max|dh|=0, Pi conserved), yet the shallow/deep concentration ratio climbs to 3.959, approaching the 4x depth ratio, which is the equal-counts equilibrium (0.625 mM / 2.5 mM). (2) The repro is sound, and I isolated the cause with a control test (d1_isolate.py). Swapping only the diffusion kernel at runtime, with no repo edit, turning diffusion off keeps the ratio at exactly 1.000. A Fick's-law kernel (flux = diff_frac*min(hA,hB)*(nA/hA - nB/hB)) on the same integer, availability-capped machinery also keeps exactly 1.000. The actual kernel reaches 3.224. The count-difference line fluid.py:356-360 (d = n - spec[s,ni,nj]; amt = stoch_round(d*diff_frac)) is therefore solely responsible. (3) Not intended or documented. DIFF_COEF is declared in m^2/s (fluid.py:29), the unit of a Fick's-law coefficient. DECISIONS 2026-09-25 and chemistry_v0_2.yaml:26 define one count as 1 umol in the cell's water column. chemistry.py:37 and polymers.py:289 compute concentration as count_mol/(depth*1000) for the thermodynamic gate and the copy step. No doc says transport equalizes counts. The not-a-bug list exempts only kinetic-only reactions, not transport, and the only diffusion test (test_phase2_water.py:137) checks underflow and conservation, not equilibrium. The best defense, that reaction rates use count-based mass action (NORM) and so cells might be equal-volume reactors, fails because the documented gate and copy energetics explicitly use depth. (4) Reachability: the dramatic 'uniform solution un-mixes 4x' scenario needs a hand-built uniform-concentration state. Natural runs start with equal counts per wet cell (init_wet), so concentration is already proportional to 1/depth (slope -1.000) at tick 0. In natural runs the kernel mainly stops that gradient from relaxing and pumps counts into newly wetted shallow cells; it does not create the gradient. This mitigates the headline but does not acquit. The law runs on every tick of every run: natural 32x32 wet-wet neighbor depth ratios have median 1.53, p90 4.00, max 6449. An unmodified full-stack A/B on the natural 32x32 world shows a shallow/deep Pi ratio of 17.28 (actual) vs 11.97 (Fick) after 1000 ticks, slope -0.714 vs -0.606. A 16x16 run over 4000 ticks keeps the ratio higher under the actual kernel at every checkpoint (9.00 vs 5.61 at tick 3000), though the slope metric was mixed at tick 4000, so short-horizon natural effects are moderate (about 0.5-1.2 kJ/mol of gate bias). Severity stays major. In any still, connected water body the kernel's fixed point is equal counts (shown directly), which permanently biases the central thermodynamic gate by RT ln(depth ratio): 3.46 kJ/mol at 4x, 8.2 kJ/mol at the 26.8x shallow/deep ratio at init. The P~P and copy dG0 values are only 19-21 kJ/mol, and SEED_PROPOSAL.md places the seed in the 'warm shallow rim'. The large long-run bias is an extrapolation from that fixed point rather than a direct measurement. There is no conservation or determinism failure.

## 6. Meteor event heats the surface layer (capacity C_SURF_DRY + CW_VOL*depth) but books only C_SURF_DRY*dT with the audit, so a meteor over water deposits several times its stated energy and trips an AuditError

- **Verdict:** guilty — severity major → major; reachable in normal runs: True
- **Where:** `firmament/operator/console.py:112-119 (dtk = e/(h*w)*exp(..)/6.5e5; t[1] += dtk; register_injection(energy=(dtk*6.5e5).sum())); detected by firmament/core/audit.py:112-119`
- **Expected:** Every operator injection registers exactly the energy it adds, so the books close and the audit passes. The meteor should deposit about its stated energy_J. *(source: audit.py docstring: 'Operator events that inject matter/energy/water register here so the books stay closed.' console.py docstring: 'Matter/energy/water injections register with the audit so the books stay closed.' audit.energy_stored uses c1 = C_SURF_DRY + CW_VOL*d for layer 1. Writeup: 'Meteor | location, energy -> crater + heat + dust'.)*
- **Actual:** This uses the exact meteor params from tests/test_phase8_console.py (energy_J=1e9, 32x32 test world, 421 wet cells). Stored energy jumps by 3.44e9 J, more than 3x the meteor's total energy_J. Only 6.01e8 J is registered, leaving 2.84e9 J unbooked. The next audit raises AuditError (rel 0.873) and halts the run. Crater and dust play no part: the same numbers appear with crater_m=0 and dust=0. test_phase8 misses this because its audit_every_ticks=500 never fires after the meteor before tick 400.
- **Evidence:** `env/r6_meteor_ledger.py`

```
ENERGY AUDIT VIOLATION
[test_phase8 meteor params] wet cells 421/1024; stored-energy jump 3.4426e+09 J, registered with audit 6.0138e+08 J, unbooked 2.8413e+09 J
[test_phase8 meteor params] AuditError: energy conservation violated at tick 10: rel 8.73e-01
[no crater, no dust] wet cells 421/1024; stored-energy jump 3.4426e+09 J, registered with audit 6.0138e+08 J, unbooked 2.8413e+09 J
[no crater, no dust] AuditError: energy conservation violated at tick 10: rel 8.73e-01
```

**Defense (re-ran, tried to acquit):** The prosecutor's repro reproduced exactly. In console.py:112-119 the meteor computes dtk using dry-rock capacity only (6.5e5 = C_SURF_DRY), applies t[1] += dtk to every cell, and books dtk*6.5e5. Every other part of the code treats layer 1 as one temperature with capacity C_SURF_DRY + CW_VOL*depth: audit.energy_stored (audit.py:78), radiation.py:44, thermal.py:20, fluid.py:150, polymers.py:408 and chemistry.py:117 ('real layer capacity'). So over water the meteor stores far more energy than it registers, 3.44e9 J stored against 6.01e8 J booked for energy_J=1e9. My controls isolate the cause. With no meteor, audits pass at the same cadence. With only the booking corrected to sum((C_SURF_DRY+CW_VOL*d)*dT), stored and booked are equal (3.4426e9) and audits pass. In a dry world, where the capacities coincide, audits pass with the code as shipped. The repro is not flawed: the stored-energy measurement is taken immediately around causal.apply_due, and crater and dust do not enter energy_stored. It is reachable without injecting any state. The repo's own tests/test_phase8_console.py EVENTS schedule, left unmodified and simply run past tick 400, raises AuditError at the first audit after the meteor (tick 500 at the test cadence, tick 1000 at the shipped default cadence). Removing only the meteor makes every audit pass. Meteors are a normal operator action through the GUI (/api/console/meteor) and the CLI (firmament event). Nothing in DECISIONS.md, RUNBOOK or the spec documents this. The docstrings promise that injections 'register with the audit so the books stay closed', and the flood branch books the 'exact stored delta'. None of the hard-rule exclusions applies. The only mitigation is that the tolerance scales with cumulative throughput, so a small meteor very late in a long run could fall under 1e-3. That lowers the chance of a crash in that edge case but does not remove the wrong energy deposit or the unbooked energy. Severity is major: a documented natural event deterministically halts runs, and over water it deposits several times its stated energy.

## 7. A refused or malformed causal command is written to events.jsonl before it is applied, so the run crashes and can never be resumed

- **Verdict:** guilty — severity critical → major; reachable in normal runs: True
- **Where:** `firmament/operator/causal.py:48-50 (commit_now appends the record, then _apply raises and causal_n is not incremented); raise sites include firmament/operator/console.py _add_land (NotImplementedError) and float(params.get(...)) in apply_natural`
- **Expected:** If a command is refused (add_land) or has bad parameters, the run's causal history should stay as it was and the run should stay resumable. Resume should re-apply only history that was actually applied. *(source: RUNBOOK 'Moving to world_default.yaml': "`add_land` is a documented refusal". causal.py docstring: resume/replay/fork "re-apply exactly the logged records". RUNBOOK Resume: "replay every causal command logged after it. The result is indistinguishable from never having stopped". DECISIONS 2026-09-25: "events.jsonl holds only causal commands ... committed at tick boundaries".)*
- **Actual:** commit_now writes the record (n=causal_n) to events.jsonl and flushes it, then _apply raises, so the log now holds a record the state never applied. The live sim dies. This also happens from the GUI god console: app.js sends +input.value, so a typo like "10mm" arrives as null. After that, every `resume`, with or without --auto-restart, queues the poisoned record, replays it at its tick and crashes with the same exception. A valid `firmament event` on the damaged run is also silently dropped (see bug 3). The only ways out are editing events.jsonl by hand or forking.
- **Evidence:** `history/repro_add_land_bricks_run.py`

```
2) event --type add_land   rc=1  -> NotImplementedError: add_land requires a full state rebuild; run `firmament fork` into a wider world config instead (v0 limitation, logged as an open question)
   events.jsonl after refusal: [(0, 10, 'add_land', {'cols': 4})]
3.1) resume --ticks 5     rc=1  -> NotImplementedError: add_land requires a full state rebuild; ...
3.2) resume --ticks 5     rc=1  -> NotImplementedError: add_land requires a full state rebuild; ...
4) event rain intensity_mm='10mm'  rc=1 -> ValueError: could not convert string to float: '10mm'
   events.jsonl: [(0, 'rain', {'intensity_mm': '10mm'})]
5) resume --ticks 5        rc=1  -> ValueError: could not convert string to float: '10mm'
6) GUI console rain (typo) -> {'queued': 'rain', 'applies_at': 'next tick boundary'}
   live sim exited rc=1 -> TypeError: float() argument must be a string or a real number, not 'NoneType'
   events.jsonl: [(0, 14, 'rain', {'cx': 8, 'cy': 8, 'radius': 6, 'intensity_mm': None})]
7) resume --auto-restart   rc=1  -> TypeError: float() argument must be a string or a real number, not 'NoneType'
8) event rain (valid)      rc=0  stdout: 'rain applied at tick 11'
   events.jsonl: [(0, 14, 'rain', {'cx': 8, 'cy': 8, 'radius': 6, 'intensity_mm': None})]
9) resume --ticks 5        rc=1  -> TypeError: float() argument must be a string or a real number, not 'NoneType'
```

**Defense (re-ran, tried to acquit):** Reproduced. I re-ran the prosecutor's exact command and also wrote my own repro: /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-history-refused-command-poisons-causal-log/defense_repro.py. One step came out differently. Step 7 of the prosecutor's script returned rc=0 on my run instead of rc=1. That is a timing effect: the live sim got to tick 17 before the GUI command landed, and `resume --ticks 5` from the tick-10 snapshot stopped at tick 15, before it reached the bad record. Step 9 then crashed as claimed. My own repro uses a longer tick bound and crashes every time.

The mechanism, from firmament/operator/causal.py:48-50: commit_now calls sched.events.append, and EventLog.append writes and flushes the line. It then calls _apply. If _apply raises, `state.causal_n += 1` never runs.

1. The code breaks its own rule. Part P1 of my repro ends with events.count=1 and causal_n=0. The next commit_now then refuses with "causal log has 1 records but state has applied 0 — refusing to fork history silently". The inconsistency the code is written to prevent is created by commit_now itself.

2. The bad record is the only cause. I copied the run directory and removed only the record that was never applied. Resume then ran to tick 70 with rc=0. On the unedited copy, every resume crashes at the same tick (14) with the same TypeError, recorded in crash.json.

3. It is reachable in normal use. gui/app.js builds the request with `body[k] = +i.value`. Node confirms that a typo like "10mm" or "0,5" becomes JSON null, and the server's console_cmd accepts it without checking. The GUI is "the live path" (RUNBOOK). A single typo in any god-console field kills a sim meant to run 24/7. After that, the RUNBOOK crash checklist step "resume --auto-restart continues from the snapshot" fails every time. The CLI path is just as direct: `firmament event --type add_land` always raises, since it is a documented refusal. So does `add_species`, even with a valid superset.

Defense arguments I considered, and why each fails:
- (a) Logging before applying is required by the spec (Build Guide: "writes an event record before applying ... so replay is exact"; console.py says the same). But that rule exists so replay is exact. Logging a command that never took effect makes replay diverge from what the live state actually applied. place_seed shows the intended pattern: "Validated first, then committed to the causal log". apply_natural does no such validation before logging.
- (b) The RUNBOOK says deterministic crashes "stop for a human". That covers world-model faults like AuditError and capacity limits, not an operator typo or a refusal. In this codebase "refused" means declined with no side effects: see the lease, the add_species tests, and commit_now's own refusal. Here, though, a refused command becomes permanent history in "the run's definition".
- (c) Recovery exists. Fork works, and it is what the add_land message suggests. But fork gives a new run_id, and fork_run does not copy lineage.sqlite. Otherwise the only fix is hand-editing events.jsonl, which no document describes.

Nothing in DECISIONS, RUNBOOK, PROGRESS or the docstrings describes or accepts this behaviour. No test covers a refusal going through the causal path; the add_species tests call _add_species directly.

On severity, I argue it down from critical to major. No snapshot data is corrupted. The fix is a one-line deletion from events.jsonl or a fork. The GUI confirm box does show the null before you confirm. Even so, the bug kills the live run, blocks every resume and auto-restart, loses progress since the last snapshot, and is easy to hit from an ordinary GUI typo.

## 8. Stopping with Ctrl-C (the documented Stop) writes no shutdown snapshot and permanently loses buffered metrics and lineage rows up to the last snapshot

- **Verdict:** guilty — severity major → major; reachable in normal runs: True
- **Where:** `firmament/cli.py:92 (`except Exception` does not catch KeyboardInterrupt and there is no finally, so neither the shutdown snapshot at line 90 nor _flush runs); firmament/io/snapshot.py:31 (save() never flushes sched.metrics or sched.lineage, so rows at or before the snapshot tick exist only in RAM; sampler.py:76 flushes every 100 samples, lineage.py:34 every 10000 rows)`
- **Expected:** Stopping the run should write a shutdown snapshot. After resume, metrics and lineage should match an uninterrupted run. Even after an unclean kill, the only loss should be progress after the last snapshot, which resume regenerates. *(source: Writeup storage table: Snapshot "Every S sim-days, on demand, and on shutdown". RUNBOOK 'Stop': "Ctrl-C (or kill). A shutdown snapshot is written on clean exit; an unclean kill loses at most snapshot_every_sim_days of progress". RUNBOOK Resume: "indistinguishable from never having stopped". RUNBOOK: metrics/*.parquet and lineage.sqlite are "never rotated away". Build guide Phase 10: "a run has survived a forced crash and resumed correctly". stop_flag is never set anywhere (grep), so an indefinite run has no other way to stop.)*
- **Actual:** Ctrl-C at about tick 317 exits with code -2. No snapshot is written for the stop tick (the latest stays tick 300), and 0 metrics rows and 0 lineage rows reach disk. Resume starts after tick 300 and truncates newer rows, so rows at or before 300 are never regenerated. Against an uninterrupted seeded run to tick 600: metric rows for ticks 5..300 are missing (60 of 120), 19 lineage copy records are missing, and 47 surviving records point to parents whose copy record is gone, which breaks the lineage tree. The final physical state is identical, so only the history is lost.
- **Evidence:** `history/repro_ctrl_c_loses_history.py`

```
Ctrl-C: exit code -2 | last stderr line: KeyboardInterrupt
sim had reached tick ~317 when stopped
snapshots after Ctrl-C       : ['tick_000000000001.zarr', 'tick_000000000300.zarr']
metrics rows on disk after Ctrl-C: 0
lineage rows on disk after Ctrl-C: 0

final tick snapshot A/B     : tick_000000000600.zarr tick_000000000600.zarr
metrics rows  uninterrupted : 120 ticks [5, 10, 15] ... [595, 600]
metrics rows  Ctrl-C+resume : 60 ticks [305, 310, 315] ... [595, 600]
metrics ticks MISSING after resume: 60 from [5] to [300]
lineage rows  uninterrupted : 184
lineage rows  Ctrl-C+resume : 165
lineage copy records MISSING after resume: 19 (ticks 31..156)
records in B whose parent's copy record is gone: 47 [(19, 10), (26, 2), (27, 7)]
final state arrays identical: True
```

**Defense (re-ran, tried to acquit):** The prosecutor's repro reproduced verbatim when I re-ran it on CPU. I then tried every defense.

(1) Realism of Ctrl-C. My own repro sends SIGINT to the whole process group of `uv run python -m firmament.cli resume` (what a terminal Ctrl-C does). It exits with code 130 and KeyboardInterrupt, with the same loss.

(2) Rigged flush cadence. My repro writes metrics every 200 ticks and snapshots every 300, so the write interval is shorter than and out of step with the snapshot interval, as in world_small.yaml (writes every 10,000 ticks, snapshots every 43,200). Rows 2..200 survive. Rows 202..300 are lost permanently, and every lost row is at or before the snapshot tick.

(3) "Ctrl-C is not a clean exit." Even granting that, SIGTERM (`kill` on the pid in meta.json), SIGKILL (the RUNBOOK's "unclean kill") and an exception crash (the Phase 10 "forced crash") all lose exactly the same 50 metric rows and 19 lineage copy rows, leaving 47 orphaned records. That breaks the RUNBOOK's promises "an unclean kill loses at most snapshot_every_sim_days of progress (resume from the last snapshot)" and resume "is indistinguishable from never having stopped". It also breaks "never rotated away" for metrics/*.parquet and lineage.sqlite. No rows appeared that were not in the uninterrupted run, so this is loss, not nondeterminism.

(4) Intended or documented. Nothing documents that rows written before a snapshot may be lost. The writeup says only that metrics and lineage are "Buffered", and lists snapshots as taken "on shutdown". The design works hard elsewhere to keep history consistent across resume (analysis_state and pending_copies are saved in the snapshot, and truncate_after trims history), so the unwritten buffers look like an oversight.

Root cause:
- cli.py:88-98 writes the shutdown snapshot (line 90) and flushes (line 91) only after a normal loop exit. `except Exception` does not catch KeyboardInterrupt, there is no finally, and there is no SIGTERM handler.
- stop_flag is never set anywhere (only scheduler.py:30 and :50), and the API has no stop endpoint, so an indefinite run can never reach line 90.
- snapshot.py save() never writes out sched.metrics or sched.lineage. Lineage is only written at 10,000 buffered rows.
- On resume, truncate_after trims history to the snapshot tick, so rows at or before that tick that were only in memory are never recreated.

Severity is major rather than critical: the final physical state is identical, but the research record (metrics, lineage phylogeny) is lost on every documented way of stopping a run.

Evidence files are in /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-history-ctrl-c-stop-loses-history/ (stop_paths.py, compare.py, result_*.json).

## 9. `firmament event` prints "<type> applied at tick T" and exits 0, but never logs or applies the event while logged history is waiting to be replayed

- **Verdict:** guilty — severity major → major; reachable in normal runs: True
- **Where:** `firmament/cli.py:226-230 (cmd_event queues the command, runs one tick_once, saves and exits; causal.apply_due at firmament/operator/causal.py:75-78 holds live mailbox commands while replay_queue is non-empty, and the held command is discarded when the process exits)`
- **Expected:** The event should be committed to events.jsonl and applied, or the command should refuse loudly, as `firmament seed` does through commit_now's RuntimeError "cannot commit a new command while logged history is replaying". *(source: cmd_event docstring: "Apply a natural event to a resumable run". Its own output "applied at tick". causal.py docstring: commands are COMMITTED at a tick boundary and written to events.jsonl. RUNBOOK: events.jsonl = "the run's definition".)*
- **Actual:** Setup uses only documented operations: a rain from the GUI console, then a Ctrl-C stop (bug 2), which leaves that rain logged after the latest snapshot. `firmament event --type drought` then prints 'drought applied at tick 6' with rc=0. events.jsonl still holds only the rain, the new tick-6 snapshot has causal_n=0, and total vapor rose instead of falling by 90%. After resume the drought never happens. The same silent drop hits any run damaged by bug 1 (step 8 of that repro).
- **Evidence:** `history/repro_event_silently_dropped.py`

```
1) run --ticks 5 -> snapshots ['tick_000000000005.zarr']
2) GUI console: {'queued': 'rain', 'applies_at': 'next tick boundary'}
   Ctrl-C; events.jsonl: [(0, 6, 'rain')]
   snapshots: ['tick_000000000005.zarr']
3) event --type drought  rc=0  stdout: 'drought applied at tick 6'
   events.jsonl now: [(0, 6, 'rain')]
   latest snapshot tick_000000000006.zarr causal_n=0
4) resume --ticks 200 -> final events.jsonl: [(0, 6, 'rain')]
   drought in causal history: False
   total vapor tick5=2.2867 tick6=2.7760 (a drought of 0.9 would leave ~0.2287)
```

**Defense (re-ran, tried to acquit):** I re-ran the prosecutor's exact command on CPU and it reproduced. The only difference was that the GUI rain was committed at tick 9 instead of 6, which is scheduling jitter. I then tried to break the case and could not.

1) Is the evidence misread? No. A clean-path control using the same config and the same `event --type drought --params {"strength":0.9}` logs the drought (events.jsonl [(0,5,'drought')]). Its tick-6 snapshot has causal_n=1, and total vapor falls from 2.2867 to 0.7183. In the accused path the tick-6 vapor is 2.7760, the same as with no drought at all, and causal_n stays 0. An in-process spy on cli._flush (defense_repro.py part C) catches the moment `cmd_event` is about to exit. At that moment sched.mailbox still holds ('drought', {'strength': 0.9}) and replay_queue holds [(0, 9, 'rain')]. The process then prints "drought applied at tick 6" and exits 0, and the mailbox is gone. Mechanism: causal.apply_due (causal.py:72-78) returns early while q[0]['tick'] > state.tick. `cmd_event` (cli.py:220-230) runs exactly one tick_once, then saves, prints and exits.

2) Intended or documented? No. The "held until replay has caught up" comment in apply_due covers the continuously running loop, where the held command does get committed later. Nothing says the offline `event` command may drop a command. The `cmd_event` docstring says "Apply a natural event to a resumable run", and the RUNBOOK lists it as the CLI path for natural events. commit_now already refuses loudly with RuntimeError("cannot commit a new command while logged history is replaying"), which shows the intended policy is a loud refusal, not a silent drop. The printed "applied at tick 6" with rc=0 is simply false. It is also not on the list of excluded non-bugs (thermo gating, cell_meters, dt, LAN, M1 baseline, ΔG, style, perf, future work). The existing test (tests/test_phase8_console.py::test_cli_event_command) covers only the clean path.

3) Reachable? Yes, without relying on the prosecutor's Ctrl-C "bug 2". The RUNBOOK's Stop section says "Ctrl-C (or kill) ... an unclean kill loses at most snapshot_every_sim_days of progress (resume from the last snapshot)". Resume is documented to "replay every causal command logged after it", so logged history after the last snapshot is an expected state. I reached it with a plain SIGTERM (part B) and with SIGKILL (part C), each after a live GUI rain. Every shipped config uses snapshot_every_sim_days 25-30 with dt 60, so the window between a live event and the next snapshot is about 36,000-43,200 ticks. The trigger is any live event followed by a kill, crash or power loss before the next snapshot, then an offline `firmament event`.

Mitigating points I considered: existing history is not corrupted, the extra tick-6 snapshot is consistent, the rain still replays at tick 9 on resume, and there is a workaround (resume first so replay catches up). These do not clear the code. The operator's explicit, state-changing command disappears while the tool reports success with exit code 0, so a scripted or cron protocol would record an intervention that never happened. Proven beyond reasonable doubt; I keep the severity at major because the failure is silent and the success message is false.

Evidence files: /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-history-offline-event-silently-dropped/defense_repro.py; code at /home/nick/Life/firmament/firmament/cli.py:220-230 and /home/nick/Life/firmament/firmament/operator/causal.py:67-81.

## 10. A natural event that fails to apply (add_land, which the RUNBOOK calls a refusal, or a GUI parameter typo) is accepted, written to events.jsonl, and then crashes the sim. Every later resume crashes at the same tick.

- **Verdict:** guilty — severity critical → major; reachable in normal runs: True
- **Where:** `firmament/operator/causal.py:48-50 (events.append before _apply); no validation at firmament/server/api.py:251 and firmament/operator/console.py:21-27; the raise happens at console.py:47 (float(None)) and console.py:129-131 (add_land)`
- **Expected:** An event the system cannot apply should be refused when it is requested, the way unknown events get a 400. RUNBOOK says add_land is 'a documented refusal'. Resume 'replay[s] every causal command logged after [the snapshot]. The result is indistinguishable from never having stopped.' A request that is refused must never enter the causal log. *(source: docs/RUNBOOK.md (Moving to 1024²: 'add_land is a documented refusal'; Resume: 'indistinguishable from never having stopped'); causal.py docstring (events are COMMITTED and applied); api.py already returns 400 for unknown events)*
- **Actual:** The API returns 200 {'queued': ...} and `firmament event` accepts the request. commit_now appends the record to events.jsonl, and then apply_natural raises inside the sim thread (NotImplementedError for add_land; TypeError for a GUI typo, because app.js serializes +"10mm" as NaN -> JSON null). The sim crashes, and events.count is now causal_n+1, so every later valid event is refused with RuntimeError. `firmament resume` replays the poisoned record and crashes again at the same tick every time. The run cannot be continued without hand-editing the causal log.
- **Evidence:** `observe/repro_poisoned_event.py`

```
[A] API reply 2: {'queued': 'rain', 'applies_at': 'next tick boundary'}
[A] sim thread crashed at tick 3: TypeError: float() argument must be a string or a real number, not 'NoneType'
[A] events.jsonl records: [('rain', 0, {'cx': 8, 'cy': 8, 'radius': 4, 'intensity_mm': None})]
[A] state.causal_n = 0   events.count = 1
[A] valid follow-up event -> RuntimeError: causal log has 1 records but state has applied 0 — refusing to fork history silently
[B] firmament run --ticks 5          -> rc=0
[B] firmament event --type add_land  -> rc=1 ['NotImplementedError: add_land requires a full state rebuild; ...']
[B] events.jsonl now: [('add_land', 0, 5)]
[B] firmament resume (attempt 1) -> rc=1 ['NotImplementedError: add_land requires a full state rebuild; ...']
[B] firmament resume (attempt 2) -> rc=1 ['NotImplementedError: add_land requires a full state rebuild; ...']
[B] crash.json: tick=5 exception=NotImplementedError('add_land requires a full state rebuild; ...')
```

**Defense (re-ran, tried to acquit):** GUILTY, with severity reduced from critical to major.

The repro reproduced verbatim. I also went beyond the prosecutor's direct endpoint calls and tested a real live `firmament run` process. It served uvicorn on CPU and had snapshots at ticks 40 and 60. I sent real HTTP POSTs through urllib, using the exact body gui/app.js produces. The inputs in gui/app.js:283 have no type attribute, so they are text inputs, and line 291 does `body[k] = +i.value`. I ran it in node: JSON.stringify({intensity_mm: +"10mm"}) gives {"intensity_mm":null}.
- Rain with a typo: the API returned 200 queued, the live process exited with code 1, and crash.json was written at tick 66. events.jsonl kept the record, and both resumes failed with rc=1 at the same tick.
- add_land sent over HTTP: identical behavior at tick 62.

Mechanism: causal.commit_now (causal.py:48-50) appends and flushes the record through EventLog.append (events.py:30-31) before _apply runs. apply_natural (console.py:47-131) has no validation, and console.request/api.py:250-252 only check the event name.

Intent: log-before-apply is specified (Build Guide:273, console.py docstring). Logging a record that can never be applied is not. EventLog's docstring says the log "together with the frozen config and seed ... fully determines the world". RUNBOOK:78 calls add_land "a documented refusal". The RUNBOOK crash checklist step 2 says resume continues from the snapshot, and only AuditError is flagged as not to resume blindly. The Build Guide (275, 280) also requires add-species to "refuse" a rate change. I tested exactly that refusal: it was accepted at request time, logged, raised ValueError at apply, and raised the same ValueError on replay. place_seed (console.py:158-171) shows the intended pattern, "Validated first, then committed", and natural events skip it.

Defense mitigations (they lower the severity):
(1) `firmament fork` is an escape hatch. fork_run (snapshot.py:121-132) keeps only records with n < snapshot causal_n. The forked log was empty and resuming the fork returned rc=0. So "cannot be continued without hand-editing" is overstated. What is lost is the original run_id, progress since the last snapshot (up to snapshot_every_sim_days, 30 by default), and lineage/metrics continuity, since fork does not copy them.
(2) The crash is loud (traceback plus crash.json), not a silent corruption.
(3) The GUI confirm dialog shows the null, and add_land/add_species are not in the GUI dropdown; they are reachable through the RUNBOOK's CLI `firmament event` or any HTTP client.

Even so, an ordinary operator action (a GUI typo, or add_land as the RUNBOOK invites) kills a live 24/7 sim, permanently breaks resume for that run_id, and leaves the causal log holding an event the world never applied. Evidence: /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-observe-poisoned-causal-log/live_http_e2e.py (.out) and add_species_refusal.py.

## 11. On wet cells the meteor adds several times more energy than it registers with the audit, so the energy audit fails

- **Verdict:** guilty — severity major → major; reachable in normal runs: True
- **Where:** `firmament/operator/console.py:112-119`
- **Expected:** The console docstring says 'Matter/energy/water injections register with the audit so the books stay closed.' The energy registered should equal the change in audit.energy_stored. For layer 1 that change is (C_SURF_DRY + CW_VOL*water_depth)*dT, not 6.5e5*dT. Flood already registers the exact stored change. *(source: firmament/operator/console.py module docstring; firmament/core/audit.py energy_stored (c1 = C_SURF_DRY + CW_VOL*d); energy conservation)*
- **Actual:** The meteor raises temp[1] by dtk and registers only dtk*6.5e5 (dry rock). Where there is water, the stored energy rises by dtk*(6.5e5 + 4.186e6*d). On a 32x32 world with the default water fraction of 0.4, the stored energy rose 6.5 times as much as the registered amount (3.7e9 J unregistered), and the next audit raised AuditError. The same meteor on a dry world balances exactly (control). The GUI-default meteor on the RUNBOOK shakeout world (world_small.yaml, 256²) leaves 4.6e9 J unregistered.
- **Evidence:** `observe/repro_meteor_wet.py`

```
--- meteor energy_J=1e+09 world water_fraction=0.0 (wet cells 0/1024, mean depth on wet 0.000 m)
stored-energy increase (audit.energy_stored): 6.754663e+08 J
energy registered with audit               : 6.754663e+08 J
unregistered energy                        : -8.344650e-06 J (ratio actual/registered = 1.0000)
AuditError: None
--- meteor energy_J=1e+09 world water_fraction=0.4 (wet cells 421/1024, mean depth on wet 2.173 m)
stored-energy increase (audit.energy_stored): 4.391498e+09 J
energy registered with audit               : 6.754663e+08 J
unregistered energy                        : 3.716032e+09 J (ratio actual/registered = 6.5014)
AuditError: energy conservation violated at tick 10: rel 8.84e-01
--- world_small.yaml 256x256, GUI-default meteor
stored-energy increase 2.379194e+10 J, registered 1.917476e+10 J, unregistered 4.617178e+09 J (x1.241)
```

**Defense (re-ran, tried to acquit):** GUILTY beyond a reasonable doubt.

(1) Reproduced: my re-run of the prosecutor's exact command gives the same output: dry world balances (-8.3e-06 J); wet world 3.716032e+09 J unregistered (x6.50) and AuditError at tick 10; the 256² GUI default leaves 4.617178e+09 J unregistered.

(2) The repro's weak spot does not hold up: it sets audit_every_ticks=10. My own probe (defense_meteor.py) keeps the defaults: tiny_cfg as-is (audit_every_ticks=500), water 0.4, and the meteor parameters from the repo's own tests/test_phase8_console.py applied at tick 260 through console.request / causal.apply_due, the same path as the GUI (api.py:251) and the scheduler (scheduler.py:38). Result: 'AuditError: energy conservation violated at tick 500: rel 2.47e-01'. The existing phase-8 test only escapes because it stops at tick 400, before the first audit after the meteor.

(3) Mechanism proven exactly: the unregistered amount equals sum(CW_VOL * water_depth * dT1) (2.835566e+09 in both columns). console.py:112-113 raises temp[1] by dtk, which by audit.py:78 stores (C_SURF_DRY + CW_VOL*d)*dtk. But console.py:119 registers only dtk*6.5e5.

(4) Causation control: registering only the missing water term makes the same run pass the audit (AuditError: None). The dry control passes. So the missing term is the sole cause.

(5) Not intended: DECISIONS, RUNBOOK and the spec never mention approximate meteor accounting. The console.py and audit.py docstrings both promise that injections register so the books stay closed. The flood event, right next to it, registers the exact stored change. The accusation is not among the listed non-bugs: the P~P gating exemption concerns reaction thermodynamics, not audit bookkeeping.

(6) Reachable: meteor is a GUI and CLI natural event, and initial_water_fraction is 0.4 in the configs.

Mitigation, which reduces severity but not guilt: the audit tolerance is 1e-3 of cumulative e_in, which grows every tick (radiation.py:49). I estimated from absorbed-energy rates, without a run, that a GUI-default meteor on world_small would stop triggering an error after about 5,400 ticks. After that the books are silently open instead of failing loudly. An early meteor, a larger energy_J or a small world still halts the sim. Per the RUNBOOK, an AuditError halts the run for a human and reproduces on resume; I did not test a resume after a meteor myself. The title's 'several times' holds for the tested spot. At the GUI default on the shakeout world the excess is 1.24x. Severity stays major because an ordinary, operator-facing natural event breaks the non-negotiable conservation audit and can stop a live run.

## 12. Lineage copy events wait in RAM until 10,000 have accumulated: the API lineage path and the depth-100 milestone read an empty tree, and an unclean kill loses events from before the snapshot for good

- **Verdict:** guilty — severity major → major; reachable in normal runs: True
- **Where:** `firmament/io/lineage.py:31-35 (flushes only at >=10000 buffered); readers at lineage.py:43-64 query sqlite only; used by firmament/server/api.py:213 and firmament/instruments/milestones.py:112; snapshots (io/snapshot.py save) and the crash path (cli.py _guarded_loop except) never flush it`
- **Expected:** (a) /api/inspect/polymer lineage_path and Milestones' lin.depth_of should reflect the copy events that have already happened. (b) The RUNBOOK says 'an unclean kill loses at most snapshot_every_sim_days of progress (resume from the last snapshot)' and resume is 'indistinguishable from never having stopped'. It also says lineage.sqlite is 'never rotated away'. *(source: docs/RUNBOOK.md (Stop; Resume / long-run mode; Logs and rotation); api.py inspect_polymer contract (lineage_path to seed); milestones.py candidate_mutant_depth100 uses lineage.depth_of)*
- **Actual:** On a live seeded run at tick 200 there were 24 copy events buffered and 0 rows in lineage.sqlite. The API returned lineage_path=[32] and depth_of=0 for polymer #32. After a manual flush the correct answer is [32, 5, 4, 2, 1], depth 4. After a SIGKILL at tick 230 (tick-200 snapshot on disk), lineage.sqlite had 0 rows. Resuming from the snapshot regenerated only the 13 events after tick 200. The 24 copy events at tick<=200 that the uninterrupted reference run records are gone permanently.
- **Evidence:** `observe/repro_lineage_buffer.py`

```
[1] tick 200: completed polymers 25, copy events buffered in RAM 24, rows in lineage.sqlite 0
[1] /api/inspect/polymer?id=32 (parent #5): lineage_path=[32]  depth_of=0
[1] same query after a manual lineage.flush(): lineage_path=[32, 5, 4, 2, 1]  depth_of=4
[2] killed child: returncode=-9 (-9 = SIGKILL), stdout={"buffered_in_ram": 37}
[2] reference (never stopped) rows: total 37, tick<=200: 24
[2] after unclean kill rows      : total 0
[2] after resume from tick-200 snapshot to 230: total 13, tick<=200: 0  (tick>200: 13 vs ref 13)
```

**Defense (re-ran, tried to acquit):** I re-ran the prosecutor's command on CPU and got the same output, line for line.

Defense attempts, and why each failed:

(1) "Buffering is in the spec." It is: the Engineering Writeup storage table lists Lineage as "Buffered". But nothing in the spec, DECISIONS.md (including the 2026-09-25 sections), the RUNBOOK or the docstrings says lineage events may be invisible to readers or lost. The code states the opposite intent. polymers.py:762 raises "refusing to lose lineage evidence" rather than drop a copy event. The RUNBOOK says lineage.sqlite is "never rotated away". It also says an unclean kill "loses at most snapshot_every_sim_days of progress", and that resume is "indistinguishable from never having stopped". Buffered writes do not allow for data older than the last good snapshot being lost permanently.

(2) "SIGKILL is an artificial trigger." My repros avoid it and inject no state.
- defense_crash_path.py lets the run end on a natural RuntimeError ("polymer capacity exhausted", the same exception that ended historical genesis run 20260911-200738), going through the real cli._guarded_loop. lineage.sqlite then held 0 rows, although 24 copy events happened at or before the last good snapshot (tick 200). Resume starts from that snapshot with an empty buffer, so those 24 events can never be recovered. Code reading explains why: the except branch in _guarded_loop writes crash.json and re-raises without calling _flush. SnapshotManager.save does not flush lineage either. LineageDB flushes only when 10,000 events are buffered, or on close(), which the CLI never calls. prune() is never called.
- defense_sigint.py sends a real SIGINT, which is what Ctrl-C sends and is the RUNBOOK's documented way to stop. The process died with KeyboardInterrupt, no shutdown snapshot was written, and lineage.sqlite held 0 of the 37 copy events (24 of them from before the snapshot). `except Exception` does not catch KeyboardInterrupt, so the snapshot and _flush lines are skipped.
- By reading the code: `--auto-restart` in cmd_resume calls os.execv, which replaces the process without flushing, so the long-run mode has the same loss.

(3) "It is not seen in practice." The historical runs are real evidence. Run 20260911-012501 crashed at tick 67000 (last good snapshot 36000), yet its lineage.sqlite has exactly 20000 rows with max tick 36237. About 30,000 ticks of copy events that the process had generated were never written. Run 011449 has exactly 10001 rows. These counts are flush-threshold artifacts.

(4) Reader staleness, part (a). The API's lineage_path and Milestones' depth_of read sqlite only and ignore the in-RAM buffer. So in any run with fewer than 10,000 copies, the inspector returns [self] for every polymer. After that, every polymer born since the last flush does the same. The spec requires the inspector to show the "lineage path back to the seed", and M1's acceptance criterion is "lineage DB shows copies of copies". Of the two parts, (a) alone is the easier one to defend as "buffered latency". Part (b), permanent loss of data from before the snapshot, is unambiguous.

Critique of the prosecutor's repro: it uses SIGKILL and a hand-rolled resume rather than cmd_resume. The real _load_from_snapshot does nothing to lineage beyond reopening it, so the outcome is the same, and my natural-path repros confirm the claim without SIGKILL.

Severity is major. It permanently loses the scientific record (lineage evidence that feeds the M1 acceptance criterion and a milestone detector) on the documented Ctrl-C stop and on any crash. It contradicts explicit RUNBOOK guarantees. It is not critical because physics and sim state are unaffected.

Repro scripts:
/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-observe-lineage-buffer-invisible-and-lost/defense_crash_path.py
/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-observe-lineage-buffer-invisible-and-lost/defense_sigint.py

## Leads (NOT convicted — no defense trial or no repro)

- (life) **Motor speed and cost depend on dt** — From reading the code only; no repro run, and every shipped config uses dt=60. The motor stages p_motor in every 60 s sub-step and pays cost_pp each time, but _transport consumes it once per tick. At dt=120 a motor polymer would pay 2x P~P per move and move half as far per sim-second, which contradicts DECISIONS 'dt is a numerical choice, not a law'. Diffusion probability also saturates once 4*0.01*dt/60 >= 1.
- (life) **Emit-forced reactions deposit no reaction heat; copy-step heat is fixed regardless of copy_energy_per_monomer** — The emit branch applies the strecker_amino stoichiometry (dh -50 kJ/mol, which chemistry deposits as 0.05 J per event) but adds no heat. The copy step always adds 0.02 J whatever e_copy is, while each P~P hydrolysis in effects adds 0.033 J. The energy audit cannot catch this because e_chem only records what is deposited, and 'ΔG values are textbook approximations' is excluded, so I did not charge it.
- (life) **ecology.bound_pairs miscounts because of orphaned partner pointers** — A downstream consequence of bug 1: ecology.sample counts (p_partner>=0).sum()//2, which includes the asymmetric links shown in repro_binder_orphan.py. I did not run the instrument separately.
- (life) **Console flood/rain/drought (and apply_seed) replace state arrays with new wp.array objects while the fluid CUDA graph holds fixed pointers** — Outside the life area and GPU-only. fluid.py itself warns that captured arrays 'must never trade places', yet console.apply_natural rebinds s.water_depth and s.vapor. It cannot be tested because the GPU is off-limits.
- (env) **GPU only: the CUDA-graph fluid substeps keep stale array pointers after flood, earthquake, volcano or meteor replace s.water_depth or s.elevation (console.py:68/78/86/110, fluid.py:518-531). The water then freezes silently and the audit does not notice.** — No GPU may be used. Proven only on a CPU emulation of graph semantics (r8_graph_stale.py): wp.capture_* is replaced by a recorder that replays launches with the array objects captured at record time. That emulation is bit-identical to the plain path before any event, which validates it. After a 5 cm flood the plain path spreads the mound (spread 0.054 -> 0.009 m). The graph path reuses the n_sub=256 graph recorded before the flood, so s.water_depth changes by only about 4e-6 m per tick (evaporation) and the mound stays a fixed 0.050000 m through tick 45. The audits pass. Replay tests cannot catch this because both runs capture identically. It needs confirmation on real CUDA.
- (env) **Dust settling is a per-tick law, so dt changes physics (radiation.py:31: dust *= 0.99999 every tick). The comment says '~sim-week', but at dt=60 the e-fold is 69 days.** — Actually proven (r5_dust_dt.py): over the same 2 sim-days, dust drops 0.014194 at dt=60 but only 0.007149 at dt=120 (ratio 1.985). This contradicts DECISIONS 'dt is a numerical choice'. Left out of the bug list only because of the 3-bug cap and its minor severity.
- (env) **Energy-audit tolerance is 1e-3 of CUMULATIVE lifetime e_in/e_out (audit.py:111-113), so the absolute leak it tolerates grows without bound over a run** — This follows the documented '0.1% of throughput' rule, and I have no repro showing a real leak that it misses. It is only a sensitivity concern for long runs.
- (env) **CFL substeps are capped at 800 regardless of dt. At dt=120, which is allowed, the cap binds at about 2.7 m depth, so under-resolved flow becomes dt-dependent.** — The code logs the clamp loudly by design, and I did not show a conservation or determinism failure caused by it separately from the phantom-current bug.
- (history) **PROVEN but left out only because of the 3-bug cap: `replay --to-tick T --verify` passes without replaying anything when the loaded snapshot is at or after T** — This one is proven, not a suspicion. Repro: /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history/repro_replay_verify_vacuous.py. By default cmd_replay loads the LATEST snapshot. If to_tick is at or below that snapshot's tick, it runs zero ticks, saves the loaded state and compares it with the same snapshot. With the run's tick-100 snapshot deliberately corrupted, the output was: "replay --to-tick 100 --verify  -> rc=0 ['replayed to tick 200 -> runs/_replays/...@100/tick_000000000200.zarr', 'VERIFY: identical']". This contradicts the cmd_replay docstring ("compare byte-for-byte against the run's own snapshot at that tick"). cli.py:161-178.
- (history) **PROVEN at function level: the single-writer lease is granted to several processes when they reclaim a stale lease at the same time (TOCTOU between read, unlink and O_EXCL create in rundir.acquire_lease)** — Proven only with barrier-synchronised acquire_lease() calls in forked processes on a lease left by a dead pid: 61 of 150 trials gave the lease to both of 2 writers, and 125 of 150 with 12 writers. Repro: /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history/repro_lease_race.py (arg = number of writers). Not shown with two full CLI processes, whose startup jitter makes the window much rarer in practice. Violates DECISIONS 2026-09-25 "A live holder blocks other writers".
- (history) **PROVEN but left out because of the cap: `firmament seed` overwrites the tick-T snapshot it loaded with a post-seed state, so an honest `replay --to-tick T --verify` then reports MISMATCH, and the pre-seed state at T is destroyed** — This one is proven. Repro: /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history/repro_seed_overwrites_snapshot.py. Before the seed: 'VERIFY: identical'. After: tick_200 causal_n goes 0 -> 1, and replay 100->200 --verify gives rc=1 "VERIFY: MISMATCH in ['p_child', 'p_motifs', 'p_state', ...]". Periodic snapshots at T are taken before the tick-T boundary commands; the one cmd_seed writes is taken after them (cli.py:213). Severity is debatable: it breaks verification and the 'fork first snapshot equals parent' invariant for the M1 fork-then-seed protocol.
- (history) **Analysis milestone candidate_mutant_depth100 reads lineage.sqlite, which is not in the snapshot (buffered up to 10000 rows, not truncated on resume, not copied on fork), so analysis.jsonl can differ between resumed or forked runs and an uninterrupted run** — Needs a lineage at least 100 generations deep to fire, which is beyond the CPU budget for a repro. The code path is milestones.py (lin.depth_of) plus the unflushed lineage buffer.
- (history) **The auto-restart counter (.restart_count) is never reset after successful progress, so the 4th transient device fault over a run's whole life stops it for good, even if the faults are months apart** — Would need an injected fake 'Warp error' plus an execv loop. Code: cli.py:139-149 only ever increments the counter, or unlinks it and re-raises once it reaches 3.
- (history) **crash.json 'last_good_snapshot' uses glob('tick_*'), which also matches tick_X.zarr.tmp and tick_X.zarr.old, so a crash during a snapshot save would name the incomplete .tmp directory as the last good snapshot** — Needs an I/O fault injected mid-save (for example ENOSPC). Code: cli.py:93.
- (history) **The natural-event RNG key at_tick*1000 + event_n collides once causal_n reaches 1000, so two different earthquakes get identical noise** — Needs at least 1000 causal events to reach. Code: console.py apply_natural np_rng(..., at_tick * 1000 + event_n).
- (observe) **PROVEN but left out only because of the 3-bug cap: a volcano with an odd mineral_counts breaks the exact element audit (console.py:96 vs :100). It adds inj//2 H2S per cell but registers inj.sum()//2 in total.** — Not unproven. /tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/observe/repro_volcano_odd.py prints 'net unregistered element change after apply: [-28, 0, 0, 0, 0, -14, 0]' and 'AuditError: element conservation violated at tick 10: [-28, 0, 0, 0, 0, -14, 0]' for mineral_counts=10001 and for 1. The even control (10000, the GUI default) balances exactly. It is reachable by typing any odd number into the GUI field. Output is in volcano_odd.out.
- (observe) **A WebSocket viewer can crash the sim thread: /ws/state stores client-supplied `layers` without validation, and build_view runs inside the sim loop (refresh_view instrument). {'layers':['motif:x']} gives int('x') ValueError; a non-list or non-string entry gives TypeError/AttributeError. The design says 'the sim never waits on the GUI'.** — httpx and websocket test clients are not installed. I read the code path (api.py:57, 80-87, 293) but did not run a live websocket session.
- (observe) **Developer edits (set_species, set_field on water/temp, edit_polymer changing p_len) never register with the audit, so the next audit raises AuditError and the live sim halts.** — Found by reading developer.py/audit.py only; no repro run. It is also arguable whether developer mode is meant to tolerate audit failure.
- (observe) **Ctrl-C raises KeyboardInterrupt, which `except Exception` in cli._guarded_loop does not catch. The shutdown snapshot and the lineage/metrics flush are skipped, although the RUNBOOK says 'Ctrl-C ... A shutdown snapshot is written on clean exit'.** — No SIGINT test was run against a live CLI process. uvicorn runs in a daemon thread, so it should not install signal handlers, but I have not verified that.
- (observe) **/api/inspect/cell does not bounds-check x,y. Negative values silently return a different cell's data under the requested coordinates, and x>=w or y>=h raises IndexError (HTTP 500).** — Found by reading the code only. It is minor, so I did not write a repro.
- (observe) **Natural-event parameters are never range-checked. A drought strength above 1 or a negative rain intensity_mm drives vapor negative. The audit registers the change, so the books balance, but the state is unphysical.** — Not executed. It may be classed as operator error rather than a defect.
- (observe) **Novelty's 'the seed's own combo is not news' rule treats whichever mask enters seen_combos first as the seed's. If the first sample already sees several combos, a genuinely new combo can be swallowed and the seed's combo reported as novel.** — This needs a run where mutants with new motif masks exist before the first metrics sample. I did not construct one.
