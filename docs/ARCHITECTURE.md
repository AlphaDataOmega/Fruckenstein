# Fruckenstein architecture

## 1. Physical flock

Three complete Open Duck Mini bodies share one MuJoCo world. Invisible torso hulls and the ordinary foot/body collision geometry make group contact physical rather than decorative. Seven apples are represented as non-colliding targets and replenished after consumption.

Each duck has its own:

- actuator and sensor slice;
- gait phase;
- Candidate-0003 crown/voxel organ;
- appetite, nociception, vitality, and mortality state;
- persistent relational effect ledger;
- morphology/repertoire/adapter fingerprint.

## 2. Body pressure

The body is serialized into four controller-facing regions: crown, hips, knees, and ankles/root. Each region derives a bounded three-component pressure from local velocity, actuator load, and command residual. Neighbor and aura coupling exchange bounded summaries.

These regions are an implementation mapping, not a claim that the controller has discovered anatomy.

## 3. Null-preserving voxel evidence

The voxel field contains:

- 27 lattice cells in `{-1, 0, 1}³`;
- 12 proposal cells having exactly two non-zero coordinates;
- 8 cube-corner direction classes;
- 1 explicit null class.

Relational tokens land according to direction and reliability. Persistent belief is normalized and locally diffused. Null evidence remains null and contributes no fabricated outward vector. Feedback from a prior choice writes into believed state; it does not bypass the field as a direct actuator command.

## 4. Candidate-0003 crown

The frozen crown contains ten peripheral funnel members plus two eye members, reduced through a three-node axial spine. It receives pressure in local frames and returns either:

- a bounded disposition when coherence, strength, and confidence gates agree; or
- an exact zero refusal.

The frozen source, parameters, geometry, and expected checksums live under `seat/crown-seat-candidate-0003-handoff/`.

## 5. Dual-timescale embodiment

The crown is the slower commitment layer. The gait/servo loop runs faster and plays an accepted relation through a coordinated whole-gait repertoire.

Exploration changes a bounded whole-gait relation—phase, blend, amplitude, or trim—rather than injecting unrelated random noise into every joint. Once a relation has adequate support and predicts its observed consequence, exploration is reduced.

## 6. Memory and body fingerprint

Persistent memory stores predicted consequences, support, learned steering trim, and gait enactment. A SHA-256 fingerprint binds memory to:

- the body-adapter version;
- the inherited gait relation;
- steering range;
- repertoire bytes;
- canonical body XML bytes.

Changing the body, repertoire, or adapter invalidates direct inheritance and requires rebaking or an explicit migration.

## 7. Motivation and mortality

Appetite rises with deprivation and decreases after eating. Nociception is derived from physical risk and unexpected consequence. These signals modulate movement, evidence reliability, and vitality.

When vitality reaches the death transition, active control stops while the body remains collidable. This preserves group consequences. None of these operational signals asserts subjective experience.

## 8. Generational development

Each candidate flock begins with fresh bodies and a copy of the same parent ledgers. Birth conditions vary deterministically, including gait phase, crown rhythm, and world seed.

A candidate is ineligible if any required gate fails. The included bake evaluates:

- finite numerical state;
- all bodies alive;
- target consumption;
- no falls;
- minimum uprightness;
- bounded nociception;
- bounded gait enactment;
- bounded steering trim.

Only an eligible winner can replace the parent ledger. If none qualifies, the parent is retained.
