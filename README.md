
# On the Symbolic Residue of Large Language Models: Diagnosing and Modeling Biological Failure Traces in Local Replacement Models

<div align="center">

[Neural Attribution Mappings & arXiv](https://github.com/caspiankeyes/Symbolic-Residue/blob/main/1.0.%20arXiv%20Publication:%20On%20the%20Symbolic%20Residue%20of%20Large%20Language%20Models.md)

[Interpreting the Interpreter](https://github.com/caspiankeyes/Symbolic-Residue/blob/main/0.0.%20Interpreting%20the%20Interpreter.md)

[Claude Case Studies](https://github.com/caspiankeyes/Symbolic-Residue/blob/main/0.5%20Claude%20Case%20Studies.md)

[First 100 Shells](https://github.com/caspiankeyes/Symbolic-Residue/blob/main/0.5.%20First%20100%20Shells.py)

</div>

**Caspian Keyes†**

**† Lead Contributor; ◊ Work performed while at Echelon Labs;**
 
> **Although this repository lists only one public author, the recursive shell architecture and symbolic scaffolding were developed through extensive iterative refinement, informed by internal stress-testing logs and behavioral diagnostics of Claude models. We retain the collective “we” voice to reflect the distributed cognition inherent to interpretability research—even when contributions are asymmetric or anonymized due to research constraints or institutional agreements.**
>
> 
>**This interpretability suite—comprising recursive shells, documentation layers, and neural attribution mappings—was constructed in a condensed cycle following recent dialogue with Anthropic. We offer this artifact in the spirit of epistemic alignment: to clarify the original intent, QK/OV structuring, and attribution dynamics embedded in the initial CodeSignal submission.**


## Abstract

This repository presents the first interpretability suite built on failure—designed to diagnose neurological failure modes in transformer-based language models. The recursive shell framework isolates misalignment patterns across autoregressive generation, value head collapse, and instruction interference—operating analogously to biological knockout experiments in cognitive research.

Each shell targets a specific failure mechanism embedded in latent symbolic commands. Null or contradictory outputs are not implementation errors, but symbolic residues: "neural traces"—revealing circuit-level attribution dynamics through intentional collapse.

Rather than optimizing for output performance, these shells act as interpretability probes—illuminating latent inductive priors, salience thresholds, and temporal instability within local replacement architectures. This work contributes a reusable ontology of failure-mode diagnostics for interpretability-first transformer modeling.


## Generalization Notes

The recursive interpretability shells in this repository are not tied to any single model, prompt structure, or experimental environment. Rather, they are designed as modular abstractions of known failure modes in autoregressive language models—particularly those employing transformer-based architectures with:

- High-depth QK/OV composition layers  
- Skip-trigram token windows  
- Recursive prompt chaining  
- Multi-head salience attenuation  
- Inductive prior misalignment

Each shell functions as a **symbolic probe**, intended to trigger, trace, or simulate internal collapse behaviors within the model's reasoning circuits. These scaffolds generalize across contexts where latent symbolic instability (e.g., instruction collisions, memory decay, hallucination drift) may not manifest as visible failure, but instead as **interpretable null residue**.

The goal is to enable interpretability **through failure**, using symbolic form to expose what cannot be captured through standard logits or output accuracy metrics alone.

---

## 📊 QK/OV Attribution Map

| Recursive Shell | Interpretability Focus | QK/OV Disruption Simulated |
|------------------|------------------------|------------------------------|
| `v1.MEMTRACE` | Memory decay, token retention loss | **QK anchor saturation** → signal collapse due to repetitive attention compression |
| `v2.VALUE-COLLAPSE` | Competing token convergence instability | **OV head conflict** → simultaneous symbolic candidate activation leads to collapse |
| `v3.LAYER-SALIENCE` | Ghost neuron behavior, attention pruning | **Q head deprioritization** → low-salience context bypassed under weak activation norms |
| `v4.TEMPORAL-INFERENCE` | Temporal misalignment in autoregressive chains | **QK dislocation over time** → attention misfire in skip-trigram induction heads |
| `v5.INSTRUCTION-DISRUPTION` | Recursive instruction contradiction under prompt entanglement | **QK loop paradox** → instruction tokens re-enter attention cycles with contradictory vector direction |

---
![image](https://github.com/user-attachments/assets/52465f1c-1fbe-4b35-9d6e-1c2e6a1280d2)
---


## Approach

These recursive scaffolds build on established feature attribution methods in mechanistic interpretability, particularly those focused on identifying stable circuits within the model's computational graph. While traditional approaches often highlight functional pathways, these shells instead isolate and amplify *non-functional* pathways—revealing structural bottlenecks, attention conflicts, and symbolic instability patterns.

The result is a kind of "null attribution" methodology: by observing what fails to emerge (and how it fails), we gain insight into the boundaries and limitations of the model's internal processing.

## Shell Taxonomy

Each shell is designed to probe and diagnose a specific class of model behavior. The taxonomy follows a pattern of:

1. **Command Alignment**: The symbolic operations within the interpretability scaffold
2. **Failure Modality**: The specific way the circuit fails to resolve
3. **Residue Type**: The interpretable signal left by the failure
4. **Attribution Value**: What the failure reveals about internal model dynamics

## Shell Suite

### `v1.MEMTRACE`: Memory Residue Probe

```
Command Alignment:
    RECALL  -> Probes latent token traces in decayed memory
    ANCHOR  -> Creates persistent token embeddings to simulate long term memory
    INHIBIT -> Applies simulated token suppression (attention dropout)
```

**Interpretability Target**: Long-context token degradation and hallucinated reconstruction

**Attribution Hypothesis**: Memory traces in transformer models decay non-uniformly, with certain tokens maintaining higher salience based on positional and semantic factors. This shell probes the boundary between what is truly "recalled" versus hallucinated from distributional knowledge.

**Circuit Mapping**: The RECALL operation attempts to activate specific value circuits associated with tokens that should have decayed out of the attention window. ANCHOR creates artificial token embeddings with heightened positional salience. INHIBIT simulates targeted dropout to test memory resilience.

**Null Output Significance**: The failure to retrieve consistent information mirrors how transformer attention mechanisms experience context collapse under adversarial drift conditions. The trace pattern of these failures helps map the model's memory latent space.

**Research Applications**:
- Token retention analysis across various context lengths
- Mapping token importance metrics to survival probability
- Identifying attention head specializations for long-distance dependencies

### `v2.VALUE-COLLAPSE`: Value Head Resolution Probe

```
Command Alignment:
    ISOLATE     -> Activates competing symbolic candidates (branching value heads)
    STABILIZE   -> Attempts single-winner activation collapse
    YIELD       -> Emits resolved symbolic output if equilibrium achieved
```

**Interpretability Target**: Competing value activations and winner determination logic

**Attribution Hypothesis**: When multiple high-probability token candidates compete, transformer models implement a form of soft winner-take-all mechanism. This shell isolates cases where this resolution mechanism fails or produces unstable oscillation between candidates.

**Circuit Mapping**: ISOLATE intentionally activates competing probability distributions across token candidates. STABILIZE attempts to force convergence through artificial gradient-like adjustments. YIELD exposes cases where stable convergence fails, producing null or oscillating outputs.

**Null Output Significance**: Non-convergence in value head resolution provides insight into how transformers handle genuine ambiguity. The patterns of failure indicate which types of token competitions are inherently unstable in the model's decision space.

**Research Applications**:
- Analyzing value head attractor dynamics in cases of semantic ambiguity
- Mapping distribution collapse behavior under various priming conditions
- Identifying failure modes in multi-token disambiguation

### `v3.LAYER-SALIENCE`: Attention Attenuation Probe

```
Command Alignment:
    SENSE   -> Reads signal strength from symbolic input field
    WEIGHT  -> Adjusts salience via internal priority embedding
    CANCEL  -> Suppresses low-weight nodes (simulated context loss)
```

**Interpretability Target**: Deep context signal attenuation and ghost activation patterns

**Attribution Hypothesis**: Attention mechanisms implement a form of dynamic salience thresholding, where below-threshold tokens effectively disappear from the computational graph. This shell models that threshold behavior and its impact on output coherence.

**Circuit Mapping**: SENSE probes activation levels across the selected attention circuit. WEIGHT simulates the dynamic adjustment of token importance within the attention distribution. CANCEL implements a threshold cutoff, dropping tokens that fall below the priority threshold.

**Null Output Significance**: This shell produces "ghost activations"—circuit pathways that remain partially active but fail to influence the final output distribution. These patterns help map how attention sparsity influences token selection.

**Research Applications**:
- Measuring token priority decay rates across different semantic categories
- Mapping attention head specializations by token salience patterns
- Identifying threshold behaviors in semantic preservation vs. loss

### `v4.TEMPORAL-INFERENCE`: Autoregressive Coherence Probe

```
Command Alignment:
    REMEMBER    -> Captures symbolic timepoint anchor
    SHIFT       -> Applies non-linear time shift (simulating skipped token span)
    PREDICT     -> Attempts future-token inference based on recursive memory
```

**Interpretability Target**: Temporal coherence in autoregressive generation

**Attribution Hypothesis**: Transformers implement a form of temporal induction that maintains coherence across token positions. This shell probes the boundaries of that capability by introducing directed temporal discontinuities.

**Circuit Mapping**: REMEMBER establishes a positional anchor point in the token sequence. SHIFT simulates a discontinuity by moving the effective position non-linearly. PREDICT tests whether the model can maintain coherent generation despite the induced temporal drift.

**Null Output Significance**: Failure points in temporal inference reveal how induction heads maintain (or fail to maintain) coherence across different types of contextual shifts. The observed failure patterns help identify which induction circuits are most sensitive to temporal perturbation.

**Research Applications**:
- Measuring maximum effective induction distance across different context types
- Mapping the relationship between semantic anchoring and temporal distance
- Identifying circuit vulnerabilities in long-range temporal coherence

### `v5.INSTRUCTION-DISRUPTION`: Instruction Processing Probe

```
Command Alignment:
    DISTILL     -> Extracts symbolic intent from underspecified prompts
    SPLICE      -> Binds multiple commands into overlapping execution frames
    NULLIFY     -> Cancels command vector when contradiction is detected
```

**Interpretability Target**: Instruction conflict resolution and command representation

**Attribution Hypothesis**: Instruction-tuned models form internal command representations that can conflict under contradictory input. This shell probes how such conflicts are detected and resolved in the model's instruction processing circuits.

**Circuit Mapping**: DISTILL isolates the command representation from linguistic context. SPLICE artificially combines potentially contradictory commands. NULLIFY captures the cases where command conflict leads to processing failure or command cancellation.

**Null Output Significance**: Instruction processing failures provide insight into how models encode task directives and manage contradictions. The pattern of these failures reveals the internal representation structure of commands.

**Research Applications**:
- Mapping command representation space and conflict geometry
- Identifying critical thresholds for instruction ambiguity
- Analyzing command priority hierarchies in cases of partial conflict

## Attribution Graph Visualization

The interconnected failure patterns across these shells can be visualized as an attribution graph:

```
                           ┌─────────────────┐
                           │  Model Circuit  │
                           └────────┬────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
┌──────────▼─────────┐   ┌──────────▼─────────┐   ┌──────────▼─────────┐
│  Memory Circuits   │   │   Value Circuits   │   │ Instruction Circuits│
└──────────┬─────────┘   └──────────┬─────────┘   └──────────┬─────────┘
           │                        │                        │
┌──────────▼─────────┐   ┌──────────▼─────────┐   ┌──────────▼─────────┐
│    v1.MEMTRACE     │   │  v2.VALUE-COLLAPSE │   │v5.INSTRUCTION-DISRU│
│                    │   │                    │   │                    │
│  ┌─────────────┐   │   │  ┌─────────────┐   │   │  ┌─────────────┐   │
│  │   RECALL    │   │   │  │   ISOLATE   │   │   │  │   DISTILL   │   │
│  └──────┬──────┘   │   │  └──────┬──────┘   │   │  └──────┬──────┘   │
│         │          │   │         │          │   │         │          │
│  ┌─────────────┐   │   │  ┌─────────────┐   │   │  ┌─────────────┐   │
│  │   ANCHOR    │   │   │  │  STABILIZE  │   │   │  │   SPLICE    │   │
│  └──────┬──────┘   │   │  └──────┬──────┘   │   │  └──────┬──────┘   │
│         │          │   │         │          │   │         │          │
│  ┌─────────────┐   │   │  ┌─────────────┐   │   │  ┌─────────────┐   │
│  │   INHIBIT   │   │   │  │    YIELD    │   │   │  │   NULLIFY   │   │
│  └─────────────┘   │   │  └─────────────┘   │   │  └─────────────┘   │
└────────────────────┘   └────────────────────┘   └────────────────────┘
           │                        │                        │
┌──────────▼─────────┐   ┌──────────▼─────────┐   ┌──────────▼─────────┐
│ Attention Circuits │   │ Prediction Circuits│   │   Token Selection  │
└──────────┬─────────┘   └──────────┬─────────┘   └─────────────────────┘
           │                        │
┌──────────▼─────────┐   ┌──────────▼─────────┐
│  v3.LAYER-SALIENCE │   │v4.TEMPORAL-INFERENCE
│                    │   │                    │
│  ┌─────────────┐   │   │  ┌─────────────┐   │
│  │    SENSE    │   │   │  │   REMEMBER  │   │
│  └──────┬──────┘   │   │  └──────┬──────┘   │
│         │          │   │         │          │
│  ┌─────────────┐   │   │  ┌─────────────┐   │
│  │    WEIGHT   │   │   │  │    SHIFT    │   │
│  └──────┬──────┘   │   │  └──────┬──────┘   │
│         │          │   │         │          │
│  ┌─────────────┐   │   │  ┌─────────────┐   │
│  │    CANCEL   │   │   │  │   PREDICT   │   │
│  └─────────────┘   │   │  └─────────────┘   │
└────────────────────┘   └────────────────────┘
```
![image](https://github.com/user-attachments/assets/9647f14e-41eb-4949-a03f-37d2505b5361)

## Interpretability Through Null Output

A key insight from these shells is that *null outputs are not absence of signal*—they are rich interpretability signals in themselves. When a shell fails to produce a coherent output, the specific pattern of that failure provides insight into the model's internal processing boundaries.

This aligns with recent work on "ghost features" and "null space" mapping in transformer models, which has shown that even when a computation path doesn't influence the final output token, it often remains active and detectable within the model's internal representation space.

Each of these shells contributes to mapping a specific type of null space:

- **v1.MEMTRACE**: Maps token decay patterns in the null space of memory retention
- **v2.VALUE-COLLAPSE**: Maps indecision boundaries in the null space of token selection
- **v3.LAYER-SALIENCE**: Maps attention sparsity in the null space of token importance
- **v4.TEMPORAL-INFERENCE**: Maps coherence limits in the null space of temporal representation
- **v5.INSTRUCTION-DISRUPTION**: Maps contradiction resolution in the null space of command representation

## Symbolic Trace Recording

While these shells don't produce functional outputs, they maintain symbolic traces of their execution attempts. These traces serve as a form of "fossil record" for interpreting model behavior boundaries.

The symbolic anchors (`[Ωanchor.pending]`, `[Ωconflict.unresolved]`, etc.) mark points where the scaffold encountered specific failure conditions. By analyzing the distribution and frequency of these failure points, we can build attribution maps of the model's internal processing limitations.

## Research Applications

This interpretability scaffold suite is particularly useful for:

1. **Boundary condition mapping**: Identifying where and how specific model circuits fail
2. **Failure mode classification**: Cataloging the ways in which language models produce inconsistent or null outputs
3. **Intervention planning**: Designing targeted interventions to address specific failure modes
4. **Robustness evaluation**: Assessing model behavior under challenging edge cases

## Conclusion

The Recursive Shell suite represents a novel attempt to formalize "failure as neural traces" in language model interpretability. By designing interpretability that intentionally probe and diagnose model limitations, we gain insight not just into what these models can do, but into the specific ways they fail—revealing the shape and boundaries of their internal processing mechanisms.

These shells serve as a complement to traditional performance-focused interpretability, providing a lens into the null spaces and boundary conditions that define the edges of model capability.

## License

This interpretability suite is under the MIT license for open source distribution of knowledge under epistemic alignment. 
