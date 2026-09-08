# The 5 × 14 grid, defined

The classifier is a coordinate system: **5 domains × 14 modes = 70 cells**. The
mode axis has been documented since it was designed. **The domain axis never
was** — 730 entries were filed against five words whose meaning was nowhere
written down, and the result was measured in 2026-09-07's Stage 7 audit: 43.3%
of a deterministic sample was filed in a cell its own anchor did not describe.

This file is the missing definition. It is the contract the anchors implement
and the standard entries are filed against.

## Why the definition was needed before anything could be re-filed

The 70 anchors in `config/mode_anchors_ontology.json` were internally
consistent but each domain was anchored in a **single register**:

| domain | anchor category | register |
|---|---|---|
| HEAL | `Body` | human physical |
| CONSTRUCT | `Build` | construction site |
| FABRICATE | `Make` | workshop / factory |
| BOND | `Connect` | human social |
| STUDY | `Mind` | human intellectual |

That is a human-daily-life ontology. It serves personal-history classification
well. It cannot serve machine telemetry, agent architectures, office documents
or semiotics — which is most of what the corpus actually contains, and all of
what the product is sold to classify. An entry like `Coolant & Lubrication`
filed in HEAL/REST is not obviously wrong; it is wrong *against an anchor that
says "rest day, recover, sleep deeply"* and right against the idea those words
were standing in for.

So the audit's 43% was measuring two different defects at once: entries in
genuinely arbitrary cells, and entries filed correctly under a reading of the
grid the anchors did not express. Re-filing against the old anchors would have
resolved the second by deleting the product's actual subject matter.

## The domain axis — what the work is directed at

Five kinds of object. Read the domain as **what is being acted on**, and the
mode as **what act**.

| domain | the object | human | machine | organisation |
|---|---|---|---|---|
| **HEAL** | **condition** — the state and capacity of something already running | sleep, training, injury | wear, coolant, condition monitoring | burnout, service health |
| **CONSTRUCT** | **structure** — the durable arrangement other work happens inside or on top of | a house, a habit's scaffolding | a rig, a platform, a schema | a department, a process |
| **FABRICATE** | **output** — discrete artifacts and throughput a process emits | a meal, a piece of writing | parts, batches, builds | deliverables, reports |
| **BOND** | **relation** — coupling between agents or components | friendship, conflict | bus, protocol, interface | contracts, accounts |
| **STUDY** | **knowledge** — acquiring and reasoning about information | learning, reflection | measurement, telemetry analysis | research, audit |

Condition, structure, output, relation, knowledge.

### The two distinctions that carry the most weight

**CONSTRUCT vs FABRICATE.** CONSTRUCT is the durable substrate — it stays, and
is inhabited or used. FABRICATE is what a process emits — units that leave.
Building the factory is CONSTRUCT; running the line is FABRICATE. Designing the
schema is CONSTRUCT; running the pipeline that produces the report is
FABRICATE.

**HEAL vs FABRICATE, under MAINTAIN.** Register-neutrality collapses some
distinctions the old anchors got for free by talking about bodies in one cell
and machines in another. Where two cells now compete, the domain decides by
object: HEAL/MAINTAIN preserves **condition** (servicing against degradation,
health-driven), FABRICATE/MAINTAIN preserves **the ability to keep producing**
(tool wear, changeover, consumables, keeping the line fed).

### What the domain axis is not

It is not a subject-matter taxonomy. "This is about machines" does not make it
FABRICATE, and "this is about people" does not make it BOND. A vibration
spectrum is STUDY/SENSE, not FABRICATE. A team's morale is HEAL, not BOND.

## The mode axis — what act

Unchanged, and as originally designed. The first six are the activity-energy
axis; the last eight extend it, with DESTROY completing the Greimas opposition
against CREATE and PLAN converging against BRAINSTORM's diverge.

| mode | act |
|---|---|
| LIFT | peak, intensity, maximum effort or load |
| REST | recover, idle, pause |
| WALK | steady flow, ongoing operation at nominal rhythm |
| WORK | execute, labour, do the routine doing |
| CREATE | make or design something that did not exist |
| LEARN | study or practise, self-directed acquisition |
| DESTROY | tear down, remove, end |
| SENSE | perceive, measure, observe |
| GUIDE | lead, mentor, direct others |
| PLAY | explore or improvise, low stakes |
| BRAINSTORM | ideate, diverge |
| MAINTAIN | upkeep, repair, preserve against decay |
| MOVE | relocate, transport, logistics |
| PLAN | sequence, organise, converge |

**WALK vs MAINTAIN** were near-duplicates in the old anchors (`CONSTRUCT/WALK`
"Steady Upkeep" against `CONSTRUCT/MAINTAIN` "Building Maintenance" shared
almost all their meaning). They are now separated: WALK is the *rhythm of
normal operation*, MAINTAIN is *work done against decay*. A platform serving
traffic is WALK; patching it is MAINTAIN.

## What an anchor is for

Each cell has exactly one anchor, and the classifier embeds **name + hints** —
nothing else in the entry affects classification. An anchor's hints are
therefore the cell's semantic centroid, and they now carry all three registers
deliberately: a cell that only speaks human cannot attract machine records, and
a grid whose cells cannot attract the data the product ingests is not a
coordinate system, it is decoration.

Hints stay **mode-pure and domain-pure**: register varies in the noun, never in
the verb. `HEAL/REST` says "sleep deeply" and "cool down after the run" and
"service in standby" — three registers of one act on one object.

## Status

The anchors implementing this contract landed with the Gate 0 regeneration; see
`research/perspective-discovery/FINAL_REPORT.md` in the physis-pro superproject
for the audit that forced it and the measurement of whether it worked.
