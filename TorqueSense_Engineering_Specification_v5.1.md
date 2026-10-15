# TorqueSense — CAD-Compiled Assembly Verification, for Any Assembly

### Engineering, product and experience specification

**PS4 · Real-time Assembly Process Verification Using Vision LLMs · Tata Motors** — v5.1, universal profile engine · camera capture · production platform · software-only build

Scope: any assembly whose steps change something visible or signal the PLC · Reference profile: tyre fitting, 5-lug hub, 114.3 mm PCD · Target machine: **HSGPC**

> This specification is written for one machine — yours — and one build: the software-only
> prototype. Every version, path, memory figure and command has been chosen against the system
> details you supplied. It consolidates every decision taken so far (software-only build,
> CAD-generated training data, Florence-2 as a local adjudicator, CUDA 13.2 with TensorRT
> removed) and corrects the places where earlier drafts were wrong. Those corrections are listed
> in Appendix A rather than buried.
>
> **What v4.1 adds.** An operations track that treats the stations as a fleet of devices:
> per-station health telemetry with statistical drift detection, a fail-visible heartbeat, a
> Jenkins pipeline for the GPU-bound gates, container images and Kubernetes manifests tested in
> CI, and a small Go fleet agent. Each addition passes the scope test in Section 1 on
> engineering grounds, not as decoration. Most were checked before they were written down: the
> drift limits were set by simulation, the agent and the soft PLC's watchdog were compiled and
> run, and the manifests and pipelines were checked by real validators. That checking found an
> OPC-UA instruction in v4.0 that would have stopped the soft PLC from starting — Appendix A,
> item 18.
>
> **What v5.0 changes.** The engine is no longer shaped around one wheel. Every assembly is a
> *profile*: inspection points with ordered state families, steps with order, precedence,
> deadlines and evidence, compiled from CAD plus a short manifest (Sections 6–11). Tyre fitting
> stays the reference profile. The problem statement's other two examples — the door-panel harness
> clip and the windshield gasket — and a non-automotive electronics enclosure are written as
> profiles too, and all four run through one interpreter with no assembly-specific code. That
> interpreter was built and tested before this revision was written: 24 tests, two of which caught
> design errors in its first draft. Building it also exposed a resolution problem v4.1 never
> checked (Appendix A, item 20).
>
> **What v5.1 adds.** Two things the prototype could skip and a plant cannot. **Camera capture**
> (Section 8): one frame-source interface with file, webcam and industrial GigE implementations;
> a named production camera; a lens chosen by the compiler from the same 16 px rule; a link budget
> that decides the pixel format and the network card; zero-copy grabbing into the engine's own
> frame pool; and a commissioning checklist. **A production platform** (Section 32): one tool
> per layer — edge PC and GPU, RKE2, the NVIDIA GPU Operator, Flux, Harbor, cosign with Kyverno,
> CloudNativePG with TimescaleDB, Garage, and the platform's own observability — and the path a
> release takes from a pull request to a station. The lens and link arithmetic is in the reference
> implementation and tested (29 tests now); the fleet agent gained a dcgm-exporter GPU source; the
> manifests validate against Kubernetes 1.36. Researching the platform overturned three v5.0
> choices (Appendix A, items 24–26).

## How to read this document

This is a **single recommended path**, not a menu. Every tool is the one pick for its layer, with
the reason it beat the alternatives. Parts I–II say what to build and why; Part III covers the
platform around the engine; Part IV is the build and the pitch; Part V takes the build from an
empty laptop to a running demo on *this* laptop.

When the platform and the verification engine compete for hours, **the engine wins**. A polished
dashboard around a detector that misreads part states loses to a plain one around a detector that
does not.

### Your machine at a glance

| | |
|---|---|
| Device | HSGPC |
| OS | Windows 11 **Home**, 26H2, build 26300.9550 |
| CPU | Intel Core i7-11800H — 8 cores / 16 threads, Tiger Lake, AVX2 + AVX-512 |
| RAM | 16 GB (15.7 GB usable) |
| GPU | NVIDIA GeForce RTX 3060 Laptop, 6 GB, Ampere, compute capability 8.6 · plus Intel UHD (display) |
| CUDA Toolkit | **13.2**, V13.2.78 (nvcc built 19 Mar 2026) |
| Python | **3.12.0** |
| Storage | ~356 GB free of 954 GB |
| User path | `C:\Users\Gurjas Gandhi` — **contains a space** |

Section 25 turns this into a component-by-component verdict.

## Contents

**Part I — Position**

- [1. Scope and scoring logic](#1-scope-and-scoring-logic)
- [2. What is already solved, and what is not](#2-what-is-already-solved-and-what-is-not)
- [3. Users and scenarios](#3-users-and-scenarios)

**Part II — System**

- [4. System architecture](#4-system-architecture)
- [5. End-to-end verification flow](#5-end-to-end-verification-flow)
- [6. The procedure model — any assembly, as data](#6-the-procedure-model--any-assembly-as-data)
- [7. The CAD compiler and pattern solvers](#7-the-cad-compiler-and-pattern-solvers)
- [8. Camera capture](#8-camera-capture)
- [9. Perception for any part](#9-perception-for-any-part)
- [10. Evidence beyond the camera](#10-evidence-beyond-the-camera)
- [11. Onboarding a new assembly](#11-onboarding-a-new-assembly)
- [12. Latency and capacity model](#12-latency-and-capacity-model)

**Part III — Platform**

- [13. Process and thread model](#13-process-and-thread-model)
- [14. Identity and access](#14-identity-and-access)
- [15. Data architecture](#15-data-architecture)
- [16. Verification and validation](#16-verification-and-validation)
- [17. Industrial and safety parameters](#17-industrial-and-safety-parameters)
- [18. Fleet health and reliability](#18-fleet-health-and-reliability)

**Part IV — Build**

- [19. Tool register](#19-tool-register)
- [20. Interface design](#20-interface-design)
- [21. Prototype build plan](#21-prototype-build-plan)
- [22. Demo script](#22-demo-script)
- [23. Risk register](#23-risk-register)

**Part V — Setup and operations, on HSGPC**

- [24. Setup roadmap](#24-setup-roadmap)
- [25. Your machine — compatibility verdict](#25-your-machine--compatibility-verdict)
- [26. Compute — what the Colab T4 is for](#26-compute--what-the-colab-t4-is-for)
- [27. Installation](#27-installation)
- [28. Configuration](#28-configuration)
- [29. Integration](#29-integration)
- [30. Testing](#30-testing)
- [31. Deployment](#31-deployment)
- [32. Production platform and hosting](#32-production-platform-and-hosting)
- [33. Caching strategy](#33-caching-strategy)

**Appendices**

- [\1. Corrections to earlier drafts](#appendix-a--corrections-to-earlier-drafts)
- [\1. Coding-agent kit deltas](#appendix-b--coding-agent-kit-deltas)
- [\1. Portfolio and interview mapping](#appendix-c--portfolio-and-interview-mapping)

---

# Part I — Position

## 1. Scope and scoring logic

PS4 grades a working verification system, not a platform. The rubric is explicit:

| Criterion | Weight | What scores 5 |
|---|---|---|
| Innovation | 20% | Strong novelty: CV + LLM integration for smart inspection |
| Technical Execution | 30% | High accuracy, robust anomaly and process validation |
| Business Relevance | 25% | Strong alignment with Tata Motors' quality operations |
| Visualization & UX | 15% | Clear video overlays and an intuitive inspection dashboard |
| Presentation & Demo | 10% | Smooth demo, strong business-impact storytelling |

The problem statement also names its own acceptance criterion: capture, detect a missing step, and
trigger an alert at the station **in under one second**. That sentence is the system's contract.

### The two tests

Every element must either **feed a graded criterion** or be **required by the one-second
contract**. Anything failing both was cut.

| Element | Verdict | Reason |
|---|---|---|
| CAD-compiled procedure | ✅ **build** | The innovation claim. Without it this is a hand-written rule engine. |
| Order, precedence and omission checks | ✅ **build** | The technical claim. Presence detection is commodity; order, precedence and omission are not. |
| Profile engine — any assembly as data | ✅ **build** | The problem statement names three different assemblies. A system that hard-codes one has answered a third of it. This is also the strongest innovation claim. |
| Three more profiles from the problem statement's own examples | ✅ **build**, staged | The door trim runs end to end; windshield and enclosure are compiled and verified at the procedure level — proof the engine is general without building four perception stacks (Section 21). |
| Lock-free C++ pipeline | ✅ **build** | Required by the contract, and it is what makes the latency number defensible. |
| Synthetic data from the same CAD | ✅ **build** | Turns the no-hardware constraint into a second innovation claim. |
| Real OPC-UA alert path | ✅ **build** | Business relevance: the alert speaks the protocol the plant already runs. |
| Florence-2 local adjudicator | ✅ **build** | The "LLM" half of the rubric's "CV + LLM", sized to fit the GPU. |
| Engineer dashboard + operator HMI | ✅ **build** | 15% of the score. |
| Fault-injection harness | ✅ **build** | Turns the demo from a recording into something a judge can break. |
| Four-station mode | ✅ **build** | Business relevance — scale shown, not asserted. |
| Station-fleet health monitoring | ✅ **build** | A camera inspector degrades silently: a smeared lens, a knocked mount, a throttling GPU. Catching that before it produces false alarms is the second question any plant asks, and the answer is business relevance. |
| Fail-visible heartbeat | ✅ **build** | A quality gate that silently stops checking is worse than none. Cheap, and the line's own PLC enforces it. |
| Jenkins pipeline on the laptop | ✅ **build** | The latency and accuracy gates need the GPU, so hosted CI cannot run them. Without automation they get run by hand, then forgotten. |
| Container images + Kubernetes manifests | ⚠️ **reframe** | Not part of the demo. They make Tier 2 concrete, and CI proves they work in a throwaway kind cluster at no cost to the laptop's 16 GB. |
| Go fleet agent | ✅ **build** | The engine cannot report its own death. A watcher must live outside the process it watches, and a static binary with no runtime is the right shape for an edge PC. |
| Anomaly log with clips | ✅ **build** | The PS asks for it by name: timestamp, station ID, video snippet. |
| Excel shift report | ⚠️ **reframe** | Not a feature for its own sake — it is how a quality engineer actually consumes this data on a Tata line. Built in C++, which also serves the Sun Fibo JD. |
| Login, accounts, roles | ❌ **cut** | Zero judge value on a single-laptop demo. Section 14 says what production needs. |
| Prometheus, Grafana, tracing | ❌ **cut** | Fleet health is stored in TimescaleDB, which is already running, and drawn by the dashboard. A second metrics stack would duplicate it for roughly 0.5 GB of RAM this machine cannot spare. In the plant, a platform stack watches the cluster itself (Section 32); station health stays in TimescaleDB. |
| Camera capture, specified for production | ✅ **build**, staged | Every verdict starts with a frame. The demo replays rendered video; the production camera, lens, link and capture code are specified and their arithmetic tested (Section 8), so the step to a real station is commissioning, not design. |
| Production platform and hosting | ⚠️ **reframe** | Not part of the demo. It answers the plant's first deployment question with one tool per layer (Section 32) and costs the laptop nothing. |
| Physical camera, ESP32, tower light | ❌ **cut** | Software-only build — no camera is bought for the demo. Section 12 of the hardware addendum is the upgrade path. |
| Qt operator HMI | ❌ **cut** | A second UI toolchain on a 16 GB machine. The HMI becomes a kiosk route of the same web app (Section 20). |

> **The governing trade.** When the platform track and the engine track compete for hours, the
> engine wins. Judges forgive a plain dashboard. They do not forgive a red light that fires on the
> correct sequence.

---

## 2. What is already solved, and what is not

A Tata Motors judge knows line-side quality tooling. Open with what exists, or they will.

### Solved elsewhere

| Capability | How it is done today | Limitation |
|---|---|---|
| Torque value per fastening | DC electric nutrunners with controllers record torque and angle for every rundown and flag out-of-spec values | Counts and values events. Has no geometric notion of *which* bolt was fastened. |
| Fastener sequence enforcement | Position-sensing torque reaction arms with encoders, or socket trays that release one socket at a time | Per-station hardware and fixtures, re-engineered for every wheel variant. Expensive to retrofit across a line. |
| Part presence | Rule-based machine vision at end-of-line | Confirms five bolts exist. Says nothing about the order they were tightened in. |
| Operator guidance | Pick-to-light, work-instruction screens | Tells the operator what to do. Does not verify what they did. |

### Not solved — the gap TorqueSense fills

| Gap | Why it matters | TorqueSense module |
|---|---|---|
| Sequence verification **without fixtures** | Position arms work, but cost scales per station and per variant | Camera-only verification, retrofittable to any station with a view of the work |
| Rules that follow design changes | Every hand-written rule set goes stale when engineering revises the part | **CAD compiler** — the procedure is derived from the assembly model |
| Training data that follows design changes | A new variant or a new assembly means a new labelled dataset | **Synthetic renders from the same CAD** — zero manual labelling |
| Explanation tied to geometry | A red light alone does not tell an engineer which part | Evidence linked to CAD inspection-point identities, ghost overlay, deterministic narrative |
| One system for every station | Each existing tool is engineered per station: a torque arm for wheels, a fixture for clips, a vision recipe per part | **Profiles** — one engine, one data file per assembly (Section 6) |
| Knowing the inspector itself is healthy | Camera systems fail quietly: a smeared lens lowers detection confidence for days before it causes a false alarm | **Fleet health** — per-station control charts that flag drift before it reaches the line (Section 18) |

> **Use this as your opening rebuttal.** "We already have torque arms" is the first objection a
> line engineer raises. The answer: torque arms solve sequence per station, with hardware, for one
> variant. TorqueSense solves it with a camera that reconfigures itself from CAD — and it does not
> replace the nutrunner controller, it sits beside it.

### What TorqueSense deliberately does not claim

**It does not measure torque.** A camera cannot see clamp load. The torque value stays the
nutrunner controller's job. TorqueSense verifies **sequence and completeness** — which bolts, in
what order, all of them. Integrating the controller's per-rundown result over OPC-UA is the Tier 2
step (Section 31). Say this before a judge asks; it is the most likely question from anyone who has
worked a line, and volunteering it is worth more than any feature.

**"Any assembly" has a definition.** TorqueSense verifies steps that change something visible at a
known point, or that the plant can signal. Cable routing, cosmetic defects and dimensional metrology
are different problems. Section 11 has the five-question check that says whether a station
qualifies — and the compiler refuses a station whose parts its cameras cannot resolve.

---

## 3. Users and scenarios

Six people. Every screen exists because one of them needs it. Demo them in this order.

### A — Rajesh, operator, Station W-14, 58-second takt

Gloves on, three metres from a screen, sodium lighting. He mounts the rear-left wheel, hand-starts
five M12 bolts, and torques them in the star pattern. Mid-cycle his supervisor calls him away. He
returns and resumes — from bolt 2, sequentially. Nothing he can see is wrong. **About seventy
milliseconds** after the out-of-order bolt is tightened, his screen floods red and draws the
corrective path across a hub schematic that matches the wheel in front of him, bolt for bolt. He
reads nothing.

*Forces:* zero-reading HMI, spatial correspondence between screen and wheel, an alert faster than
the next action.

### B — Priya, process engineer, end of shift

Forty anomalies across four stations. She opens the dashboard, sorts by station, and opens one.
Six-second clip, the frame-by-frame bolt-state strip, the CAD view with the offending bolt in red,
and a one-line explanation generated from the state machine's own facts. She marks two as false
positives — each writes a hard-negative sample back to the training set. She exports the shift to
Excel for the morning quality meeting.

*Forces:* evidence linked to geometry, a disagreement loop, the export format the plant actually uses.

### C — Arvind, manufacturing engineer, new wheel variant

Engineering releases a four-lug hub for a new model. Arvind opens the revised assembly in FreeCAD
and re-runs the compiler. The procedure regenerates — sequence 1, 3, 2, 4, derived from geometry —
and the synthetic training set regenerates from the same model overnight. No rules typed, no images
labelled.

*Forces:* the CAD compiler, the render pipeline, the claim that reconfiguration is a file, not a project.

### D — The judge

Handed the laptop and asked: *"Pick a fault. Any fault."* They click one. The system catches it.
Then they watch four stations run at once on the same machine.

*Forces:* a fault-injection harness that cannot influence the verdict, and a multi-station mode that
fits in 6 GB of VRAM.

### E — Meera, controls engineer, twelve stations' edge PCs

Monday, 07:40. Nothing is alarming. But the fleet view shows W-16's detector-confidence chart in
amber since Saturday night, with its ambiguous-detection rate creeping up beside it and inference
latency flat. The playbook pairs that pattern with lens contamination or a lighting change, not a
software fault. She cleans the lens at the 09:00 break; within the hour both charts are back
inside their limits. No false alarm ever reached an operator. That afternoon she rolls a retrained
model onto W-15 alone, compares it with its neighbours for one shift, and promotes it to the rest
of the line the next morning. When W-12's PC crashed last month, its station went amber in two
seconds and the line lead knew not to trust it.

*Forces:* drift detection that fires before failure, a playbook from signal pattern to likely
cause, per-station rollout and rollback, a station that goes visibly amber the moment its
inspector stops.

### F — Neha, industrial engineer, the door line

A different line, a different assembly: front-door trim, six harness clips, a twelve-pin connector,
then the trim panel. Neha has the door CAD from engineering and the station's PLC tags from
controls. She writes a forty-line manifest and compiles it. The compiler refuses: four clips at the
ends of the door span 15 px from cameras mounted at 500 mm. She lowers the camera bar to 420 mm and
recompiles clean. No model is trained — snap clips and connectors are already in the family
library. After a validation shift the station goes live on the same engine as the wheel line, and
its first real catch is a trim panel pressed home over an open clip.

*Forces:* profiles as data, a compiler that refuses unresolvable rigs, a reusable family library,
precedence checks, onboarding measured in days.

---

# Part II — System

## 4. System architecture

One C++ engine that runs any profile, one local vision-language sidecar, and for each assembly one
CAD model feeding both the rules and the training data. Nothing runs in the cloud at demo time.

```mermaid
flowchart TB
  subgraph DT["Design time — runs when the CAD changes"]
    CAD["Assembly CAD<br/>any product"] --> STEP["STEP AP214"]
    MAN[/"manifest.yaml<br/>groups · steps · evidence"/] --> CC
    CAD --> GLTF["glTF"]
    STEP --> CC["CAD compiler · C++ · OCCT<br/>IPs · pattern solvers · coverage"]
    CC --> PJ[("profile<br/>procedure.json · coverage.json")]
    GLTF --> BP["BlenderProc · seeded<br/>domain randomisation"]
    BP --> DS[("auto-labelled crops per family<br/>+ scenario videos")]
    DS --> COL["Colab T4<br/>family classifiers · hands/tools detector"]
    COL --> ONNX[("ONNX models<br/>+ SHA-256 manifest")]
  end
  subgraph RT["Runtime — HSGPC · RTX 3060 6 GB"]
    FS["IFrameSource<br/>file · webcam · GigE camera"] --> CAP["capture"]
    CAP -- "SPSC ring" --> INF["inference<br/>detector + IP-state classifiers"]
    INF -- "SPSC ring" --> VER["verify<br/>ProcedureEngine · any profile"]
    VER -- "SPSC ring" --> ALR["alert<br/>OPC-UA client"]
    VER -- "SPSC ring" --> IO["io<br/>WebSocket · events · narrator"]
    CAP --> RING[("SharedClipRing<br/>Win32 shared memory · 46 MB")]
  end
  subgraph SC["Local sidecar — Python · same GPU"]
    FL["Florence-2-base<br/>phrase grounding · OCR"]
    CW["clip writer<br/>JPEG ring → MP4"]
  end
  subgraph PL["Platform — Docker · WSL2 · 4 GB cap"]
    TS[("TimescaleDB")]
    GA[("Garage · S3")]
    API["FastAPI<br/>events · health · EWMA drift job"]
    NR["Node-RED<br/>tag monitor"]
  end
  PLC["Soft PLC · open62541 server<br/>:4840 · heartbeat watchdog"]
  AGT["fleet-agent · Go<br/>host health · engine liveness"]
  UI["React dashboard + operator HMI<br/>:5173 · /fleet"]
  PJ --> VER
  ONNX --> INF
  ALR --> PLC
  PLC --> NR
  IO -- "ambiguous IP" --> FL
  PLC -. "evidence · release request" .-> VER
  RING --> FL
  RING --> CW
  CW --> GA
  IO --> API
  API --> TS
  IO --> UI
  API --> UI
  AGT -. "probe :8095" .-> IO
  AGT --> API
  classDef key fill:#E6F1F2,stroke:#0E7C86,stroke-width:1.5px;
  class CC,VER,FL key;
```

*Figure 1 — System architecture. The teal elements are load-bearing: the CAD compiler is the
innovation, the verify thread is the contract, and Florence-2 is the reasoning layer. Note that the
CAD model feeds both the left branch (rules) and the right branch (training data). Colab appears
only at design time. The fleet agent and the PLC's heartbeat watchdog sit outside the engine on
purpose: both exist to notice when the engine has stopped, which the engine cannot do itself. Swap the profile and the same runtime verifies a different
assembly.*

### Why the core is C++, and the reasoning layer is not

The hot path must allocate nothing, lock nothing, and finish in a fraction of a frame. That is a C++
job. Florence-2 is a 230-million-parameter PyTorch model invoked a few times a minute; rewriting its
inference in C++ buys nothing and costs days. The boundary between them is a shared-memory ring and
one HTTP call on localhost, never a blocking call from the verify thread.

### Why the vision-language model is off the critical path

Florence-2 takes 200–400 ms per call on this GPU. The frame budget is 33 ms. It reasons about the
frames the detector is unsure of; it never decides an alert. This is the architectural answer to the
rubric's "CV + LLM" — the integration is real, and it is placed where a 230M model is actually good.

---

## 5. End-to-end verification flow

Five lanes, two loops. The adjudication loop resolves ambiguity; the review loop makes the detector
better.

```mermaid
flowchart TB
  subgraph L1["Operator — any station"]
    A1["Fit parts"] --> A2["Fasten · clip · mate · seat"] --> A3["Request release"]
    A9["Sees HMI: green, or red with corrective path"]
  end
  subgraph L2["Engine — C++ · 30 fps"]
    B1["Hands and tools · IP states"] --> B2{"State change<br/>committed?"}
    B3{"Confidence in<br/>ambiguous band?"}
    B4["Interpreter<br/>order · precedence · evidence · deadlines"] --> B5{"Violation?"}
  end
  subgraph L3["Sidecar — Florence-2"]
    C1["Ground the IP's state in its crop<br/>confirm or reject"]
  end
  subgraph L4["Plant interface"]
    D1["OPC-UA: Andon.Red · Conveyor.Hold"]
    D2["Event + 6 s clip → TimescaleDB · Garage"]
  end
  subgraph L5["Process engineer — end of shift"]
    E1["Dashboard: clip · IP strip · CAD view"] --> E2{"Agree?"}
    E3["False positive → hard negative"]
    E4["Excel shift report"]
  end
  A3 --> B1
  B1 --> B3
  B3 -- "yes" --> C1
  C1 -. "adjudication loop" .-> B2
  B3 -- "no" --> B2
  B2 -- "yes" --> B4
  B5 -- "yes" --> D1
  B5 -- "yes" --> D2
  B5 -- "no" --> A9
  D1 --> A9
  D2 --> E1
  E2 -- "no" --> E3
  E3 -. "review loop" .-> B1
  E2 -- "yes" --> E4
  classDef refer fill:#FBECEB,stroke:#B3261E;
  classDef routine fill:#EAF4EE,stroke:#1B7F4C;
  classDef retake fill:#FBF3E6,stroke:#B26B00;
  classDef key fill:#E6F1F2,stroke:#0E7C86;
  class D1 refer;
  class C1,E3 retake;
  class B4 key;
```

*Figure 2 — End-to-end flow. The amber paths are the two loops most implementations omit. The
adjudication loop is asynchronous: the interpreter never waits for it, and an inspection point in
the ambiguous band is held in its previous committed state until either the detector or Florence-2 resolves it.*

### Decision logic for one inspection-point event

```mermaid
flowchart TD
  S(["Frame arrives"]) --> D1{"IP state differs<br/>from committed state?"}
  D1 -- "no" --> N(["No event"])
  D1 -- "yes" --> D2{"Seen for 2 consecutive frames?"}
  D2 -- "no" --> N
  D2 -- "yes" --> D3{"Detector confidence ≥ 0.65?"}
  D3 -- "no, 0.35–0.65" --> AJ["Hold previous state<br/>enqueue Florence adjudication"] --> N
  D3 -- "yes" --> CM["Commit transition"]
  CM --> D4{"Reaches or leaves<br/>a step's state?"}
  D4 -- "no" --> OK(["Update state, no alert"])
  D4 -- "yes" --> D5{"Order, precedence<br/>and evidence hold?"}
  D5 -- "yes" --> ADV(["Advance the step"])
  D5 -- "no" --> V["SEQUENCE · PRECEDENCE<br/>EVIDENCE_MISSING · REGRESSION"] --> AL(["Alert + event"])
  classDef refer fill:#FBECEB,stroke:#B3261E;
  classDef routine fill:#EAF4EE,stroke:#1B7F4C;
  classDef retake fill:#FBF3E6,stroke:#B26B00;
  class V,AL refer;
  class ADV routine;
  class AJ retake;
```

*Figure 2b — Per-inspection-point decision logic. Two choices are visible. A state change is committed only after
two consecutive frames (debounce), which stops single-frame detector flicker from firing alerts. And an
ambiguous detection never commits — it holds the last known state, so ambiguity can delay an alert by
one adjudication but can never cause a false one.*

Deadlines run separately: each step has a `max_duration_s`, and a step that exceeds it raises
`STEP_TIMEOUT` whatever else is happening. The release gate is the last deadline: when the PLC asks
to let the part go, anything incomplete raises `INCOMPLETE_AT_RELEASE` and the part is held. That is
how a part that is never fitted is caught — absence produces no event, so it must be caught by time
(Section 6).

---

## 6. The procedure model — any assembly, as data

The engine knows nothing about wheels. Wheel fitment is one **profile**: a data file the engine
loads. A door-trim station, a windshield cell and an electronics bench are other profiles, run by the
same binary with no code changed. This section defines the vocabulary every profile is written in.

The problem statement itself demands this. It names three different checks — *"Is the wiring harness
clip for the door panel fastened? Has the torque wrench been applied to all four wheel bolts? Is the
weather-sealing gasket seated correctly before the windshield is installed?"* — and a system that
hard-codes one of them has answered a third of the question.

### Five ideas cover every assembly step

| Idea | What it is | Examples |
|---|---|---|
| **Inspection point (IP)** | A place on the assembly where a verifiable change happens. Position and surface normal come from CAD; it belongs to a state family; its distinguishing feature has a size | a bolt, a snap clip, a connector, a gasket segment, a label, a panel |
| **State family** | An *ordered* list of states. Reaching a later state implies the earlier ones | `threaded_fastener`: absent → seated → fastened |
| **Step** | A constraint over IPs: `complete` (all of them, any order) or `sequence` (an exact order) | "all six clips closed"; "bolts in star order" |
| **Precedence** | `requires`: a step may not make progress until other steps are complete | "gasket seated before the glass goes in" |
| **Deadline and release gate** | Every step has `max_duration_s`; the cycle ends at the PLC's release request, when anything incomplete holds the part | "all bolts torqued before the vehicle moves on" |

Plus one modifier: **evidence**. A step may demand that a visual transition be corroborated by a
plant signal within a time window — the nutrunner's OK, a connector's lock switch (Section 10).

### State families — the library

Each family is learned once and reused by every profile that contains that kind of part.

| Family | States, in order | Distinguishing feature |
|---|---|---|
| `threaded_fastener` | absent → seated → fastened | head height and the witness mark on the head face |
| `snap_clip` | open → closed | the latch tab's position |
| `connector` | unmated → mated | gap between housings; lock lever position |
| `seal_segment` | unseated → seated | lip contact along the flange |
| `panel` | absent → placed → installed | the panel edge against its datum |
| `label` | absent → applied | the label's outline in its target box |
| `presence` | absent → present | the part's silhouette |

**Why order matters — found by testing, not by design.** The first version of the reference
interpreter treated the families as unordered sets. Its first run flagged every bolt that went from
`seated` to `fastened` as a *regression* of the hand-start step. With ordered families, reaching a
later state satisfies earlier steps, and only a move to a *lower* state is a regression — a loosened
bolt, a clip that sprang open.

### Six ways a step can fail

| Violation | Raised when | Example |
|---|---|---|
| `SEQUENCE` | a target reaches the state out of the listed order | bolt 2 torqued when bolt 3 was due |
| `PRECEDENCE` | a step makes progress before a step it requires is complete | trim panel installed over an open clip |
| `EVIDENCE_MISSING` | a transition is not corroborated by its plant signal in time | connector looks mated; lock switch never closed |
| `REGRESSION` | a completed target falls back to a lower state | a torqued bolt loosened: the whole pattern must be redone |
| `STEP_TIMEOUT` | an active step exceeds its deadline | five bolts started, one never torqued |
| `INCOMPLETE_AT_RELEASE` | the PLC asks to release the part with a required step unfinished | the label was never applied |

The last two are how **omissions** are caught. A missing step produces no event — there is nothing to
see — so it can only be caught at a deadline. That is why every required step must have one (I15):
either its own `max_duration_s`, or at the latest the release gate. Wrong *order* is caught at the
instant it happens; a wrong *absence* is caught when time runs out. The problem statement's
"before the vehicle moves to the next station" is literally the release gate.

### A profile, written out

The wheel profile's manifest — the human-authored half — is ten lines of intent:

```yaml
# profiles/wheel-fitment-rl/manifest.yaml
profile: wheel-fitment-rl
groups:
  - id: bolt
    family: threaded_fastener
    part_number: BLT-M12-1.25-45        # joins this group to its STEP occurrences
    feature_mm: 17                      # M12 head face, which carries the witness mark
steps:
  - { id: hand_start, group: bolt, kind: complete, state: seated, max_duration_s: 20 }
  - { id: torque, group: bolt, kind: sequence, order: derive, state: fastened,
      requires: [hand_start], max_duration_s: 16,
      evidence: { tag: Station.W14.Nutrunner.ResultOK, equals: true, within_ms: 1500 } }
```

The door-trim profile, for the problem statement's harness-clip question, uses the same vocabulary
for a different assembly:

```yaml
# profiles/door-trim-harness-fl/manifest.yaml
profile: door-trim-harness-fl
groups:
  - { id: clip,      family: snap_clip, part_number: CLP-HRN-08,  feature_mm: 4 }
  - { id: connector, family: connector, part_number: CON-DR-12P,  feature_mm: 6 }
  - { id: panel,     family: panel,     part_number: TRM-DR-FL,   feature_mm: 60 }
steps:
  - { id: clips, group: clip, kind: complete, state: closed, max_duration_s: 40 }
  - { id: connector, group: connector, kind: complete, state: mated, max_duration_s: 20,
      evidence: { tag: Station.D07.ConnectorLock, equals: true, within_ms: 1000 } }
  - { id: trim_panel, group: panel, kind: complete, state: installed,
      requires: [clips, connector], max_duration_s: 30 }
```

The compiler (Section 7) joins each manifest to its STEP file, places every inspection point,
derives `order: derive` sequences from geometry, checks that a camera can resolve each point, and
writes `procedure.json`. The engine reads only `procedure.json`.

### The executable specification

The interpreter was written first as a small Python reference model, `reference/tsref/procedure.py`
— about 150 lines with no knowledge of any assembly — and tested against four profiles before a word
of this section was written:

| Profile | Source | Scenarios tested |
|---|---|---|
| `wheel-fitment-rl` | PS4 reference process | correct star; adjacent order; missing bolt (timeout and release); torqued-looking with no nutrunner OK; loosened bolt; torquing before every bolt is started |
| `door-trim-harness-fl` | PS4: "harness clip … fastened?" | correct; panel installed over an open clip; connector without its lock signal |
| `windshield-bond` | PS4: "gasket seated … before the windshield" | correct; glass placed with one segment unseated |
| `enclosure-cover` | non-automotive electronics | correct crisscross then label; label forgotten |

Twenty-nine tests in all (five of them for the camera arithmetic in Section 8), including pattern solvers, optics and load-time rejection of a state that
is not in a part's family. All pass. Every violation kind in the table above is produced by at least
one scenario.

The C++ `ProcedureEngine` is the production implementation of the same model, and it is
**differentially tested** against the reference (I16): every scenario's event stream is fed to both,
and their violation lists must be identical. The reference is the specification; the C++ is the
fast copy. When they disagree, the reference wins until a test proves it wrong.

**Keeping it allocation-free.** A profile is compiled into flat arrays at load time: a rank table per
inspection point, one 64-bit mask per step for its targets, a fixed-size pending-evidence ring. A
committed transition touches only the steps that target that one point. Limits that follow: 256
inspection points per profile and 64 targets per step — far beyond one station's work content.
Invariant I1 holds for every profile, not just the wheel.

---

## 7. The CAD compiler and pattern solvers

Two inputs, one output. **Geometry** comes from the STEP file, because geometry is the one thing every
CAD system exports reliably. **Intent** — which parts are inspection points, which family, which
constraints — comes from the manifest, because custom attributes do not survive every CAD export and
should be reviewed like code anyway. The compiler joins them by part number.

```mermaid
flowchart LR
  STEP[/"assembly.step"/] --> R1["STEP reader<br/>OCCT XCAF"]
  MAN[/"manifest.yaml"/] --> J
  R1 --> R2["DFS over the assembly tree<br/>every occurrence · its transform"]
  R2 --> J["join by part number<br/>→ inspection points"]
  J --> P["pattern solvers<br/>circle · line · field"]
  J --> O["optics check<br/>resolution · frame · facing"]
  J --> X["occlusion ray-cast<br/>OCCT"]
  P --> PJ[("procedure.json")]
  O --> CR[("coverage.json")]
  X --> CR
  CR --> G{"every IP resolved?"}
  G -- "no" --> F["refuse to deploy<br/>name the points and the shortfall"]
  G -- "yes" --> OK(["profile ready for rendering"])
```

*Figure 3 — The compiler. It refuses as well as produces: a profile with an inspection point no camera
can resolve never reaches a station (I17).*

### Pattern solvers — the order is derived, not typed

When a manifest says `order: derive`, the compiler classifies the fasteners' layout and picks the
solver:

| Layout | Detected when | Solver | Example result |
|---|---|---|---|
| **Circle** | all points equidistant from the centroid | star: odd N strides ⌊N/2⌋; even N fastens diametric pairs | 5-bolt hub: 1, 3, 5, 2, 4 |
| **Line** | points collinear | centre-out, alternating sides | 7-screw seam: 4, 5, 3, 6, 2, 7, 1 |
| **Field** | anything else | centre-out; each next point the one most nearly opposite the last | 2×3 cover: 2, 5, 3, 4, 6, 1 |

Results on a circle:

| N | Sequence | Note |
|---|---|---|
| 4 | 1, 3, 2, 4 | opposite pairs |
| 5 | 1, 3, 5, 2, 4 | single cycle, stride 2 |
| 6 | 1, 4, 2, 5, 3, 6 | opposite pairs, rotating |
| 8 | 1, 5, 3, 7, 2, 6, 4, 8 | cross pattern |

An explicit `order: [ … ]` in the manifest overrides derivation when the OEM's work instruction
mandates something else; `procedure.json` records which it was, `derived` or `specified`.

```cpp
// cad-extract/src/StarPatternSolver.cpp
// Returns 1-based IDs in fastening order. `by_angle` is sorted by angle about the centroid,
// rotated to start at member 1, and oriented so it runs in the direction of the numbering.
std::vector<int> StarPatternSolver::solve(std::span<const Bolt> by_angle) {
  const int n = static_cast<int>(by_angle.size());
  std::vector<int> order;
  order.reserve(n);

  if (n % 2 == 1) {
    // Odd N: stride floor(N/2) is always coprime with N, so one cycle visits every bolt.
    for (int k = 0, i = 0; k < n; ++k, i = (i + n / 2) % n)
      order.push_back(by_angle[i].id);
    return order;
  }

  // Even N: fasten diametrically opposite pairs (j, j + N/2). Choose the order of the
  // pair starts by striding N/4 through the half-circle, falling back to the first
  // unvisited start when the stride cycles early (N = 8 would otherwise loop 0,2,0,2).
  const int half = n / 2;
  const int step = std::max(1, n / 4);
  std::vector<bool> seen(half, false);
  for (int s = 0, placed = 0; placed < half; ++placed) {
    if (seen[s]) s = static_cast<int>(std::find(seen.begin(), seen.end(), false) - seen.begin());
    seen[s] = true;
    order.push_back(by_angle[s].id);
    order.push_back(by_angle[s + half].id);
    s = (s + step) % half;
  }
  return order;
}
```

Each solver is tested against properties, not hard-coded answers: every fastener appears exactly
once; on a circle each consecutive pair obeys the spacing rule; on a line the distance from the
centre never decreases and sides alternate; every solver returns a permutation on 200 random
layouts. Three bugs were caught this way across revisions: the star solver looped forever on N = 8;
it started at whichever bolt happened to sit nearest the x-axis, giving 2, 5, 3, 1, 4 — a valid star,
but not the documented one; and the field solver paired corners across the short axis instead of
the diagonal until "most nearly opposite" replaced "on the other side".

### Can a camera see it? — the resolution check

For every inspection point and every camera, the compiler computes the projected size of the point's
distinguishing feature with a pinhole model:

$$f_{px} = \frac{W/2}{\tan(\mathrm{HFOV}/2)}, \qquad \text{feature}_{px} = f_{px}\cdot\frac{\text{feature}_{mm}}{d_{mm}}\cdot\cos\theta$$

where $W$ is the width of the frame the crop is taken from, $d$ the distance and $\theta$ the angle
between the feature's normal and the line of sight. A point counts as resolved only if it is inside
the frame, faces the camera (θ < 70°) and spans at least **16 px** — a starting threshold that each
profile's real-image validation confirms or raises (Section 11).

These figures are from the compiler's reference implementation, run on the four profiles:

| Rig | Smallest feature | Unresolved |
|---|---|---|
| Wheel, crops from the 960 px downscale | 12.8 px | 5 of 5 |
| **Wheel, crops from the 1,920 px frame** | **25.6 px** | **0** |
| Door, one camera at 1.2 m | 5.8 px | 7 of 8 |
| Door, three cameras at 500 mm | 15.1 px | 4 of 8 |
| **Door, three cameras at 420 mm** | **17.4 px** | **0** |
| **Windshield, four cameras at 800 mm** | **16.6 px** | **0** |
| Enclosure, one camera at 400 mm | 15.9 px | 4 of 8 |
| **Enclosure, one camera at 350 mm** | **17.8 px** | **0** |

The first row is a correction to v4.1, which processed every frame at 960×540. At that resolution the
wheel's bolt-head face spans under 13 px: too few to trust a state judgement. The detector still runs
on the downscale; inspection-point crops now come from a small pool of full-resolution frames
(Section 9). Nothing in v4.1 checked this. The compiler now checks it for every point of every profile.

**Occlusion** is a separate test, run on the CAD itself: a ray from each camera's centre to each
inspection point, intersected with the assembly's other bodies using OCCT's
`IntCurvesFace_ShapeIntersector`. If the first surface hit is not the point's own part, that camera
cannot see it. The reference model does not include this test, and it has not been run yet; it is the
C++ compiler's job.

The compiler writes `coverage.json` — every point, its best camera, its feature size, its occlusion
result — and refuses to emit a deployable profile if any point is unresolved (I17).

---

## 8. Camera capture

Everything downstream of the capture thread is the same whatever the camera. Frames enter through
one interface, `IFrameSource`, and three implementations cover the whole life of the system.

| Source | Used for | Library | Built in |
|---|---|---|---|
| `FileSource` | rendered scenarios: the demo, the harness, the latency gate | OpenCV + FFmpeg | every build |
| `UvcSource` | development with a USB webcam on the laptop | OpenCV, DirectShow backend | Windows builds |
| `PylonSource` | production: industrial GigE Vision cameras | Basler pylon C++ SDK | production builds, `SUNVERIFY_WITH_PYLON=ON` |

A fourth, `ReplayDetector`, skips frames altogether and feeds recorded detections; it is what the
sanitizer builds and the kind cluster run. The software-only build needs no camera at all.

### The production camera

The problem statement names Basler and FLIR. The pick is the **Basler ace 2 Basic a2A1920-51gcBAS**.

| Property | Value | Why it matters here |
|---|---|---|
| Sensor | Sony IMX392, **global shutter** | Hands and tools move during exposure. A rolling-shutter webcam skews them; a global shutter does not |
| Resolution | 1920 × 1200, 3.45 µm pixels | Enough to resolve every reference profile with a stock lens (below) |
| Frame rate | up to 51 fps | 30 fps runs with headroom |
| Interface | GigE, **PoE** or 12–24 VDC | One cable carries power and data up to 100 m — the standard on a shop floor |
| Lens mount | C-mount | The lens is chosen per station, by the compiler |
| I/O | 1 opto-isolated input, 2 GPIO, on an M8 6-pin connector | Strobe output for lighting; trigger input if ever needed |
| Operating temperature | −10 to 60 °C | A line-side enclosure, not an office |

The Pro model has the same sensor and frame rate; its extra features are not used by this workload,
so the Basic is the right buy. The SDK is pylon: a C++ API on Windows, Linux x86-64 and Linux ARM,
with GigE Vision and USB3 Vision drivers. It is proprietary and downloaded from Basler — read its
licence before shipping it in an image. `IFrameSource` keeps the engine free of it everywhere else.

### Which lens? — the compiler picks it

A stock focal length must be **short enough** to cover the field the station needs and **long
enough** to resolve the smallest feature at 16 px. The compiler computes both bounds and takes the
longest stock lens inside the window — the most pixels per millimetre that still sees everything.
From the reference implementation, `reference/tsref/capture.py`, for this camera:

| Station | Distance | Field needed | Smallest feature | Lens | Feature in pixels | Field covered |
|---|---|---|---|---|---|---|
| Wheel | 900 mm | 500 mm | 17 mm bolt face | **8 mm** | 43.8 px | 745 mm |
| Door trim, per camera | 420 mm | 400 mm | 4 mm clip latch | **6 mm** | 16.6 px | 464 mm |
| Windshield, per camera | 800 mm | 700 mm | 10 mm gasket lip | **6 mm** | 21.7 px | 883 mm |
| Enclosure | 350 mm | 260 mm | 5 mm screw head | **8 mm** | 33.1 px | 290 mm |

The door's window is narrow — 5.8 to 7.0 mm, and only the 6 mm lens fits. When no stock lens fits,
the function returns nothing and the compiler says why: one camera covering the whole door from
1.2 m needs a lens shorter than 7.9 mm for the field and longer than 16.6 mm for the latch. That rig
cannot work, and it is better to learn so from a test than from a station.

### Will the link carry it? — Bayer on the wire

| Pixel format on the wire | Bandwidth at 30 fps | Fits one GigE link? |
|---|---|---|
| **BayerRG8** | **581 Mbit/s** | **yes** |
| YCbCr422_8 | 1,161 Mbit/s | no |
| RGB8 | 1,742 Mbit/s | no |

So the camera sends raw Bayer data and the host demosaics it, once, on the capture thread. And two
cameras cannot share a gigabit port at this rate: each camera gets its own port, from a **4-port
PoE+ GigE network card** in the edge PC — no switch between camera and computer, one less device to
fail. The camera ports run jumbo frames (MTU 9000) with the camera's packet size set to the largest
the card carries; on Linux the socket receive buffers are raised so a burst never overflows the
kernel. These are node settings, applied by provisioning (Section 32).

### Zero-copy into the frame pool

Invariant I1 — no allocation per frame — has to hold from the camera inward. pylon lets an
application supply its own image buffers through `IBufferFactory`, installed with
`CInstantCamera::SetBufferFactory`; buffers are requested when grabbing starts. So the driver writes
every frame straight into the engine's pre-allocated full-resolution pool:

```cpp
// control-app/src/sources/PylonSource.cpp (production builds only)
class PoolBufferFactory final : public Pylon::IBufferFactory {
 public:
  explicit PoolBufferFactory(RawFramePool& pool) : pool_(pool) {}
  void AllocateBuffer(size_t bytes, void** buffer, intptr_t& context) override {
    auto slot = pool_.lendToDriver(bytes);         // pre-allocated at start-up; never grows
    *buffer = slot.data;
    context = slot.index;
  }
  void FreeBuffer(void* /*buffer*/, intptr_t context) override { pool_.returnFromDriver(context); }
  void DestroyBufferFactory() override {}          // owned by PylonSource, not by pylon
 private:
  RawFramePool& pool_;
};

// start-up, before grabbing — the factory must outlive every buffer it lent
camera_.SetBufferFactory(&factory_, Pylon::Cleanup_None);
camera_.MaxNumBuffer = 8;                                    // 8 raw Bayer frames: 8 × 2.3 MB
camera_.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly);  // fall behind → drop, never queue
```

`GrabStrategy_LatestImageOnly` matches the rest of the pipeline: when the engine falls behind it
drops old frames rather than queueing them, and every drop is counted in `dropped_frame_rate`. In
production the full-resolution crop pool *is* this raw pool — 8 Bayer frames, about 18 MB — and
crops are cut from it at CAD-projected positions (Section 9).

### Settings are configuration, not defaults

Exposure, gain and white balance are fixed at commissioning, stored in the profile's `cameras.yaml`,
and saved into a camera user set that loads at power-up. When `PylonSource` starts, it reads every
setting back from the camera and compares it with `cameras.yaml`. A mismatch — a camera swapped,
reset or reconfigured — refuses to verify: the station shows AMBER and says why. A camera that
quietly lost its exposure setting after a power cut is the kind of fault that otherwise looks like
a model problem for a week.

**Exposure.** A hand moving at 0.5 m/s travels 1 mm in 2 ms. At the wheel's 0.39 mm per pixel that
is about 2.6 px of smear on moving hands, and none on parts that have stopped — which are the parts
being judged. So exposure is 2 ms or less, with **strobed diffuse LED bar lighting** driven from the
camera's GPIO by its exposure-active signal. The station's images then depend on its own light, not
on the plant's.

### Time and synchronisation

Each camera free-runs at 30 fps. Every frame is stamped with the host's monotonic clock when the
grab completes; the camera's own tick count is kept with it for diagnostics. Multi-camera stations
need no hardware synchronisation: steps are seconds apart, and a camera-to-camera skew of one frame
period cannot reorder them. If a profile ever needs two events on different cameras ordered within
100 ms, the escalation is a hardware trigger — one camera's exposure-active output wired to the
others' opto inputs — not a software fix.

### When a camera fails

A device-removal callback marks the source lost and stops it publishing frames. The verify thread's
frame counter stops, so the heartbeat stops, so the PLC turns the station AMBER within 2 s — no
camera-specific alarm path is needed (Section 18). A reconnect loop retries every 2 s and re-runs
the settings check before the first frame is trusted.

Every bucket also records the camera's device temperature and its packet-resend count. These are
checked against fixed limits from the camera's specification rather than charted with EWMA, so the
false-alarm budget for the six drift charts is unchanged.

### Calibration

Intrinsics come from a ChArUco board and OpenCV's `CharucoDetector` and `calibrateCamera`:
`scripts/calibrate.py --camera cam-1` writes them into the profile's `cameras.yaml`, and a
reprojection error above 0.5 px RMS — a starting gate — sends the technician back for more board
poses. The extrinsics are either fixed at commissioning or recovered every frame from fiducials
(Section 9).

### Development: a USB webcam

`UvcSource` opens any UVC webcam through OpenCV's DirectShow backend, requests MJPG at
1920 × 1080 and 30 fps, and turns autofocus and auto-exposure off, with the chosen values written to
`cameras.yaml` exactly as for a production camera. It is a development tool: webcams are
rolling-shutter, autofocus drifts, and the resolution check runs on its numbers like any other
camera. It never ships to a station.

### Commissioning checklist — one station

| Step | Done when |
|---|---|
| Lens from `lens_for` mounted, focus locked | the compiler's chosen focal length is on the camera |
| Camera on its own NIC port, jumbo frames on | `pylon Viewer` or the engine reports zero resends over 10 minutes |
| Exposure ≤ 2 ms, strobe wired, settings saved to the user set | settings read-back matches `cameras.yaml` after a power cycle |
| Intrinsics calibrated | reprojection error ≤ 0.5 px RMS |
| Extrinsics fixed or fiducials visible | anchor residual stable for one shift |
| Coverage re-run with the installed camera's numbers | `coverage.json` clean |

---

## 9. Perception for any part

Two stages, split by what generalises.

| Stage | Question | Model | Input | Trained |
|---|---|---|---|---|
| **Scene** | Where are the hands and tools? | YOLO11n, ONNX Runtime CUDA EP | 960×540 downscale | once, shared by every profile |
| **Inspection point** | Which of its two or three states is this point in? | one small classifier per state family (MobileNetV3-Small, 96×96 crop) | crops at CAD-projected positions, from the full-resolution frame | once per family, reused by every profile with that family |

The inspection-point stage is what makes the system universal. At an inspection point the question
is never *"what is this object?"* — an open problem — but *"which of these two or three known states
is it in?"* That is a closed, small classification problem, and it can be learned from renders of
one part family and reused wherever that family appears. A snap-clip classifier trained for the door
also serves a dashboard harness, a seat-wiring loom and a battery-tray cover.

All the crops visible in a frame are classified in one batch. Committing, debouncing and adjudication
are unchanged from the wheel design (Figure 2b): two consecutive frames, a 0.65 confidence floor, and
an ambiguous band that holds the last state and asks Florence-2.

### Where are the inspection points in the image? — registration

| Station | Registration | How |
|---|---|---|
| Fixed fixture, assembly always in the same place | fixed | camera extrinsics from calibration, once |
| Moving carrier, pallet or conveyor | fiducials | AprilTag markers on the fixture, located every frame with OpenCV's ArUco module, pose by `solvePnP` |

Either way, every inspection point's pixel position is the CAD position projected through the current
pose. The mean distance between where a part is detected and where it should be — the anchor residual
— is the fleet-health signal that catches a knocked camera or a mis-seated fixture (Section 18).

### Training data for a new family

The same pipeline as the wheel: the part's CAD, rendered by BlenderProc in every state of its family
with domain randomisation, labels written automatically. A new profile built only from known families
needs **no new training at all** — only its own scenario fixtures for validation. A new family costs
one render run and one classifier training run on Colab.

### Cold start: Florence-2, advisory only

Before a new family's classifier exists, Florence-2 can ground a plain description — *"closed plastic
clip"* — in each crop, zero-shot. That is useful for trying a profile out and for seeding labels. It
is never allowed to raise an alert: a zero-shot model has not passed a validation gate, so in cold
start the station shows its verdicts as advisory and the PLC is not written to. The same rule as
everywhere else in this design: anything that has not been validated may inform, never decide.

### Memory: a full-resolution pool for crops

The detector's frame pool stays at 12 frames of 960×540 (18.7 MB). Crops need the full-resolution
frame for one frame period, so a second pool holds 4 full-resolution frames (4 × 6.2 MB = 25 MB),
leased and returned exactly like the first. Both pools are allocated at start-up; I1 holds. On a
production camera the second pool is the camera driver's own buffer set — eight raw Bayer frames
written in place by the driver (Section 8).

### What this stage cannot do

It classifies states at points. It does not trace a cable's route between clips, judge cosmetic
damage, or measure a gap to tenths of a millimetre. Route verification needs segmentation along a
path, and metrology needs calibrated optics. Both are different problems, and this design says so
rather than stretch.

---

## 10. Evidence beyond the camera

Some facts no camera can establish. A universal system has to accept evidence from the plant as well
as from pixels.

| Fact | Why vision cannot establish it | Evidence source |
|---|---|---|
| A fastener reached its torque | clamp load is invisible | nutrunner controller: per-rundown OK, torque and angle over OPC-UA |
| A connector's latch engaged inside its housing | the latch is hidden | connector-position-assurance switch, or a continuity test |
| The right part variant was fitted | many variants look alike | barcode or RFID scan, compared with the vehicle's bill of materials |
| A press-fit reached its force | force is invisible | press controller's force-window result |
| A fluid was filled | volume is hidden | fill machine's result |

A profile uses this in two ways.

**Corroboration.** A visual transition must be matched by a plant signal within a time window: the
wheel bolt looks torqued *and* the nutrunner reported OK within 1.5 s. Either alone is weaker. A
witness mark can be painted on a bolt that was never torqued; a nutrunner OK can belong to a different
bolt. Together, one says *which* bolt and the other says *it reached torque*. The reference tests
include both failures: five bolts that look torqued with no nutrunner result raise five
`EVIDENCE_MISSING`; a connector that looks mated with no lock signal raises one.

**Signal-only steps.** An inspection point can be of family `signal`, with no camera at all: its state
comes from a tag mapping — `press.ForceOK = true → done`. The interpreter does not care where a state
came from, so precedence, deadlines and the release gate apply to it exactly as to a visual point.
That is how a station whose critical step is invisible still gets omission detection.

Plant signals reach the interpreter as ordinary events. The alert thread's OPC-UA client subscribes
to the profile's evidence and release tags; each change is timestamped on receipt and handed to the
verify thread over its own SPSC ring. The verify thread still never touches the network.

### Mixed-model lines — the variant decides the profile

Tata lines build several variants on one line. At cycle start the engine reads the variant code the
PLC publishes for the arriving vehicle and selects that variant's compiled procedure. Every variant
for a station is compiled and loaded at start-up, so switching is an index change, not a file load:
no allocation, no latency. A variant code with no compiled profile holds the station AMBER instead of
guessing.

---

## 11. Onboarding a new assembly

A new station is a new profile directory, not a new program.

```mermaid
flowchart LR
  A["1 · Author manifest<br/>groups · steps · evidence"] --> B["2 · Compile<br/>orders · coverage"]
  B --> C{"coverage clean?"}
  C -- "no" --> R["move or add cameras<br/>change lens or distance"] --> B
  C -- "yes" --> D["3 · Render<br/>scenario fixtures · new families only"]
  D --> E["4 · Train<br/>only families not yet in the library"]
  E --> F["5 · Validate<br/>scenarios · real photos · gate"]
  F --> G{"gate passed?"}
  G -- "no" --> D
  G -- "yes" --> H["6 · Canary<br/>one station, one shift, vs peers"]
  H --> I(["Live"])
  classDef gate fill:#FBF3E6,stroke:#B26B00;
  class C,G gate;
```

*Figure 3b — Onboarding. Two gates, both before anything reaches the line: one on geometry, one on
evidence.*

### What it costs — estimates, to be replaced by measurements

| Work | Known families only | With one new family |
|---|---|---|
| Manifest: groups, steps, evidence tags | 2–4 h | 2–4 h |
| Compile and fix coverage | under 1 h, plus any camera moves | the same |
| Render scenario fixtures | about 30 min of machine time | plus the family's training renders |
| Train | nothing | one Colab run, 2–3 h |
| Real validation set: photograph and label | about 2 h | about 3 h |
| Gate and canary | one shift | one shift |
| **Engineer time** | **about one day** | **two to three days** |

These are planning estimates. The first real onboarding should time every row and replace them.

### The profile directory

```
profiles/door-trim-harness-fl/
├── manifest.yaml        authored and reviewed in a pull request
├── assembly.step        from engineering; its revision is the profile's CAD revision
├── cameras.yaml         intrinsics, poses, registration method
├── procedure.json       compiled — never edited by hand
├── coverage.json        compiled — resolution, framing, occlusion for every point
├── narratives.yaml      one sentence template per violation kind and step
├── playbook.yaml        fleet-health patterns specific to this station, if any
├── scenarios/           fixtures with .expected.json, rendered from the same CAD
└── VALIDATION.md        gate results, signed off before going live
```

The whole directory is versioned as one unit; its hash is the `profile_version` stamped on every
anomaly row and health bucket, next to `cad_revision` and `model_version`.

### The gate (I14)

A profile goes live only when:

1. `coverage.json` has no unresolved inspection point;
2. every scenario fixture produces exactly its expected verdict, in both the reference interpreter
   and the C++ engine;
3. each family's per-state recall on that station's real photographs is reported with a confidence
   interval, and the recall on the *bad* states — open clip, unseated gasket — meets the plant's
   threshold;
4. a canary shift on one station shows no more false alarms than its peers.

### Is a station a candidate at all? — the applicability check

Before promising a station, answer five questions. Any "no" means vision verification is the wrong
tool for that step, not that the step cannot be verified.

| Question | If "no" |
|---|---|
| Does every step to be verified change something visible at a point, or produce a plant signal? | a signal-only step, or out of scope |
| Is each visible change at least 16 px at a camera position you can actually mount? | different lens, closer camera, a second camera |
| Is each point unoccluded by the operator for at least part of its step? | a second viewpoint, or corroborating evidence |
| Does the cycle have a release signal the engine can read? | add one to the PLC; without it omissions are caught only by timeouts |
| Can the assembly's pose be known — a fixed fixture or fiducials? | add fiducials to the fixture |

Out of scope, whatever the answers: cable routing between clips, cosmetic inspection, dimensional
metrology, and anything whose only evidence is inside a closed part with no sensor.

---

## 12. Latency and capacity model

### Define the clock before quoting the number

Two latencies, both measured with HdrHistogram over 100 runs, reported at p50/p99/p99.9:

| Latency | From | To | What it means |
|---|---|---|---|
| **Engine latency** | frame in which the transition **commits** | OPC-UA write acknowledged | how fast the software reacts |
| **Evidence latency** | **first** frame showing the torqued bolt | OPC-UA write acknowledged | what the operator experiences |

Evidence latency includes the debounce frame. Quote both, and say which is which. A judge who has
measured anything will ask.

### Budget at 30 fps, single station

| Stage | Expected |
|---|---|
| Frame source (decode from file) | 2 ms |
| Letterbox + normalise to 640² | 2 ms |
| YOLO11n, ONNX Runtime CUDA EP, FP32 | 14 ms |
| NMS + ByteTrack + CAD identity + state vector | 3 ms |
| IP crops from the full-resolution frame, family classifiers batched | ≈ 4 ms, estimate |
| Interpreter evaluate | < 0.1 ms |
| SPSC hop to alert thread | < 0.1 ms |
| OPC-UA write to localhost soft PLC | 3 ms |
| **Engine latency, p99** | **≈ 35 ms** |
| Debounce (one additional frame) | 33 ms |
| **Evidence latency, p99** | **≈ 70 ms** |

The crop stage is the one stage that grows with the number of inspection points in view, and its
figure is an estimate until `latency_test` measures it.

**Invariant I4:** evidence latency p99 ≤ 150 ms at 30 fps single-station. Measured with GPU clocks
locked, on AC power. Against the problem statement's one second, that is a fourteen-fold margin.
Against a real PLC, add roughly 15 ms for the fieldbus — say so.

### VRAM — single station, demo mode

The Intel UHD drives the display, so the RTX 3060's 6 GB is almost entirely available for compute.

```
RTX 3060 Laptop — 6 GB
  torquesense.exe  CUDA context                 ~0.35 GB
                   ORT CUDA arena (capped)       1.00 GB
  sidecar.py       CUDA context + torch          ~0.45 GB
                   Florence-2-base fp16 weights  ~0.46 GB
                   activations, capped           ~0.60 GB
  ───────────────────────────────────────────────
  Total                                          ~2.9 GB  ✓  3 GB headroom
```

### Four-station mode — honest figures

Four `torquesense.exe` processes, one shared sidecar. Each process carries its own CUDA context.

| | Per station | ×4 |
|---|---|---|
| CUDA context | 0.35 GB | 1.40 GB |
| ORT arena, capped | 0.40 GB | 1.60 GB |
| Sidecar (shared) | — | 1.50 GB |
| **Total** | | **~4.5 GB / 6 GB** ✓ |

**Frame rate drops to 15 fps per station.** Four processes time-slice one GPU; 60 inferences per
second aggregate is comfortable, 120 is not. Fastening a bolt takes seconds, so 15 fps loses nothing
for verification. Evidence latency rises to roughly 180 ms — still six times inside the one-second
requirement, but outside the 150 ms single-station invariant. **I4 is defined for the single-station
hero mode; multi-station mode reports its own measured figure against the problem statement's
1 second.** If GPU utilisation exceeds 80% in `nvidia-smi`, drop to three stations rather than
lowering the frame rate further.

Earlier drafts claimed four stations at 30 fps under 40% GPU. That was not measured and is
withdrawn (Appendix A).

### Adjudication capacity

Florence-2 serves one request at a time at 200–400 ms, so ~3 per second. A bolt enters the
ambiguous band perhaps once per cycle under good rendering, and cycles are 58 seconds long. Four
stations generate well under one request per second. The bounded queue (depth 16) drops the oldest
request on overflow; a dropped adjudication delays a state commit, it never produces an alert.

---

# Part III — Platform

## 13. Process and thread model

Six kinds of process on the host, four containers in Docker — five in CI mode. Every boundary exists for a reason that can
be stated in one sentence.

| Process | Language | Port | Why it is separate |
|---|---|---|---|
| `torquesense.exe` (one per station) | C++20 | 8080 WS · 8090 MJPEG · 8095 control | The hot path. Nothing else may share its address space. |
| `softplc.exe` | C++ · open62541 server | 4840 | Stands in for the line PLC. Separate so the alert path crosses a real network boundary. |
| `sidecar` | Python · PyTorch | 8001 | Florence-2 and the clip writer. A stall here must never stall the engine. |
| `fleet-agent.exe` (one per station) | Go | none — push only | Reports host health and whether the engine is alive. It cannot live in the process it watches, and it listens on no port. |
| Vite dev server | Node | 5173 | Dashboard and operator HMI. |
| Docker Desktop (WSL2) | — | 5432 · 3900 · 8000 · 1880 · 8081 | TimescaleDB, Garage, FastAPI, Node-RED; Jenkins on 8081 in CI mode only. |

### Threads inside `torquesense.exe`

```mermaid
flowchart LR
  subgraph P["Pinned — physical cores 4 to 7 · HIGH_PRIORITY_CLASS"]
    C["capture<br/>logical CPU 8"] -- "SPSC 32" --> I["inference<br/>logical CPU 10"]
    I -- "SPSC 32" --> V["verify<br/>logical CPU 12"]
    V -- "SPSC 16" --> A["alert<br/>logical CPU 14<br/>OPC-UA write · heartbeat"]
  end
  subgraph U["Unpinned — normal priority"]
    O["io<br/>WebSocket · HTTP · narrator · logging"]
    K["control<br/>scenario switch · health"]
  end
  V -- "SPSC 64" --> O
  A -- "SPSC 16 · ack + latency" --> O
  C -- "JPEG" --> R[("SharedClipRing")]
  A --> PLC["OPC-UA :4840"]
  classDef key fill:#E6F1F2,stroke:#0E7C86;
  class V,A key;
```

*Figure 4 — Thread model. The alert thread exists for one reason: the OPC-UA write is blocking I/O,
and blocking I/O must never run on the verify thread. Giving the alert its own pinned thread keeps the
verify thread allocation-free and non-blocking while keeping the alert path short. Everything slow —
string formatting, JSON, HTTP, logging — lives on the unpinned io thread.*

### The heartbeat — liveness tied to work, not to a thread

Every 500 ms the alert thread increments `Station.W14.Heartbeat` — but only if the verify thread's
frame counter has moved since the last beat. A frozen camera, a hung inference thread and a crashed
process all look the same to the PLC: the counter stops. The PLC turns that into AMBER (Section 18).

```cpp
// control-app/src/AlertThread.cpp — the beat is tied to frames actually verified
if (now >= next_beat_) {
  const auto verified = frames_verified_.load(std::memory_order_relaxed);  // a counter: relaxed is enough
  if (verified != last_verified_) {
    last_verified_ = verified;
    plc_.writeUInt32(heartbeat_node_, ++beat_);  // blocking I/O: one reason this is its own thread
  }
  next_beat_ = now + std::chrono::milliseconds{500};
}
```

The alert thread also returns each acknowledgement and its measured latency to the io thread, over
its own SPSC ring. That is how an anomaly row learns its `engine_ms` and `evidence_ms`.

The i7-11800H exposes physical cores as logical CPU pairs (0,1), (2,3) … (14,15). Pinning to the
even logical CPUs 8, 10, 12, 14 puts each pipeline thread on its own physical core and leaves the
hyperthread siblings idle, so no pipeline thread shares execution resources with the OS or the
browser. `SetThreadAffinityMask` returns 0 on failure; check it, because a silently failed pin makes
every latency figure meaningless.

### Windows-specific scheduling — three things a Linux-trained engineer misses

**Timer resolution.** `Sleep` and `std::this_thread::sleep_for` on Windows wake on the system timer,
historically 15.6 ms. Replaying video at 30 fps with that timer produces uneven frame pacing that
looks like a detector problem. The capture thread paces itself with a high-resolution waitable timer:

```cpp
// control-app/src/Affinity.cpp — one of the two files allowed platform #ifdefs
#ifdef _WIN32
HANDLE make_frame_timer() {
  return CreateWaitableTimerExW(nullptr, nullptr,
                                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
}
#endif
```

**Power throttling.** Windows 11 applies EcoQoS to processes it considers background, lowering clock
speed on a laptop. Opt the engine out explicitly at startup:

```cpp
PROCESS_POWER_THROTTLING_STATE s{};
s.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
s.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
s.StateMask   = 0;   // 0 = throttling OFF for the controlled policy
SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &s, sizeof(s));
SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);   // never REALTIME_PRIORITY_CLASS
```

**`REALTIME_PRIORITY_CLASS` is forbidden.** It can preempt the input stack and the disk cache
manager, and a demo laptop that stops responding to the mouse is worse than any latency figure.

---

## 14. Identity and access

**Cut for the prototype, specified for production.** Applying the scope test from Section 1: login
earns no rubric points on a single-laptop demo and costs a day.

What the prototype does instead:

| Measure | Implementation |
|---|---|
| Nothing listens on the network that does not need to | Every service binds `127.0.0.1` except the soft PLC, which binds all interfaces so the Node-RED container can reach it |
| Write endpoints are not open | FastAPI `POST` routes require an `X-API-Key` header read from `.env` |
| The fault harness cannot be mistaken for production | Separate route, styled as a test tool, and the control port refuses requests not from localhost |

What production needs — say this in the pitch, do not build it: OIDC for the engineer dashboard
(plant SSO), no login on the operator HMI (it is a kiosk at a station, identified by station ID, not
by person), and OPC-UA security mode `SignAndEncrypt` with certificates between the engine and the
PLC. Operator identity, where needed, comes from the plant's existing badge system via MES — the
camera system should not become a second identity authority on the shop floor.

---

## 15. Data architecture

Pixels never enter the relational store. Events are rows; clips are objects; the row holds the
object key.

### TimescaleDB schema

```sql
-- platform/migrations/001_schema.sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE anomalies (
  ts              TIMESTAMPTZ NOT NULL,
  station_id      TEXT        NOT NULL,
  cycle_id        UUID        NOT NULL,
  step_id         TEXT        NOT NULL,
  profile         TEXT        NOT NULL,      -- e.g. wheel-fitment-rl, door-trim-harness-fl
  profile_version TEXT        NOT NULL,      -- hash of the profile directory
  violation_type  TEXT        NOT NULL,      -- SEQUENCE | PRECEDENCE | EVIDENCE_MISSING | REGRESSION | STEP_TIMEOUT | INCOMPLETE_AT_RELEASE
  observed_ip     TEXT,                      -- e.g. bolt-2, clip-4
  expected_ip     TEXT,
  detail          JSONB       NOT NULL DEFAULT '{}',
  engine_ms       REAL        NOT NULL,      -- commit frame -> OPC-UA ack
  evidence_ms     REAL        NOT NULL,      -- first evidence frame -> OPC-UA ack
  clip_key        TEXT,                      -- Garage object key, filled by the sidecar
  narrative       TEXT        NOT NULL,      -- from the interpreter's facts, never empty
  adjudicated     BOOLEAN     NOT NULL DEFAULT FALSE,
  cad_revision    TEXT        NOT NULL,
  model_version   TEXT        NOT NULL,
  reviewed_at     TIMESTAMPTZ,
  false_positive  BOOLEAN
);
SELECT create_hypertable('anomalies', 'ts');
CREATE INDEX ON anomalies (station_id, ts DESC);
CREATE INDEX ON anomalies (violation_type, ts DESC);

CREATE TABLE cycles (
  ts          TIMESTAMPTZ NOT NULL,
  station_id  TEXT        NOT NULL,
  cycle_id    UUID        NOT NULL,
  passed      BOOLEAN     NOT NULL,
  duration_s  REAL        NOT NULL
);
SELECT create_hypertable('cycles', 'ts');

CREATE MATERIALIZED VIEW ftr_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', ts) AS bucket, station_id,
       avg(passed::int)::real     AS first_time_right,
       count(*)                   AS cycles
FROM cycles GROUP BY bucket, station_id;
```

`cad_revision` and `model_version` on every row are not decoration. When engineering revises the
CAD or the detector is retrained, those two columns are how an engineer separates "the line got
worse" from "the system changed."

### Fleet health tables

```sql
-- platform/migrations/002_fleet_health.sql
CREATE TABLE station_health (            -- one row per station per bucket, from the engine
  bucket_start        TIMESTAMPTZ NOT NULL,
  station_id          TEXT        NOT NULL,
  bucket_s            SMALLINT    NOT NULL,        -- 300 in production, 5 in demo mode
  fps                 REAL        NOT NULL,
  dropped_frame_rate  REAL        NOT NULL,
  ip_conf_mean      REAL,                        -- NULL when no point committed in the bucket
  ambiguous_rate      REAL        NOT NULL,
  anchor_residual_px  REAL,
  evidence_p99_ms     REAL,
  camera_temp_c       REAL,                        -- NULL for file and webcam sources
  packets_resent      INTEGER,                     -- GigE resends in the bucket; fixed limit, not charted
  adjudications       INTEGER     NOT NULL DEFAULT 0,
  model_version       TEXT        NOT NULL,
  cad_revision        TEXT        NOT NULL
);
SELECT create_hypertable('station_health', 'bucket_start');

CREATE TABLE host_health (               -- every 10 s, from the fleet agent
  ts               TIMESTAMPTZ NOT NULL,
  station_id       TEXT        NOT NULL,
  node             TEXT        NOT NULL,
  engine_up        BOOLEAN     NOT NULL,
  gpu_temp_c       REAL,
  sm_clock_ratio   REAL,
  gpu_mem_mib      REAL,
  disk_free_bytes  BIGINT
);
SELECT create_hypertable('host_health', 'ts');
SELECT add_retention_policy('host_health', INTERVAL '30 days');

CREATE TABLE health_events (             -- drift episodes, offline periods, baseline resets
  ts          TIMESTAMPTZ NOT NULL,
  station_id  TEXT        NOT NULL,
  kind        TEXT        NOT NULL,       -- DRIFT | DRIFT_CLEARED | ENGINE_OFFLINE | ENGINE_ONLINE | BASELINE_RESET
  metric      TEXT,
  detail      JSONB       NOT NULL DEFAULT '{}'
);

CREATE TABLE health_chart_state (        -- EWMA state: a 24 h baseline must survive a restart
  station_id  TEXT        NOT NULL,
  metric      TEXT        NOT NULL,
  state       JSONB       NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (station_id, metric)
);
```

`station_health` is kept for a year — one row per station per five minutes is about 105,000 rows a
year per station, which is nothing. `host_health` samples every ten seconds and keeps thirty days.

### Object layout in Garage

```
torquesense/
  clips/{station_id}/{yyyy}/{mm}/{dd}/{cycle_id}.mp4     6 s, H.264, 960x540
  cad/{cad_revision}/view_{cycle_id}.png                 CAD view with offending bolt highlighted
  hardneg/{model_version}/{cycle_id}_{frame}.jpg         engineer-flagged false positives
```

### Retention and privacy

Clips show workers' hands and sometimes faces. Under India's DPDP Act 2023, identifiable video of
employees is personal data. Three design rules, all cheap: the camera frames the **hub region of
interest** rather than the operator; clips keep **30 days** by default, then are deleted by a Garage
lifecycle job; and hard-negative frames are **cropped to the hub** before they leave the plant for
retraining. Say this in the business section — it is the question an HR or union representative
would raise before any Tata plant installed a camera at a station.

---

## 16. Verification and validation

Three questions, three kinds of evidence.

### Is the engine correct? — the invariants

| # | Invariant | Enforced by |
|---|---|---|
| I1 | **Zero heap allocation** in the steady-state hot path | `NoAllocationTest`, counting allocator, 10,000 frames |
| I2 | **No mutex** in `perception/` or `procedure/` | review + TSan suite in WSL2 |
| I3 | **Florence-2 never blocks an alert** | `AdjudicationNeverBlocksVerify`, circuit-breaker tests |
| I4 | **Evidence latency p99 ≤ 150 ms**, single station, 30 fps | `latency_test`, hard CI gate on the laptop |
| I5 | **Clean** under `/W4 /WX` (MSVC), `-Wall -Wextra -Wpedantic -Werror`, ASan, UBSan, TSan | CI matrix |
| I6 | **Tests before implementation** in `procedure/` | review rule |
| I7 | **Peak VRAM < 3.2 GB single-station, < 5 GB four-station** | `MemoryCeiling` soak test |
| I8 | **Never commit** weights, renders, videos, `.env`, `garage.toml` | `.gitignore` + DVC |
| I9 | **Renders are reproducible**: identical labels, pixels within PSNR ≥ 45 dB | `test_render_reproducible.py` |
| I10 | **Synthetic and real mAP reported separately**, always | `training/eval.py` |
| I11 | **Florence-2 is unreachable from `Pipeline::tick()`** | link-time test: the engine library does not link the HTTP client |
| I12 | **Model artefacts are verified before load**: the ONNX file's SHA-256 matches `models/manifest.json`; Florence-2 is pinned to a commit `revision` | the engine refuses to start on a mismatch; `ManifestMismatchTest` |
| I13 | **Fail visible**: if the engine's heartbeat stops for 2 s, the station shows AMBER and `InspectionOnline` goes false — decided by the PLC | `tests/plc/test_heartbeat.py`; the kind smoke test scales the engine to zero |
| I14 | **No profile goes live without passing its gate**: coverage clean, every scenario verdict exact in both implementations, per-family recall on real photos reported with confidence intervals, one canary shift | `VALIDATION.md` in the profile directory; `profile_gate.py` in CI |
| I15 | **Every required step has a deadline** — its own `max_duration_s`, or the release gate | the profile loader rejects a profile otherwise |
| I16 | **The C++ engine and the reference interpreter agree** on every scenario's violation list | differential test over every profile's fixtures |
| I17 | **No unresolved inspection point**: each one in frame, facing a camera, unoccluded, and at least 16 px across | the compiler refuses to emit the profile |

I9 was tightened from "byte-identical" in an earlier draft. Cycles renders with a fixed seed are
reproducible on one GPU and driver, but GPU path-tracing is not guaranteed bit-exact across driver
versions. Identical labels and a PSNR threshold is the honest contract.

I11 is enforced structurally rather than by review: the verify thread's library, `sunverify_procedure`,
does not link `cpp-httplib`. If someone writes a direct call to the sidecar from the hot path, it
fails to link.

I13 was tested while writing this revision, against a soft PLC built on open62541 1.4.21 — the
version vcpkg ships: AMBER before the first heartbeat, cleared while the counter moved, and back
2.2 s after the last beat (the 2 s rule plus the 250 ms check interval). Section 30 has the test.

### Is the detector accurate? — the two numbers

`training/eval.py` reports mAP50 and mAP50-95 on two sets, never one:

| Set | Size | What it proves | What it cannot prove |
|---|---|---|---|
| Synthetic held-out | 400 frames, unseen seeds | the model learned the rendered classes | that it works on real pixels |
| Real validation | 40 phone photos of real wheel hubs, labelled in Label Studio | `hub`, `wheel`, `bolt_absent`, `bolt_seated` transfer to real imagery | `bolt_torqued` — real hubs do not carry your yellow witness marks |

That last cell is the honest limit, and you should state it on the slide. The torque-mark class is
validated on synthetic data only. Closing that gap is one afternoon with a real hub, a paint pen,
and forty photos — it is Tier 2 work, not a flaw in the method.

### Is the procedure right? — the scenario test matrix

Twenty-five table-driven cases minimum, run on rendered scenario videos with `.expected.json`
siblings. The two that matter most are the ones that must **not** alert: hand occlusion mid-sequence,
and a second person entering the frame. False alarms on a production line get a system unplugged
faster than missed defects get it noticed.

---

## 17. Industrial and safety parameters

### OPC-UA — governs the interface

The alert is an OPC-UA write (IEC 62541), the protocol most modern PLCs and MES layers expose. The
prototype's soft PLC publishes:

```
ns=2;s=Station.W14.Andon.Red          Boolean
ns=2;s=Station.W14.Andon.Amber        Boolean
ns=2;s=Station.W14.Andon.Green        Boolean
ns=2;s=Station.W14.Conveyor.HoldReq   Boolean      ← a request, not a command
ns=2;s=Station.W14.LastViolation      String
ns=2;s=Station.W14.CycleState         Int32
ns=2;s=Station.W14.VariantCode        String       ← selects the variant's procedure at cycle start
ns=2;s=Station.W14.ReleaseRequest     Boolean      ← the release gate: may this part leave?
```

### Functional safety — what this system is not

TorqueSense is a **quality** system, not a **safety** function. It is not designed or claimed to
IEC 61508 or ISO 13849, and it must never be the only thing standing between an operator and a
hazard. That is why the conveyor tag is `HoldReq` — a *request*. The PLC owns the conveyor and
decides whether to honour the request according to its own validated logic. A camera system that
directly commanded machinery would be a safety function and would need certification it does not
have.

This distinction is worth one sentence in the pitch. It tells a Tata engineer you know where the
boundary is.

### Torque practice — governs the claim

Star or cross-pattern tightening spreads clamp load evenly across the hub flange. Witness marks
show which fasteners have been torqued. The nutrunner controller remains the authority on torque
value. TorqueSense adds what neither provides without fixtures: geometric verification of sequence
and completeness, derived from the CAD.

### DPDP Act 2023 — governs the video

Covered in Section 15: region-of-interest framing, 30-day retention, hub-cropped retraining data.

---

## 18. Fleet health and reliability

The verification engine answers *was this wheel fitted correctly?* This section answers the
question a plant asks next: *is the inspector itself still working?*

Camera systems rarely fail loudly. A smeared lens, a mount knocked two degrees, a fan clogged with
dust — each lowers performance gradually, and the first symptom an operator sees is a false alarm,
by which point trust is already spent. So the stations are treated as a fleet of devices and
watched the way a storage team watches a drive fleet: telemetry from every device, statistical
drift detection against each device's own normal, and a playbook from signal to likely cause.

### Two tiers, two speeds

| Tier | Detects | Speed | Who acts | Effect on the line |
|---|---|---|---|---|
| **Fail-visible** | engine dead, camera frozen, pipeline hung | 2 s | the PLC, automatically | AMBER, `InspectionOnline` false; the PLC decides whether parts may pass |
| **Drift** | slow degradation | minutes to hours | a maintenance engineer | none — a ticket |

Health rules never raise or suppress a quality alert. Keeping the two systems apart means a noisy
health signal can never stop the line, and a healthy-looking station can never mask a real defect.

### Fail-visible — the PLC owns it

```mermaid
sequenceDiagram
  participant V as verify thread
  participant A as alert thread
  participant P as PLC watchdog, every 250 ms
  participant L as Andon and HMI
  loop every 500 ms, while frames are being verified
    V->>A: frames_verified has moved
    A->>P: Heartbeat = n + 1
    P->>L: InspectionOnline true · Amber off
  end
  Note over V,A: camera freezes, a thread hangs, or the process dies
  P->>P: Heartbeat unchanged for 2 s
  P->>L: InspectionOnline false · AMBER
```

*Figure 4b — The heartbeat. The engine proves it is working by changing a number; the PLC notices
when it stops. Nothing has to report a failure, which is the point: the failures that matter are
the ones that cannot report themselves.*

The watchdog lives in the PLC, not in the fleet agent, because the PLC owns the line, runs on its
own hardware, and keeps running when the PC does not. This is the standard industrial pattern for a
supervised device. The soft PLC's version, as built and tested:

```c
/* softplc/src/watchdog.c — compiled and run against open62541 1.4.21 */
#define STALE_MS 2000
static UA_UInt32  last_beat   = 0;
static UA_DateTime last_change = 0;            /* 0 = never seen: inspection not ready */

static void watchdog(UA_Server *s, void *data) {          /* registered every 250 ms */
    UA_Variant v; UA_Variant_init(&v);
    if (UA_Server_readValue(s, UA_NODEID_STRING(ns, "Station.W14.Heartbeat"), &v) != UA_STATUSCODE_GOOD)
        return;
    UA_UInt32 beat = *(UA_UInt32 *)v.data;
    UA_Variant_clear(&v);
    UA_DateTime now = UA_DateTime_nowMonotonic();
    if (beat != last_beat) { last_beat = beat; last_change = now; }
    UA_Boolean online = last_change != 0 && (now - last_change) < STALE_MS * UA_DATETIME_MSEC;
    set_bool(s, "Station.W14.InspectionOnline", online);
    set_bool(s, "Station.W14.Andon.Amber", !online);
}
/* in main(), once per configured station (W-14 shown; the soft PLC reads the station list
   from its command line, so four-station mode gets four independent watchdogs):
   UA_Server_addRepeatedCallback(server, watchdog, station, 250, NULL); */
```

A station is AMBER from the moment the soft PLC starts until the engine's first beat. That is
intended: *not ready* is not *OK*. During the demo, start-up takes a few seconds and the HMI says why.

### The six signals

Five come from the engine's io thread, aggregated per bucket; one comes from the fleet agent.

| Signal | Source | Bad direction | What moves it | σ floor |
|---|---|---|---|---|
| `ip_conf_mean` | engine | down | lens contamination, lighting, focus | 0.005 |
| `ambiguous_rate` | engine | up | the same causes, usually earlier | 0.002 |
| `anchor_residual_px` | engine | up | camera knocked or mount loosened | 0.5 px |
| `evidence_p99_ms` | engine | up | thermal throttling, GPU contention | 2 ms |
| `dropped_frame_rate` | engine | up | capture link, decode or disk stalls | 0.001 |
| `sm_clock_ratio` | fleet agent | down | throttling | 0.01 |

`anchor_residual_px` exists only because of this design. Inspection points come from projecting CAD
positions into the image, so the engine knows where every part *should* appear. The mean distance
between detections and those projections is close to constant — until someone knocks the camera.
It rises long before detection fails.

The σ floor stops a near-constant baseline from producing an estimated spread of zero, which would
make the smallest blip look like a catastrophe.

### Drift detection — limits set by simulation, not by habit

Each station runs one **one-sided EWMA control chart** per signal, on bucket means (five minutes in
production). One-sided, because confidence rising is never a problem; the chart only watches the
bad direction, which halves false alarms for free.

$$z_t = \lambda u_t + (1-\lambda) z_{t-1}, \qquad u_t = \pm\frac{x_t-\hat\mu_0}{\hat\sigma_0}, \qquad \text{drift when } z_t > L\sqrt{\tfrac{\lambda}{2-\lambda}\left(1-(1-\lambda)^{2t}\right)}$$

$\hat\mu_0$ and $\hat\sigma_0$ come from a **phase I baseline**: the median and the scaled median
absolute deviation of the first 288 buckets — 24 hours — after a deploy. Median and MAD rather than
mean and standard deviation, so one bad hour in the baseline does not poison it.

**The budget.** At most one false drift ticket per station per month: six charts, 8,640 five-minute
buckets a month. Every figure below is from a Monte Carlo simulation of independent, normal bucket
means, run while writing this section: 4,000 simulated charts per estimated-baseline row, and 800
to 2,000 long in-control runs for the known-parameter rows.

| Configuration | False tickets, per station-month | Median delay to flag a shift of 0.5σ · 1σ · 2σ |
|---|---|---|
| Textbook λ 0.2, L 3, parameters *known* | ≈ 48 | — |
| λ 0.05, L 3.9, parameters *known* | ≈ 0.87 | mean run length 71 · 15 · 4.4 buckets |
| λ 0.05, L 3.9, *24 h estimated* baseline | ≈ 1.7 — over budget | — |
| **λ 0.05, L 4.2, 24 h estimated baseline — chosen** | **≈ 0.92** | **73 · 17 · 5 buckets (≈ 6 h · 85 min · 25 min)** |

The textbook setting has an in-control average run length of about 1,090 buckets — under four days
per chart, more than one false ticket a day per station. A maintenance engineer would mute that
within a week.

**What the simulation exposed.** With the parameters *known*, λ 0.05 and L 3.9 meets the budget. With
parameters *estimated* from a baseline — the only way they are ever obtained — it does not. An error
of a quarter of a sigma in $\hat\mu_0$ acts like a small permanent shift, and a chart tuned to catch
small shifts catches that too. The effect shrinks slowly with baseline length:

| Baseline length at L 3.9 | 2 h | 12 h | 24 h | 48 h | 96 h | known |
|---|---|---|---|---|---|---|
| Charts with a false ticket within a month | 46.9% | 34.1% | 28.4% | 24.1% | 20.3% | 14.5% |

So L is raised to 4.2 for the 24-hour baseline that is actually used. The first version of the unit
test, written with a two-hour baseline, failed for exactly this reason — the right outcome for a test.

**The assumption to state out loud.** The simulation assumes independent buckets. Real health data
is autocorrelated — GPU temperature especially — which shortens in-control run lengths. Bucket means
of thousands of frames are close to normal, but not independent. So every ticket's outcome is logged
(`DRIFT`, then confirmed or dismissed by the engineer), and L is recalibrated after two weeks of
real data. That makes the false-alarm budget a measured number rather than a promise.

**Baselines reset on redeploy.** A new model or a new CAD revision legitimately changes confidence
and residuals. When `model_version` or `cad_revision` changes, that station's charts return to phase
I and a `BASELINE_RESET` event is logged. Without this, every deployment would look like drift.

```python
# api/fleet/ewma.py — the core of the chart; tests in api/tests/test_ewma.py
@dataclass
class Chart:
    metric: str
    direction: Literal["up", "down"]      # which way is bad
    sigma_floor: float
    lam: float = 0.05
    L: float = 4.2                        # set by simulation for a 288-bucket estimated baseline
    baseline_n: int = 288                 # 24 h at 5-minute buckets; shorter inflates false alarms
    clear_n: int = 6                      # buckets back inside limits before a drift episode closes

    def update(self, x: float) -> tuple[State, bool]:
        if self.state is State.BASELINE:
            self._phase1.append(x)
            if len(self._phase1) >= self.baseline_n:
                self.mu0 = statistics.median(self._phase1)
                mad = statistics.median(abs(v - self.mu0) for v in self._phase1)
                self.sigma0 = max(1.4826 * mad, self.sigma_floor)
                self.state = State.IN_CONTROL
            return self.state, False
        sign = 1.0 if self.direction == "up" else -1.0
        self.t += 1
        self.z = self.lam * sign * (x - self.mu0) / self.sigma0 + (1 - self.lam) * self.z
        out = self.z > self.limit()
        if self.state is State.IN_CONTROL and out:
            self.state, self._inside = State.DRIFT, 0
            return self.state, True                       # a new episode: raise one ticket
        if self.state is State.DRIFT:
            self._inside = 0 if out else self._inside + 1
            if self._inside >= self.clear_n:
                self.state = State.IN_CONTROL             # hysteresis: no flapping tickets
        return self.state, False
```

The drift job runs in the API once per bucket: it loads each chart's state from
`health_chart_state`, feeds it the new bucket, writes the state back, and inserts a `health_events`
row when an episode opens or closes. It is plain Python; at four stations and six charts it does 24
updates every five minutes.

```mermaid
flowchart LR
  E["engine io thread<br/>bucket aggregates"] --> SH[("station_health")]
  G["fleet-agent<br/>every 10 s"] --> HH[("host_health")]
  SH --> J["drift job · each bucket<br/>EWMA per station × signal"]
  HH --> J
  J <--> CS[("health_chart_state")]
  J --> HE[("health_events")]
  HE --> F["fleet view · attention list"]
  SH --> F
```

*Figure 4c — Health data path. Two writers, one job, one reader. Chart state is persisted every
bucket, so a restart costs nothing.*

### From signal to cause — the playbook

The fleet view does not stop at "W-16 is amber". It shows which signals moved together and what that
pattern usually means.

| Signals moving together | Likely cause | Action |
|---|---|---|
| `ip_conf_mean` ↓ and `ambiguous_rate` ↑, latency flat | lens contamination, lighting change | clean the lens or check lighting at the next break |
| `anchor_residual_px` ↑ | camera knocked, mount loose | check the ghost overlay alignment; re-tighten and re-calibrate |
| `ambiguous_rate` ↑ on one bolt position only | partial occlusion: a cable or fixture in view | inspect the station |
| `evidence_p99_ms` ↑ and `sm_clock_ratio` ↓ | thermal throttling | check fans, filters, cabinet temperature |
| `evidence_p99_ms` ↑, clocks normal | contention: another process on the GPU or the pinned cores | check what else is running |
| `dropped_frame_rate` ↑ | capture link, decode or disk stalls | check the camera cable and the clip writer's free disk |
| everything shifts just after a deploy | not drift — a new model | baselines reset; if the shift is bad, the canary has failed |

### Canary, promote, roll back

Drift charts judge a station against its own past. A canary must be judged against its **peers over
the same shift**: they answer different questions, and a canary's own charts are still learning a
new baseline anyway.

1. Deploy the new model to one station. Its charts reset to phase I, as expected.
2. Over one shift, compare it with the other stations on first-time-right rate, engineer-confirmed
   false positives, ambiguous rate and evidence p99.
3. Promote it to the rest of the line, or roll back.

On the laptop, a deploy is restarting one station's engine with a different `perception.model_path`.
In Tier 2 it is `kubectl set image` on one Deployment, and roll-back is `kubectl rollout undo`
(Section 31). Either way the model's SHA-256 is checked against `models/manifest.json` before the
engine will load it (I12):

```json
{
  "yolo11n_wheel.onnx": {
    "sha256": "<64 hex characters>",
    "version": "yolo11n-w14-s20260721",
    "trained_from": "data/synth/3f2a91c0b7de-g3-s20260721",
    "accuracy_report": "docs/accuracy.json"
  }
}
```

The hash is computed with picosha2, a header-only library, so the check adds no platform code.

### The fleet agent

One per station, every ten seconds: probe the engine's `/health`; sample the GPU — through
`nvidia-smi` on the laptop, through the node's dcgm-exporter on a station (`-gpu-source=dcgm`,
Section 32); read free disk space; push one JSON record to the API. Written in Go using only the
standard library — a single static binary of about 6 MB, cross-compiled for Windows and Linux from
either, with nothing third-party to patch on an edge PC.

When the API is down it spools to a capped JSONL file (5 MiB, so a dead API cannot fill the disk).
Every record goes *through* the spool, oldest first:

```go
// fleet-agent/main.go — one path for every record
line, _ := json.Marshal(rec)
if err := sp.Append(line); err != nil {
    slog.Error("spool append failed; record lost", "err", err) // full or unwritable: say so
}
sent, pending, err := sp.Flush(func(b []byte) error { return post(ctx, client, *apiURL, apiKey, b) })
if pending > 0 {
    slog.Warn("api unreachable; records held in spool", "pending", pending, "sent", sent)
}
```

The first version sent the new record directly and replayed the spool afterwards. An outage test
caught it: after the API came back, records arrived out of order — and the drift job reads buckets
in time order. Routing everything through the spool fixed it. Re-run, 17 records crossed a simulated
outage and arrived complete and in order.

The agent also reports `engine_up`. That is how the fleet view knows a station is offline even
though its engine — the thing that would normally report — is the thing that died.

### Security, as a fleet property

| Concern | Prototype | Tier 2 |
|---|---|---|
| A tampered or wrong model | SHA-256 against the manifest before load (I12) | the same, with the manifest signed |
| An agent as an attack surface | listens on no port; push only | the same |
| Certificate expiry — the classic fleet-wide outage | — | OPC-UA certificate expiry dates reported by the agent as a health signal, ticketed 30 days out |
| Images from the internet on the line | — | in-plant registry, pinned tags (Section 31) |

---

# Part IV — Build

## 19. Tool register

One pick per layer, each chosen because it fits something already in the stack *and* this machine.
Every item is free and open source, or free tier.

### CAD and synthetic data

| Layer | Pick | Reason |
|---|---|---|
| CAD modelling | [FreeCAD](https://www.freecad.org/) 1.0 or later, Assembly workbench | Free, exports STEP and glTF from the same document. |
| CAD kernel, C++ | [Open CASCADE Technology](https://github.com/Open-Cascade-SAS/OCCT) via vcpkg | The kernel FreeCAD is built on. Reading STEP assemblies through `XCAFDoc_ShapeTool` is CAD-kernel work in C++ — the same class of work as NX Open. |
| Rendering + auto-labels | [BlenderProc](https://github.com/DLR-RM/BlenderProc) | Built for exactly this problem: domain-randomised renders with COCO annotations. Downloads and manages its own Blender, so there is no Blender version to get wrong. |
| Label review, real set | [Label Studio](https://github.com/HumanSignal/label-studio) in Docker | Only for the 40 real validation photos. Run it when needed, stop it after. |
| Dataset versioning | [DVC](https://dvc.org/) | Renders and weights are never committed (I8); DVC records which seed produced which model. |

### Perception and reasoning

| Layer | Pick | Reason |
|---|---|---|
| Detector | [Ultralytics YOLO11n](https://github.com/ultralytics/ultralytics) | Hands and tools, shared by every profile. 2.6M parameters, 14 ms on this GPU through ONNX Runtime. |
| Inspection-point states | MobileNetV3-Small from [torchvision](https://pytorch.org/vision/), one per state family, exported to ONNX | A closed two- or three-way choice on a 96×96 crop: small enough to batch every point in view in a few milliseconds, and reusable wherever the family appears. |
| Registration | AprilTag fiducials through OpenCV's ArUco module | Places inspection points on a moving carrier every frame. |
| Training | [Google Colab](https://colab.research.google.com/) T4 | Section 26 explains why it is used for this and nothing else. |
| Inference runtime, C++ | [ONNX Runtime](https://github.com/microsoft/onnxruntime) ≥ 1.27, **`gpu_cuda13`** build, CUDA execution provider | Matches your CUDA 13.2 toolkit. TensorRT is deliberately not used (Section 25). |
| GPU libraries | [cuDNN](https://developer.nvidia.com/cudnn) 9.x **for CUDA 13** | Required by the ONNX Runtime CUDA EP. The CUDA 13 build, not the CUDA 12 one. |
| Tracking | ByteTrack, reimplemented (~200 lines) | IoU association with a Kalman predictor. Reimplementing avoids a Python dependency on the hot path; bolt identity comes from CAD anchoring anyway. |
| Vision-language model | [Florence-2-base](https://huggingface.co/florence-community/Florence-2-base) via [transformers](https://github.com/huggingface/transformers) | 230M parameters, fits beside the detector in 6 GB. Phrase grounding is the exact task adjudication needs. |
| PyTorch | [PyTorch](https://pytorch.org/) ≥ 2.11, **cu130** wheels | The first line with CUDA 13 wheels. Florence-2 only; the C++ engine does not use it. |

### Engine, C++

| Layer | Pick | Reason |
|---|---|---|
| Compiler | MSVC from Visual Studio Build Tools, C++ desktop workload | Native Windows ABI; ONNX Runtime's prebuilt binaries are MSVC builds. |
| Build | [CMake](https://cmake.org/) + [Ninja](https://ninja-build.org/) + [vcpkg](https://github.com/microsoft/vcpkg) manifest mode | Ninja keeps the Visual Studio version out of CMake entirely. Manifest mode pins every library in one file. |
| Video I/O | [OpenCV](https://opencv.org/) with FFmpeg | File replay, webcam, JPEG encode for the clip ring. |
| Production camera | [Basler ace 2 a2A1920-51gcBAS](https://www.baslerweb.com/en/shop/a2a1920-51gcbas/) | Global shutter, 1920×1200, GigE with PoE; one camera per NIC port (Section 8). |
| Camera SDK | [Basler pylon](https://docs.baslerweb.com/pylon-software-suite) C++ | GigE Vision driver; `IBufferFactory` lets the driver write into the engine's own pool. Production builds only. |
| Camera calibration | OpenCV `objdetect` ChArUco + `calibrateCamera` | Intrinsics with a 0.5 px RMS gate, written into `cameras.yaml`. |
| OPC-UA | [open62541](https://github.com/open62541/open62541) | Client in the engine, server in the soft PLC. One library, both sides. |
| WebSocket server | [IXWebSocket](https://github.com/machinezone/IXWebSocket) | Small, Windows-friendly, no libuv dependency. |
| HTTP | [cpp-httplib](https://github.com/yhirose/cpp-httplib) | MJPEG stream, control endpoint, sidecar client. Header-only. |
| JSON | [nlohmann/json](https://github.com/nlohmann/json) | Off the hot path only. |
| Excel | [OpenXLSX](https://github.com/troldal/OpenXLSX) | Shift report written from C++. |
| Logging | [spdlog](https://github.com/gabime/spdlog) | Async logger on the io thread. |
| Tests | [GoogleTest](https://github.com/google/googletest) · [Google Benchmark](https://github.com/google/benchmark) | Unit, property, and microbenchmark. |
| Latency histograms | [HdrHistogram_c](https://github.com/HdrHistogram/HdrHistogram_c) via FetchContent | p99 and p99.9 without storing every sample. |

### Platform and front end

| Layer | Pick | Reason |
|---|---|---|
| Time-series store | [TimescaleDB](https://github.com/timescale/timescaledb) on PostgreSQL 17 | Hypertables and continuous aggregates make first-time-right per hour one query. |
| Object store | [Garage](https://garagehq.deuxfleurs.fr/) v2.3 | MinIO stopped publishing community Docker images in October 2025 and archived its repository in 2026. Garage is a single-binary S3-compatible store that runs in one 256 MB container. |
| API | [FastAPI](https://fastapi.tiangolo.com/) | Events in, history out. |
| Tag monitor | [Node-RED](https://nodered.org/) + [node-red-contrib-opcua](https://github.com/mikakaraila/node-red-contrib-opcua) | Shows real OPC-UA tags as industrial indicators — proof the alert is a real protocol write, not a coloured `div`. |
| Front end | React + [Vite](https://vite.dev/) + TypeScript + [Tailwind](https://tailwindcss.com/) + [shadcn/ui](https://ui.shadcn.com/) | Dashboard and the operator HMI from one codebase. |
| Charts · icons | [Recharts](https://recharts.org/) · [Lucide](https://lucide.dev/) | — |
| Typography | IBM Plex Sans and Plex Mono via [Fontsource](https://fontsource.org/) | Tabular figures for latency readouts; self-hosted, works offline. |
| End-to-end tests | [Playwright](https://playwright.dev/) | Dashboard flows and the fault harness. |
| Containers | Docker Desktop, WSL2 backend | The only backend Windows 11 Home supports. |

### Operations

| Layer | Pick | Reason |
|---|---|---|
| CI for the GPU gates | [Jenkins](https://www.jenkins.io/) LTS (`jenkins/jenkins:lts-jdk21`) in Docker, with an inbound agent on Windows | Hosted CI has no GPU. GitHub recommends self-hosted runners only for private repositories, because a pull request from a fork can run code on the runner. Jenkins polling this repository's `main` branch never runs fork code, and exposes nothing on the laptop. |
| Java for the agent | [Eclipse Temurin](https://adoptium.net/) 21 | The Jenkins agent is a Java process. |
| Fleet agent | [Go](https://go.dev/), standard library only | One static binary, cross-compiled for Windows and Linux, no runtime to install. |
| Model integrity | [picosha2](https://github.com/okdshin/PicoSHA2) via vcpkg | Header-only SHA-256: the manifest check needs no platform code. |
| Orchestration, Tier 2 | [RKE2](https://docs.rke2.io/) 1.36 | Hardened Kubernetes with containerd 2.3; NVIDIA documents its GPU Operator for it explicitly. The full platform register is in Section 32. |
| Manifests | [Kustomize](https://kustomize.io/), built into `kubectl` | One base, two overlays (`kind`, `edge`). Hashed ConfigMaps roll a station when its procedure changes. |
| Manifest testing | [kind](https://kind.sigs.k8s.io/) in GitHub Actions + [kubeconform](https://github.com/yannh/kubeconform) | A real cluster in CI at no cost to the laptop's RAM; schemas checked strictly. |
| GPUs in Kubernetes | [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/) | Driver, toolkit, device plugin and dcgm-exporter as one unit; GPUs injected through CDI, so pods need no runtime class. |
| OPC-UA test client | [asyncua](https://github.com/FreeOpcUa/opcua-asyncio) | Python; reads and writes PLC tags in the heartbeat and cluster smoke tests. |
| Decision records | Architecture Decision Records in `docs/adr/` | Every decision this spec argues for, in one numbered file each, where a reviewer can find it. |
| Reference interpreter | Python, `reference/tsref` | The executable specification of the procedure model; the C++ engine is differentially tested against it (I16). |
| Occlusion test | OCCT `IntCurvesFace_ShapeIntersector` | Ray-casts camera-to-point lines against the assembly's own bodies at compile time. |

---

## 20. Interface design

### The design thesis

Two audiences, two surfaces, one codebase. **The operator reads nothing; the engineer reads
everything.** The HMI is a full-screen kiosk route of the same React app as the dashboard, so one
WebSocket, one set of tokens, one build.

#### Colour tokens

```css
/* web/src/styles/tokens.css */
:root {
  --bg:            #0F1115;   /* shop-floor dark: no glare on a station screen */
  --surface:       #171A21;
  --surface-2:     #1F232C;
  --text:          #E8EAED;
  --text-muted:    #9AA0A6;
  --line:          #2A2F3A;

  --state-ok:      #22C55E;   /* complete */
  --state-active:  #F5A524;   /* in progress, next bolt */
  --state-fault:   #EF4444;   /* violation */
  --state-info:    #0E7C86;   /* CAD provenance, links */

  --focus:         #7DD3FC;
  --radius:        8px;
}
```

The three state colours differ in luminance as well as hue, so they remain distinguishable in
greyscale and for red-green colour blindness. Check this on a projector before demo day; projectors
wash out amber first.

#### Type scale

| Token | Size / weight | Used for |
|---|---|---|
| `--t-hmi-hero` | 96 px / 700 | "NEXT: BOLT 3" on the operator HMI |
| `--t-hmi-label` | 40 px / 600 | step labels at three metres |
| `--t-h1` | 24 px / 600 | dashboard zone titles |
| `--t-body` | 14 px / 400 | dashboard text |
| `--t-mono` | 13 px / 500, Plex Mono, tabular | latency badge, timestamps, tag values |

### The signature element: the part map

```
        ╭───────────╮
      ╭─┤     1     ├─╮
    ╭─┤ ●         ○ ├─╮      ● torqued    ○ seated
    │ 5 ●         ○ 2 │      ◉ next, pulsing at 1 Hz
    ╰─┤ ●       ◉ ├─╯
      ╰─┤ 4     3 ├─╯        positions are the CAD bolt angles,
        ╰───────────╯        rotated to the camera's view
```

The part map is generated from the profile: the assembly's outline as the camera sees it, with a
marker at every inspection point, coloured by state. For the wheel it is the hub schematic above;
for the door it is the trim outline with six clip markers and a connector. Positions come from
`procedure.json` through the camera pose, so on-screen positions correspond to the physical part.
On violation the screen floods `--state-fault`, the offending point strobes, and for a sequence the
corrective order draws as a numbered arc. The operator never reads a sentence.

### The second invention: the ghost overlay

A semi-transparent CAD wireframe of the *correct* assembly state, projected onto the live feed with
the same extrinsics. In the software-only build the extrinsics are exact — Blender's camera is
known — so the overlay registers perfectly. When reality and CAD diverge, the divergence is visible
as a shape.

### Screen specifications

#### Operator HMI — `/hmi/:station`, Chrome kiosk

| Element | Rule |
|---|---|
| Step rail | five steps, top, 40 px labels; completed steps filled |
| Part map | centre, 60% of viewport height; generated from the profile |
| Next action | one line, 96 px |
| Cycle timer | bottom right, mono, turns amber past 80% of takt |
| Violation | full-bleed red, corrective arc, no text beyond "STOP — BOLT 4" |
| Interaction | none. The HMI accepts no input. |
| Inspection offline | if the WebSocket is silent for 1 s, full-bleed amber and "INSPECTION OFFLINE". The HMI does not wait to be told; the PLC's own AMBER follows within 2 s |

#### Engineer dashboard — `/`

Four zones, one screen, no tabs:

| Zone | Content |
|---|---|
| **A — Live feed** (top-left, 60%) | MJPEG from the engine on `:8090`, canvas overlay with boxes and CAD bolt IDs, ghost overlay toggle on `G`, and a **latency badge** showing engine and evidence latency, updating per frame |
| **B — Procedure timeline** (top-right) | steps bound to the interpreter, elapsed against allotted; a **CAD provenance chip**: `RL_WHEEL_ASSY_REV_C.step · compiled 09:14` |
| **C — Anomaly feed** (bottom-left) | cards with looping clip, violation type, station; modal with the bolt-state strip, CAD view, narrative, and **Confirm / False positive** |
| **D — Shift analytics** (bottom-right) | most-violated steps, first-time-right by hour, **Export to Excel** served by the engine's `XlsxReporter` |

#### Fault-injection harness — `/harness`

Styled deliberately as a test tool — monospace, muted border, "TEST HARNESS" in the header — so
nobody mistakes it for production UI. It selects which pre-rendered scenario feeds `IFrameSource`
at the next cycle boundary. It cannot influence the verdict: it changes the input, never the logic.

| Button | Scenario file |
|---|---|
| Correct cycle | `cycle_correct_01.mp4` |
| Skip bolt 3 | `fault_missing_bolt.mp4` |
| Adjacent order | `fault_adjacent_order.mp4` |
| Interrupt and resume | `fault_interrupt_resume.mp4` — the hero fault |
| Wheel off-axis | `fault_wheel_off_axis.mp4` |
| Tool never picked | `fault_no_tool.mp4` |
| Hand occlusion | `edge_hand_occlusion.mp4` — must **not** alert |
| Second person | `edge_second_person.mp4` — must **not** alert |
| Switch profile | loads the door-trim profile and its scenarios into the same running engine |

#### Station grid — `/line`

Four tiles, each with station state, cycle progress, and live latency. One summary line: first-time
right for the line, and the most-missed step.

#### Node-RED tag monitor — `localhost:1880/ui`

Three lamp indicators and a conveyor-hold indicator bound to the soft PLC's OPC-UA tags. Displayed
beside the dashboard during the demo. It is visually plain on purpose: it should look like the
SCADA screen a line engineer already knows.

#### Fleet view — `/fleet`

| Element | Rule |
|---|---|
| Station matrix | one row per station: engine status, model version, CAD revision, then the six signals |
| Signal cell | 24 h sparkline with the centre line and control limit drawn. Colour carries state only: neutral in control, amber in drift, dotted grey while learning a baseline, red when offline |
| Attention list | open drift episodes, oldest first, each with the playbook's likely cause and action |
| Canary tag | on any station running a model version the rest of the line is not |
| History banner | seeded synthetic history is labelled as synthetic, on screen, whenever it is shown |

A cell never shows a raw number without its limit. A value means nothing to a maintenance engineer
until it is drawn against that station's own normal.

---

## 21. Prototype build plan

### Demo topology

```
HSGPC (AC power, clocks locked)
├─ Docker: timescaledb · garage · api · node-red
├─ softplc.exe            opc.tcp://host.docker.internal:4840
├─ sidecar                http://127.0.0.1:8001     Florence-2 + clip writer
├─ torquesense.exe --station W-14   ws://127.0.0.1:8080  mjpeg :8090  control :8095
├─ fleet-agent.exe -station W-14    pushes to the API; probes :8095; listens on nothing
└─ Vite                   http://127.0.0.1:5173     /  /hmi/W-14  /harness  /line
Projector: dashboard on screen 1 · HMI kiosk on screen 2 if available
```

### The engine contract

Everything crossing a process boundary is defined once in `contracts/`, as JSON Schema, and
generated into C++ structs and TypeScript types. Contract drift becomes a compile error.

```jsonc
// contracts/state.schema.json — engine → dashboard, ~15 Hz over WebSocket
{
  "station_id": "W-14",
  "frame_seq": 18233,
  "cycle_id": "3f9c…",
  "step": "S4",
  "profile": "wheel-fitment-rl",
  "ips": [ { "id": "bolt-1", "family": "threaded_fastener", "state": "fastened", "conf": 0.97, "u": 824, "v": 576 }, … ],
  "next": ["bolt-3"],
  "latency": { "engine_ms_p50": 24.9, "engine_ms_p99": 34.6, "evidence_ms_p99": 69.3 },
  "verdict": null
}

// contracts/anomaly.schema.json — engine → API, once per violation
{
  "ts": "2026-…Z", "station_id": "W-14", "cycle_id": "3f9c…",
  "profile": "wheel-fitment-rl", "step_id": "torque", "violation_type": "SEQUENCE",
  "observed_ip": "bolt-2", "expected_ip": "bolt-3",
  "engine_ms": 33.1, "evidence_ms": 67.8,
  "ring_end_seq": 18233,
  "narrative": "Bolt 2 torqued before bolt 3. Expected 1 → 3 → 5 → 2 → 4. …",
  "cad_revision": "C", "model_version": "yolo11n-w14-s20260721"
}
```

### Forty-hour build order — one engineer

| Hours | Work | Gate |
|---|---|---|
| 0–5 | FreeCAD assembly; STEP + glTF export; vcpkg manifest install **started in background** (it takes an hour) | five fasteners at 114.3 mm PCD |
| 5–10 | `synth/generate.py`; 4,000 renders + 10 scenarios; 40 real photos labelled | I9 reproducibility test passes |
| 10–14 | Colab training: hands/tools detector and the `threaded_fastener` classifier; ONNX export; local ORT smoke test | both mAP figures reported (I10); CUDA EP confirmed |
| 14–20 | `SpscRingBuffer`, both frame pools, `StateVector`; OCCT reader, manifest join, pattern solvers, resolution check | TSan torture test; solver property tests; wheel `coverage.json` clean |
| 20–26 | Generic `ProcedureEngine` loading `procedure.json`; differential test against `reference/tsref`; `OpcUaAlerter`; soft PLC | reference and C++ agree on every wheel scenario (I16); `NoAllocationTest` green |
| 26–30 | Pipeline orchestration, pinning, HdrHistogram; `SharedClipRing` | **I4 latency gate** |
| 30–34 | Sidecar: Florence-2, clip writer; Docker stack; FastAPI | anomaly row + clip land end to end |
| 34–38 | Dashboard four zones, HMI, harness, station grid, Node-RED flow | demo runbook passes once |
| 38–40 | Four-station mode; rehearse four times; record fallback capture | GO on the demo checklist |

The vcpkg note in hour 0 matters on this machine. OpenCV with FFmpeg and Open CASCADE are both
built from source by vcpkg, and together take roughly an hour on the i7-11800H. Start it first and
let it run while you model in FreeCAD.

### Operations track — twelve more hours

Build the hackathon core first. The operations track must never delay the demo; it can be built
before or after it.

| Hours | Work | Gate |
|---|---|---|
| 40–43 | Engine health buckets; heartbeat on the alert thread; soft PLC watchdog; HMI offline state | I13: AMBER within 2.5 s of the last beat |
| 43–46 | Health tables, EWMA drift job, `/fleet`, seeded history with one drifting station | 7 EWMA tests; drift visible on `/fleet` |
| 46–48 | Go fleet agent with spool | `go test -race`; outage test delivers every record in order |
| 48–50 | Jenkins controller, Windows agent, `Jenkinsfile` | one green Jenkins build with the latency histogram archived |
| 50–52 | Dockerfiles, Kustomize base and overlays, `k8s-smoke` job | green in GitHub Actions |

Write the ADR for each decision as you make it, not afterwards — `docs/adr/0001-remove-tensorrt.md`,
`0002-garage-not-minio.md`, `0003-florence-local.md`, `0004-jenkins-for-gpu-gates.md`,
`0005-ewma-limits-by-simulation.md`, `0006-plc-owns-fail-visible.md`,
`0007-soft-plc-default-server-url.md`. Each one is a page: context, decision, consequences.

### Profiles track — ten more hours

Proves the engine is general. Build it after the hackathon core, never instead of it.

| Hours | Work | Gate |
|---|---|---|
| 52–55 | Door-trim CAD in FreeCAD — panel, six clips, connector; manifest; compile | `coverage.json` clean |
| 55–58 | `snap_clip` and `connector` renders and classifiers; door scenario fixtures | both mAP figures for each new family |
| 58–60 | The door profile end to end in the same engine; harness "switch profile" | panel over an open clip caught as `PRECEDENCE` |
| 60–62 | Windshield and enclosure: manifests, compile, scenario event streams through both interpreters | the two implementations agree; no perception models built |

The last row is deliberately cheap. Two more assemblies verified at the procedure level show the
engine is not wheel-shaped, without the cost of two more perception stacks. Say exactly that if asked.

### The demo pack

| Item | Location |
|---|---|
| 10 scenario videos + `.expected.json` | `data/scenarios/` |
| Frozen ONNX model, tagged | `models/yolo11n_wheel.onnx` |
| Wheel profile, revision C | `profiles/wheel-fitment-rl/` |
| Door-trim profile with its scenarios | `profiles/door-trim-harness-fl/` |
| Four-lug revision for the live reconfiguration | `cad/wheel_assembly_4lug.FCStd` |
| Fallback screen recording of a perfect run | `data/demo_fallback.mp4` |
| Seeded 30-day history for the analytics zone | `scripts/seed_history.py` |

### Offline insurance

The entire demo runs with the Wi-Fi off. Nothing at runtime calls the internet: Florence-2 weights
are cached in `%HF_HOME%` after first download, fonts are self-hosted, and Docker images are pulled
in advance. Turn Wi-Fi off during the final rehearsal to prove it.

### Repository layout

```
C:\dev\torquesense\
├── AGENTS.md                          spec for coding agents (Appendix B)
├── CMakeLists.txt  CMakePresets.json  vcpkg.json
├── .env.example    docker-compose.yml    Jenkinsfile
├── .github\
│   ├── workflows\ci.yml
│   └── pull_request_template.md       build phase · invariants touched · gate output pasted
├── cad\
│   ├── wheel_assembly.FCStd           5-lug, revision C
│   ├── wheel_assembly_4lug.FCStd      for the live reconfiguration demo
│   └── export\                        .step  .gltf  (generated)
├── contracts\                         JSON Schemas → C++ and TS types
├── profiles\                          one directory per assembly: wheel, door trim, windshield, enclosure
├── reference\                         Python · tsref: interpreter, compiler, pattern solvers, optics
├── cad-extract\                       C++ · OCCT reader, StarPatternSolver, ProcedureEmitter
├── perception\                        C++ · SpscRingBuffer, FramePool, YoloOnnxDetector, Tracker
├── procedure\                         C++ · ProcedureEngine, Verdict, IAlerter, OpcUaAlerter,
│                                            TemplateNarrator, XlsxReporter
├── ipc\                               C++ · SharedClipRing (Win32 + POSIX backends)
├── control-app\                       C++ · main, Pipeline, AlertThread (heartbeat), Affinity,
│                                            IFrameSource, ReasoningClient, HealthBuckets
├── softplc\                           C++ · open62541 server, heartbeat watchdog · node-red\flows.json
├── fleet-agent\                       Go · main.go, internal\probe (engine, gpu, disk, spool)
├── sidecar\                           Python · Florence-2 adjudicator, clip writer, FastAPI
├── synth\                             Python · BlenderProc generation, scenario sequences
├── training\                          Python · colab_train.ipynb, eval.py, dataset.yaml
├── api\                               Python · FastAPI, fleet\ewma.py + job.py
├── docker\                            engine · api · fleet-agent Dockerfiles
├── deploy\k8s\                        base\  overlays\kind\  overlays\edge\
├── platform\
│   ├── migrations\001_schema.sql  002_fleet_health.sql
│   └── garage\garage.toml.example
├── web\                               React · dashboard, /hmi, /harness, /line, /fleet
├── tests\
│   ├── latency\  soak\  scenarios\    C++ integration tests
│   ├── fixtures\detections\           per-scenario detector output, for ReplayDetector
│   ├── plc\test_heartbeat.py          I13
│   ├── k8s\smoke.py                   run by CI against kind
│   └── e2e\                           Playwright
├── scripts\                           bootstrap.ps1  start-demo.ps1  start-jenkins-agent.ps1
│                                      seed_history.py  gen-contracts.sh
├── data\                              DVC-tracked: synth\  scenarios\  real_val\
├── models\                            DVC-tracked · manifest.json committed
└── docs\
    ├── VALIDATION.md                  both mAP figures, latency histograms, scenario matrices, drift calibration
    ├── adr\                           0001-remove-tensorrt.md … 0007-soft-plc-default-server-url.md
    ├── functional-specification.md
    └── test-scenarios.md
```

---

## 22. Demo script

Five minutes, rehearsed to the second.

| Time | On screen | Line |
|---|---|---|
| 0:00 | Slide: the warranty claim | "Steering shudder eleven months after assembly. Root cause: five bolts torqued in the wrong order. The nutrunner logged five good rundowns. Every check passed." |
| 0:25 | Slide: torque arms | "Position arms solve this — per station, per variant, with hardware. We solve it with a camera that reconfigures from CAD." |
| 0:45 | FreeCAD → run the compiler | "We don't write the inspection rules. We compile them from the assembly." |
| 1:05 | `profiles/wheel-fitment-rl/procedure.json` | "Star pattern derived from bolt geometry. Nobody typed this." |
| 1:15 | Grid of rendered frames | "The same model generates the training data. Four thousand frames, zero manual labels." |
| 1:35 | Correct cycle; HMI green; Node-RED green | *Silence. Let it run.* |
| 2:10 | **Hand the laptop to a judge**: "Pick a fault." | — |
| 2:20 | Red floods the HMI; Node-RED `HoldReq` true | "Seventy milliseconds from the first frame showing that bolt. And that's real OPC-UA — point it at the station PLC and the same binary drives the tower." |
| 2:45 | Anomaly card | Read the narrative aloud. "That sentence comes from the state machine, not a language model — it's a fact, not a guess." |
| 3:05 | Modal: CAD view, bolt strip | "Florence-2 runs locally beside the detector, confirming bolts the detector isn't sure about. Nothing here touches the cloud." |
| 3:20 | `/fleet`, four stations | "One laptop, four stations — and it watches itself. This is a month of seeded history: W-16's lens degraded, and its confidence chart crossed the control limit before a single false alarm." |
| 3:40 | Close W-15's engine window | "A quality gate that silently stops checking is worse than none." W-15's lamp in Node-RED goes AMBER in about two seconds — set by the PLC, not the engine — and `/fleet` marks it offline. |
| 4:00 | Harness: switch profile to the door trim | "Your problem statement asked three questions. This is the second one." A trim panel goes on over an open clip: `PRECEDENCE`, red. "Same binary. A new variant is a recompile; a new assembly is a new profile." |
| 4:20 | Slide: business | "We verify sequence, not torque — the nutrunner keeps that job. We sit beside it." |
| 4:40 | Close | "The CAD model is the single source of truth — for what the line should do, and for what the camera should see." |

### What this does not do yet

State these before the judges find them. It converts weaknesses into a roadmap.

1. **It does not measure torque.** Tier 2 reads the nutrunner's per-rundown result over OPC-UA and
   joins it to the bolt identity.
2. **The torque-mark class is validated on synthetic data only.** Real validation needs forty photos
   of a real hub with witness marks.
3. **It has run on one GPU class.** Edge deployment on an industrial PC or Jetson is designed (the
   Linux build exists for the sanitizers) but not demonstrated.
4. **The drift limits are calibrated on simulated data.** Real health data is autocorrelated, so
   the false-alarm rate is re-measured, and L recalibrated, after two weeks on a real line.
5. **Two of the four profiles stop at the procedure level.** Windshield and enclosure are compiled
   and verified by both interpreters on scripted event streams; their perception models are not built.
6. **The occlusion test is designed, not run.** The resolution, framing and facing checks are tested;
   ray-casting against the CAD is the C++ compiler's job and has not been exercised.

---

## 23. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ONNX Runtime silently falls back to CPU | **High** on CUDA 13 | Detector at ~3 fps, no error | Download the `gpu_cuda13` asset; startup check fails loudly if `CUDAExecutionProvider` is missing (Section 29) |
| cuDNN DLL not found | Medium | CUDA EP fails to load | `where cudnn64_9.dll` in the gate; add its directory to `PATH` |
| First inference takes ~1 s | **Certain** | Pollutes latency stats; first demo frame stalls | 20 warm-up inferences before the engine reports ready (Section 33) |
| Laptop GPU throttles | High | p99 doubles | AC power, clocks locked, cooling pad; `ThrottleDetection` warns if median inference exceeds 20 ms |
| Space in user path | **Certain** if ignored | vcpkg, CMake, Docker mounts fail with misleading errors | Everything under `C:\dev\torquesense` |
| vcpkg build takes an hour | Certain | Blocks C++ work | Start it in hour 0; binary caching makes it once |
| Windows Defender scans 4,000 renders | High | Rendering and builds 2–3× slower | Exclusion for `C:\dev\torquesense` (Section 27) |
| WSL2 takes half the RAM | High | Pipeline pages to disk | `.wslconfig` cap at 4 GB |
| Node-RED cannot reach the soft PLC | Medium | Tag monitor blank | Connect Node-RED to `opc.tcp://host.docker.internal:4840` and leave the server URL at its default — the server echoes it back; allow `softplc.exe` on private networks (Section 28) |
| Florence-2 OOM beside the detector | Low | Sidecar crashes | ORT arena capped; torch memory fraction capped; circuit breaker keeps the engine running |
| Sim-to-real gap | Medium | Judge asks about real pixels | Report both mAP figures; state the torque-mark limit honestly |
| BlenderProc phase overruns | Medium | Compresses everything after | Render 1,000 frames first, train, then render the rest while integrating |
| Colab disconnects mid-training | Medium | Lost epochs | `save_period=5`, train to Drive, resume from `last.pt` |
| Four stations exceed GPU | Medium | Frame drops | 15 fps per station; fall back to three stations |
| Firewall prompt blocks port 4840 | Medium | OPC-UA unreachable from Docker | Allow `softplc.exe` on private networks during setup, not on stage |
| Projector washes out amber | Medium | States indistinguishable | Check tokens on the venue projector |
| Overriding the soft PLC's server URL | — | Server binds nothing and exits | Leave `serverUrls` at its default (Section 28; Appendix A, item 18) |
| Jenkins on port 8080 | Certain if defaulted | Collides with the engine's WebSocket | Jenkins mapped to `127.0.0.1:8081` |
| GitHub self-hosted runner on a public repo | — | A fork's pull request runs code on the laptop | Not used: Jenkins polls `main` only |
| Jenkins plus the platform exceed the WSL cap | Medium | Containers OOM-killed | Jenkins is a compose profile, run in CI mode with the platform stopped (Section 25) |
| Running the Jenkins agent as Administrator to lock clocks | Medium | Every build step gets admin rights | Agent runs as your user; the pipeline records clocks and power instead |
| False drift tickets | Medium | Engineers stop trusting `/fleet` | Limits sized by simulation for the real baseline length; every ticket's outcome logged; L recalibrated after two weeks |
| Chart baselines lost on an API restart | Certain without persistence | 24 h of learning lost | State persisted every bucket; round-trip tested |
| AMBER during engine start-up | Certain | Looks like a fault on stage | Intended — not ready is not OK. Start the stack before the audience is seated |
| An inspection point too small to judge | Certain on some rigs | Unreliable verdicts | The compiler refuses the profile and names the points and the shortfall (I17) |
| A mis-authored manifest | Medium | A wrong order or a missing step enforced at a station | Manifests reviewed in pull requests; schema validation; the scenario gate (I14) |
| A new part family with no training data | Certain for new assemblies | No classifier for its states | Florence-2 cold start, advisory only: the station cannot write to the PLC until the family passes the gate |
| The wrong variant's procedure on a mixed-model line | Medium | False alarms or missed steps | Variant code read at cycle start; an unknown code holds the station AMBER |
| The operator's body hides a point for a whole step | Medium | The step cannot be seen | A second camera position, or corroborating plant evidence (Section 10) |
| Two cameras share one gigabit link | Medium | Dropped packets, resends, dropped frames | One camera per NIC port; Bayer on the wire (581 Mbit/s at 30 fps); resends checked every bucket (Section 8) |
| A camera loses its settings after a power cut or swap | Medium | Verdicts degrade and look like a model fault | Settings saved to a user set and read back at start; a mismatch holds the station AMBER |
| A rolling-shutter webcam used for validation | Medium | Moving hands skew; numbers do not transfer | `UvcSource` is development only; validation figures come from rendered video or the global-shutter camera |
| The pylon SDK's licence | Low | Cannot ship it inside an image | Read Basler's licence before building the production image; `IFrameSource` keeps the engine buildable without it |
| The GPU Operator's supported OS or Kubernetes list moves | Medium | A station cannot be upgraded on schedule | Check the release's platform-support page before each upgrade; canary one station first (Section 32) |

### Turning this into a portfolio artifact

1. **`docs/VALIDATION.md`** — both mAP figures with the sets described, latency histograms at
   p50/p99/p99.9 with the clock defined, the scenario test matrix, and the conditions of measurement.
   Almost nobody writes this file, which is exactly why it is the strongest hiring signal in the repo.
2. **A green CI badge** over the sanitizer matrix. "TSan-clean lock-free ring buffer" is a claim; a
   badge is evidence.
3. **A four-minute engineering walkthrough**, not the pitch: why the alert thread exists, why
   TensorRT was removed, why the narrative is deterministic, why I9 is a PSNR bound and not byte
   equality.

---

# Part V — Setup and operations, on HSGPC

## 24. Setup roadmap

Six phases, five gates. Do not start a phase until the gate before it passes — every gate here
exists because skipping it produces a failure that surfaces hours later looking like a different
problem.

```mermaid
flowchart LR
  P1["1 · Prepare machine<br/>C:/dev path · driver r580+<br/>long paths · Defender exclusion<br/>WSL2 + Docker · .wslconfig"] --> G1{"GPU ok?<br/>nvidia-smi · cc 8.6"}
  G1 -- "fail" --> P1
  G1 -- "pass" --> P2["2 · Install<br/>Build Tools · vcpkg features<br/>cuDNN 9 cu13 · ORT gpu_cuda13<br/>Python 3.12 venv · torch cu130<br/>Go · Temurin 21"]
  P2 --> G2{"CUDA EP live?<br/>ort_smoke.exe · torch.cuda"}
  G2 -- "fail" --> P2
  G2 -- "pass" --> P3["3 · Configure<br/>.env · compose · Garage layout<br/>station_w14.yaml · soft PLC hostname"]
  P3 --> P4["4 · Integrate<br/>contracts → types · shared ring<br/>sidecar · OPC-UA · WebSocket"]
  P4 --> G3{"Hour-one contract green?"}
  G3 -- "fail" --> P4
  G3 -- "pass" --> P5["5 · Test<br/>GoogleTest · WSL2 TSan/ASan · go test<br/>latency gate · pytest · Playwright<br/>Jenkins GPU run · kind smoke in CI"]
  P5 --> G4{"All gates pass?"}
  G4 -- "fail" --> P5
  G4 -- "pass" --> P6["6 · Deploy<br/>freeze model · tag · rehearse<br/>fallback capture · Wi-Fi off"]
  P6 --> T0(["Tier 0 — laptop demo"])
  P6 --> T1(["Tier 1 — public replay link"])
  P6 -.-> T2(["Tier 2 — station edge · pitched"])
  classDef gate fill:#FBF3E6,stroke:#B26B00;
  classDef key fill:#E6F1F2,stroke:#0E7C86;
  class G1,G2,G3,G4 gate;
  class P6,T0 key;
```

*Figure 5 — Setup roadmap. Gate G2 is the one specific to your CUDA 13.2 installation: it proves
the ONNX Runtime CUDA execution provider actually loads and runs on the GPU, because the most likely
failure on this machine is a build that silently runs on the CPU instead.*

---

## 25. Your machine — compatibility verdict

Every component clears its requirement. Three need action before you start; none needs an upgrade.

| Component | Yours | Requirement | Verdict |
|---|---|---|---|
| OS | Windows 11 Home 26H2, build 26300 | Docker Desktop on Home uses the WSL2 backend; WSL2 available | ✅ **pass** |
| CPU | i7-11800H, 8C/16T, AVX2 + AVX-512 | Pipeline needs 4 free physical cores | ✅ **pass** cores 4–7 pinned |
| RAM | 16 GB (15.7 usable) | Develop mode ~12 GB | ⚠️ **tight** see memory plan |
| GPU | RTX 3060 Laptop, 6 GB, Ampere cc 8.6 | CUDA 13 supports Ampere | ✅ **pass** VRAM budgeted in §12 |
| NVIDIA driver | not shown | CUDA 13.x requires **r580 or later** | ⚠️ **check** first |
| CUDA Toolkit | 13.2, V13.2.78 | ONNX Runtime CUDA 13 builds need ≥ 13.0 | ✅ **keep** do not downgrade |
| Python | 3.12.0 | PyTorch cu130 wheels support 3.12 | ✅ **pass** upgrade the patch |
| Java | not installed | Jenkins inbound agent needs Java 17 or later | ⚠️ **install** Temurin 21 |
| Go | not installed | Fleet agent's module declares Go 1.24 or later | ⚠️ **install** current Go |
| kubectl, kind | ships with Docker Desktop / not needed | the kind smoke test runs in GitHub Actions, not on the laptop | ✅ **nothing to do** |
| Storage | ~356 GB free | vcpkg ~15 GB · renders ~6 GB · Docker ~5 GB · Florence ~1 GB | ✅ **pass** |
| Camera | none needed | the demo replays rendered video; any USB webcam works for `UvcSource` in development | ✅ **optional** |
| User path | `C:\Users\Gurjas Gandhi` | — | ❌ **move** the space breaks tools |

### Five things specific to this machine

**1 · The space in your user path will bite you.** vcpkg, CMake, some Docker bind mounts, and shell
scripts mishandle `C:\Users\Gurjas Gandhi`. Put the project at `C:\dev\torquesense`, the vcpkg
binary cache at `C:\dev\vcpkg-cache`, BlenderProc's Blender at `C:\dev\tools\blender`, and the
Hugging Face cache at `C:\dev\hf-cache`. Every tool that would default into your home directory gets
an explicit path in Section 28. This costs nothing now and an afternoon later.

**2 · Keep CUDA 13.2 — and do not use TensorRT.** The ecosystem supports CUDA 13 now: ONNX Runtime
ships stable CUDA 13 builds from 1.27, and PyTorch publishes cu130 wheels from 2.11. Downgrading
buys nothing — CUDA 13's improvements target Blackwell, and your Ampere GPU runs identically on
either branch. TensorRT, however, names CUDA **13.2 Update 1** specifically in its support matrix,
and chasing that pin costs an evening to save five milliseconds against a 33 ms frame budget. The
ONNX Runtime CUDA execution provider runs YOLO11n in ~14 ms, which is enough.

> **The most likely install failure on this machine.** The CUDA 13 ONNX Runtime asset is named
> `onnxruntime-win-x64-gpu_cuda13-<ver>.zip`. The similarly named `…-gpu-<ver>.zip` is the CUDA 12
> build. It loads, cannot find the CUDA 12 runtime, and the detector runs on the CPU. Gate G2 exists
> to catch this.

**3 · Python 3.12.0 is the first 3.12 release.** It works, but install the latest 3.12 patch for
bug and security fixes. Same minor version, so every wheel in this document still applies. Do not
jump to 3.13 or 3.14 mid-project.

**4 · This is a hybrid-graphics laptop — which helps.** The Intel UHD drives the display, so the
desktop, the browser and the projector output do not consume the RTX 3060's VRAM. Nearly all 6 GB
is available for compute. What matters is power: laptop GPUs throttle heavily on battery and when
hot. Every latency number in this document assumes AC power, Windows power mode **Best performance**,
and GPU clocks locked (Section 28).

**5 · Windows 11 Home has no Group Policy editor, and does not need one.** Everything this build
configures — long paths, Defender exclusions, power mode — is done from PowerShell or Settings.

### Memory plan — four operating modes

16 GB holds the whole stack, but not the whole stack *and* a render or a local training run. Run
one mode at a time. Figures are estimates; confirm in Task Manager.

| Mode | Running | Stopped | Approx. RAM |
|---|---|---|---|
| **Render** | BlenderProc, one terminal | Docker, sidecar, browser, IDE | Blender 4–6 GB |
| **Develop** | VS Code, engine (debug), sidecar, Docker at 4 GB cap, Vite, one browser | — | 12–13 GB |
| **Demo** | Engine (release), soft PLC, sidecar, Docker, Vite, browser with two tabs | IDE, chat apps, OneDrive sync | ~10 GB |
| **Line** | Demo, with four engine instances and four fleet agents | — | ~11.5 GB |
| **CI** | Jenkins controller (Docker, 1 GB cap), Jenkins agent, one build-and-test run | Platform containers, browser, IDE | ~7–9 GB while compiling |

> **The single biggest saving: Docker's cap.** Without `.wslconfig`, WSL2 may claim up to half your
> RAM, and Docker Desktop will hold it even when containers are idle. Capping WSL2 at 4 GB is the
> difference between Develop mode fitting and Windows paging the engine to disk mid-benchmark.

---

## 26. Compute — what the Colab T4 is for

**Training YOLO11n. Nothing else.** Not inference, not the vision-language model, not rendering.

### Why train on Colab when the 3060 could do it

The 3060 *can* train YOLO11n — at 640 px and batch 16 it fits comfortably in 6 GB. An earlier draft
said the laptop GPU would thrash; that was wrong, and it is corrected in Appendix A. The real
reasons to use Colab are scheduling and heat, not capability:

| Reason | Detail |
|---|---|
| **Parallelism** | While Colab trains for two to three hours, the laptop renders scenario videos and you write C++. On one GPU those two jobs serialise. |
| **Thermals** | A multi-hour training run on a laptop holds the GPU at its thermal limit. You want the laptop cool and its clocks stable for latency measurement afterwards. |
| **Throughput is comparable** | A T4 and an RTX 3060 Laptop are in the same FP16 class for a model this small. Colab is not faster; it is *elsewhere*. |

If Colab is unavailable, train locally overnight with `batch=16, imgsz=640, workers=4` — same
result, same `ultralytics` command.

### Why the vision-language model does not run on Colab

An earlier version of this design served Qwen2.5-VL from a Colab T4 through a tunnel. Florence-2
replaced it precisely because it fits locally: a 230M-parameter model needs roughly 0.46 GB for
fp16 weights and about 1.2–1.5 GB as a whole process. Removing the tunnel removed the single
largest demo-day risk — a free Colab session idling out in the middle of a presentation.

### The Colab training notebook

```python
# training/colab_train.ipynb — run on a T4 runtime
!pip -q install ultralytics
from google.colab import drive; drive.mount('/content/drive')

ROOT = '/content/drive/MyDrive/torquesense'
!cp {ROOT}/synth_w14_s20260721.zip /content/ && unzip -q /content/synth_w14_s20260721.zip -d /content/data

from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='/content/data/dataset.yaml',
    epochs=100, imgsz=640, batch=32,        # T4 has 16 GB; 32 fits
    close_mosaic=10, patience=25,
    cache='disk',                            # decode JPEGs once, not every epoch
    project=f'{ROOT}/runs', name='w14',
    save_period=5,                           # survive a Colab disconnect
    seed=20260721, deterministic=True,
)

best = YOLO(f'{ROOT}/runs/w14/weights/best.pt')
best.export(format='onnx', opset=17, imgsz=640, simplify=True, dynamic=False)
```

If the session disconnects, resume with `YOLO(f'{ROOT}/runs/w14/weights/last.pt').train(resume=True)`.
Copy `best.onnx` to `C:\dev\torquesense\models\yolo11n_wheel.onnx`. The ONNX file is portable across
GPUs; there is no engine to rebuild on the laptop.

---

## 27. Installation

In this order. Each step's verification is the gate for the next. All blocks are PowerShell.

### 1 · Machine preparation — PowerShell as Administrator

```powershell
# Project root outside the user folder (the space in "Gurjas Gandhi" breaks tools)
New-Item -ItemType Directory -Force C:\dev\torquesense, C:\dev\tools, C:\dev\vcpkg-cache, C:\dev\hf-cache

# Long paths: vcpkg and node_modules exceed 260 characters
New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force

# Defender: real-time scanning of 4,000 renders and a vcpkg build costs hours
Add-MpPreference -ExclusionPath C:\dev

# WSL2 for Docker and the sanitizer builds
wsl --install -d Ubuntu-24.04      # skip if already installed
wsl --update

# GPU and driver — gate G1
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

Expect `NVIDIA GeForce RTX 3060 Laptop GPU`, a driver of **580.xx or higher**, and `8.6`. If the
driver is older, install the current Game Ready or Studio driver from NVIDIA before anything else. A
CUDA 13 toolkit on a pre-580 driver fails at runtime with errors that look like library problems.

Then set Windows **Settings → System → Power → Power mode → Best performance**.

### 2 · Toolchain — PowerShell as a normal user

```powershell
winget install --id Git.Git -e
winget install --id Kitware.CMake -e
winget install --id Ninja-build.Ninja -e
winget install --id OpenJS.NodeJS.LTS -e
winget install --id Docker.DockerDesktop -e
winget install --id FreeCAD.FreeCAD -e
winget upgrade --id Python.Python.3.12 -e          # 3.12.0 → latest 3.12 patch
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override `
  "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

git config --global core.longpaths true
node --version          # needs 22.12 or newer for current Vite
py -3.12 --version
```

Start Docker Desktop once, accept the WSL2 backend (the only option on Windows Home), then close it
until Section 28 caps its memory.

### 3 · CUDA 13 runtime libraries

You already have the CUDA 13.2 toolkit. Two more pieces:

**cuDNN 9.x for CUDA 13** — from <https://developer.nvidia.com/cudnn>, choose Windows, x86_64, and
the **CUDA 13** variant. Then find where the DLL landed and confirm it is on `PATH`:

```powershell
Get-ChildItem "C:\Program Files\NVIDIA\CUDNN" -Recurse -Filter cudnn64_9.dll | Select FullName
where.exe cudnn64_9.dll        # must print a path; if not, add that folder to the user PATH
where.exe cudart64_13.dll      # CUDA 13 runtime — from your toolkit
where.exe cublas64_13.dll
```

If `cudart64_13.dll` is not found, add the directory that contains it under `%CUDA_PATH%` to your
user `PATH`. Find it with `Get-ChildItem $env:CUDA_PATH -Recurse -Filter cudart64_13.dll`.

**ONNX Runtime, CUDA 13 build** — from <https://github.com/microsoft/onnxruntime/releases>, take the
newest release that has an asset named **`onnxruntime-win-x64-gpu_cuda13-<version>.zip`** (1.27 or
later):

```powershell
$ort = "1.27.0"   # replace with the newest release that ships a gpu_cuda13 asset
$zip = "onnxruntime-win-x64-gpu_cuda13-$ort.zip"
Invoke-WebRequest "https://github.com/microsoft/onnxruntime/releases/download/v$ort/$zip" -OutFile "$env:TEMP\$zip"
Expand-Archive "$env:TEMP\$zip" C:\dev\tools
Rename-Item "C:\dev\tools\onnxruntime-win-x64-gpu_cuda13-$ort" C:\dev\tools\onnxruntime
```

Not the `…-gpu-<version>.zip` asset. That one is the CUDA 12 build.

### 4 · C++ dependencies — vcpkg manifest with two feature sets

```powershell
git clone https://github.com/microsoft/vcpkg C:\dev\tools\vcpkg
C:\dev\tools\vcpkg\bootstrap-vcpkg.bat -disableMetrics
[Environment]::SetEnvironmentVariable("VCPKG_ROOT", "C:\dev\tools\vcpkg", "User")
[Environment]::SetEnvironmentVariable("VCPKG_DEFAULT_BINARY_CACHE", "C:\dev\vcpkg-cache", "User")
```

`vcpkg.json` splits dependencies so the sanitizer builds stay fast:

```json
{
  "name": "torquesense",
  "version-string": "0.4.1",
  "dependencies": ["gtest", "benchmark", "nlohmann-json", "spdlog", "picosha2"],
  "features": {
    "plant-io": {
      "description": "OPC-UA, WebSocket and HTTP: everything the headless Linux engine needs",
      "dependencies": ["open62541", "ixwebsocket", "cpp-httplib"]
    },
    "full": {
      "description": "Windows build: video, CAD kernel, Excel",
      "dependencies": [
        { "name": "opencv4", "features": ["ffmpeg", "jpeg"] },
        "opencascade",
        "openxlsx"
      ]
    }
  }
}
```

Then pin the port versions once, so a teammate, the CI runner and the engine container all build
exactly the same libraries:

```powershell
C:\dev\tools\vcpkg\vcpkg.exe x-update-baseline --add-initial-baseline
```

That writes a `builtin-baseline` commit into `vcpkg.json`. It is why the engine Dockerfile clones
vcpkg with full history: resolving a baseline needs it.

The `windows-release` preset builds `plant-io` and `full`; the `linux-headless` preset — the engine
container — builds only `plant-io`; the sanitizer presets build only the core set. vcpkg's current
`open62541` port is 1.4.21, the version the soft PLC's watchdog was tested against. **The first `full` build takes roughly an hour** — OpenCV with FFmpeg and Open CASCADE
compile from source. Start it in hour 0 (Section 21). The binary cache makes every later build
seconds.

### 5 · Python environment

```powershell
cd C:\dev\torquesense
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

python -m pip install --upgrade pip

# 1. torch FIRST, from the cu130 index. Installing anything that depends on torch before
#    this pulls the CPU-only wheel from PyPI and nothing will tell you.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. everything else
pip install transformers accelerate pillow `
            fastapi "uvicorn[standard]" httpx pydantic `
            boto3 av numpy opencv-python `
            ultralytics onnx onnxruntime `
            blenderproc dvc pytest

# Gate G2, Python half
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from transformers import Florence2ForConditionalGeneration; print('florence ok')"
```

Expect a `+cu130` torch version, `True`, and `NVIDIA GeForce RTX 3060 Laptop GPU`. If `is_available()`
prints `False`, a CPU wheel won: `pip uninstall -y torch torchvision` and repeat step 1.

The Florence-2 import needs a `transformers` release that includes the model natively. If that import
fails, `pip install -U transformers`. The `timm` and `einops` packages are **not** needed for the
native class — they were only required by the older remote-code checkpoint.

`onnxruntime` (CPU) in the venv is for `training/eval.py` sanity checks only. The engine uses the
C++ CUDA 13 build from step 3.

### 6 · BlenderProc and Blender

```powershell
blenderproc quickstart --blender-install-path C:\dev\tools\blender
```

BlenderProc downloads its own Blender into the path you give it. Without `--blender-install-path`
it installs under your home directory, which contains the space. The quickstart renders one test
scene; open the output to confirm it works.

### 7 · Front end

```powershell
cd C:\dev\torquesense
npm create vite@latest web -- --template react-ts
cd web
npm install
npm install recharts lucide-react react-router-dom `
            @fontsource/ibm-plex-sans @fontsource/ibm-plex-mono
npm install -D tailwindcss @tailwindcss/vite vitest @playwright/test `
               quicktype json-schema-to-typescript
npx shadcn@latest init
npx playwright install chromium
npm run dev
```

If `npm install` warns that install scripts were skipped and `npm run dev` then fails with an
esbuild or missing-binary error, approve the package using the command npm prints and reinstall.

### 8 · Docker images, pulled in advance

```powershell
docker pull timescale/timescaledb:latest-pg17
docker pull dxflrs/garage:v2.3.0
docker pull nodered/node-red:latest
docker pull heartexlabs/label-studio:latest
docker pull jenkins/jenkins:lts-jdk21
```

Pull them now, on good Wi-Fi. The demo runs offline (Section 21).

### 9 · Operations tooling

```powershell
winget install --id GoLang.Go -e
winget install --id EclipseAdoptium.Temurin.21.JDK -e
go version; java -version

# build the fleet agent (from the repository root; -C runs the build inside fleet-agent\)
go build -C fleet-agent -trimpath -ldflags="-s -w" -o ..\build\fleet-agent\fleet-agent.exe .
```

The agent's module asks for Go 1.24 or later, so any current Go release builds it. It uses only the
standard library, so `go build` downloads nothing.

---

## 28. Configuration

One `.env` as the single source of truth. Never committed.

### WSL2 memory cap

Create `%USERPROFILE%\.wslconfig` (this file *must* live in your home folder; the space is harmless
here because nothing parses it as a path):

```ini
[wsl2]
memory=4GB
processors=4
swap=2GB
```

Then `wsl --shutdown` and restart Docker Desktop. Four logical processors for WSL leaves twelve for
Windows — including the four the engine pins.

### Environment variables — user scope, set once

```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "C:\dev\hf-cache", "User")
[Environment]::SetEnvironmentVariable("ORT_ROOT", "C:\dev\tools\onnxruntime", "User")
# add ONNX Runtime's DLLs to PATH so torquesense.exe finds them
$p = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$p;C:\dev\tools\onnxruntime\lib", "User")
```

### .env — repository root

```bash
# ── generate secrets:  python -c "import secrets; print(secrets.token_hex(32))"

# database
POSTGRES_USER=torque
POSTGRES_PASSWORD=change-me
POSTGRES_DB=torquesense
DATABASE_URL=postgresql://torque:change-me@127.0.0.1:5432/torquesense

# object store — Garage
S3_ENDPOINT=http://127.0.0.1:3900
S3_REGION=garage
S3_BUCKET=torquesense
S3_ACCESS_KEY=from-garage-key-create
S3_SECRET_KEY=from-garage-key-create

# API
API_KEY=generate-me
API_URL=http://127.0.0.1:8000

# sidecar
SIDECAR_URL=http://127.0.0.1:8001
FLORENCE_MODEL=florence-community/Florence-2-base
TORCH_MEMORY_FRACTION=0.30
```

### docker-compose.yml

```yaml
name: torquesense
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg17     # pin to the exact tag after first pull
    env_file: .env
    ports: ["127.0.0.1:5432:5432"]
    volumes:
      - tsdata:/var/lib/postgresql/data
      - ./platform/migrations:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER}"]
      interval: 5s
      retries: 10
    mem_limit: 1g

  garage:
    image: dxflrs/garage:v2.3.0
    ports: ["127.0.0.1:3900:3900", "127.0.0.1:3903:3903"]
    volumes:
      - ./platform/garage/garage.toml:/etc/garage.toml:ro
      - garage-meta:/var/lib/garage/meta
      - garage-data:/var/lib/garage/data
    mem_limit: 256m

  api:
    build: ./api
    env_file: .env
    environment:
      DATABASE_URL: postgresql://torque:${POSTGRES_PASSWORD}@timescaledb:5432/torquesense
    depends_on:
      timescaledb: { condition: service_healthy }
    ports: ["127.0.0.1:8000:8000"]
    mem_limit: 384m

  node-red:
    image: nodered/node-red:latest                # pin after first pull
    ports: ["127.0.0.1:1880:1880"]
    volumes: ["./softplc/node-red:/data"]
    extra_hosts: ["host.docker.internal:host-gateway"]
    mem_limit: 256m

  jenkins:
    image: jenkins/jenkins:lts-jdk21
    profiles: ["ci"]                              # not started by a plain `docker compose up`
    ports: ["127.0.0.1:8081:8080"]                # 8080 is the engine's WebSocket
    environment:
      JAVA_OPTS: "-Xmx512m"
    volumes: ["jenkins-home:/var/jenkins_home"]
    mem_limit: 1g

volumes: { tsdata: {}, garage-meta: {}, garage-data: {}, jenkins-home: {} }
```

Limits sum to about 1.9 GB, comfortably inside the 4 GB WSL cap. Jenkins adds 1 GB but sits in the
`ci` profile: it starts only with `docker compose --profile ci up -d jenkins`, in CI mode, with the
demo platform stopped. After the first successful
`docker compose up`, run `docker compose images` and replace every `latest` with the exact tag shown.
Unpinned images are how a working demo breaks the night before.

### platform/garage/garage.toml

Garage reads its secrets from this file, so commit only `garage.toml.example`.

```toml
metadata_dir       = "/var/lib/garage/meta"
data_dir           = "/var/lib/garage/data"
db_engine          = "sqlite"
replication_factor = 1

rpc_bind_addr   = "[::]:3901"
rpc_public_addr = "127.0.0.1:3901"
rpc_secret      = "PASTE_64_HEX_CHARS"

[s3_api]
s3_region     = "garage"
api_bind_addr = "[::]:3900"
root_domain   = ".s3.garage.localhost"

[admin]
api_bind_addr = "[::]:3903"
admin_token   = "PASTE_TOKEN"
metrics_token = "PASTE_TOKEN"
```

Initialise the single-node layout once. The Garage image has no shell, so call the binary directly:

```powershell
docker compose up -d
docker compose exec garage /garage status                        # copy the node ID
docker compose exec garage /garage layout assign -z dc1 -c 20G <NODE_ID>
docker compose exec garage /garage layout apply --version 1
docker compose exec garage /garage bucket create torquesense
docker compose exec garage /garage key create torquesense-app    # → S3_ACCESS_KEY / S3_SECRET_KEY
docker compose exec garage /garage bucket allow --read --write --owner torquesense --key torquesense-app
```

### The soft PLC — leave its server URL at the default

OPC-UA clients first ask a server for its endpoints, then connect. v4.0 of this document warned that
the soft PLC would advertise the machine name `HSGPC`, which a Docker container cannot resolve, and
told you to override `serverUrls` with `host.docker.internal`. Both halves were wrong. Tested against
open62541 1.4.21, the version vcpkg ships:

- With the default configuration the server **echoes the URL each client used** in its GetEndpoints
  reply. With the server running under a hostname that does not resolve, asyncua, the open62541
  client and node-opcua — the library under Node-RED's OPC-UA nodes — all connected, read and wrote.
- In 1.4, `serverUrls` also decides **which address the server binds to**. A name that is not a local
  interface — `host.docker.internal`, or a Kubernetes Service name — leaves the server with no socket.
  It logs "The server has no server socket" and exits.

So the soft PLC calls `UA_ServerConfig_setMinimal(cfg, 4840, NULL)` and nothing more. Clients connect
with whatever name reaches it: `localhost` from the engine, `host.docker.internal` from the Node-RED
container, `softplc` inside Kubernetes.

When Windows Firewall prompts for `softplc.exe`, allow **private** networks — the Node-RED container
reaches it through Docker's host gateway. Do this during setup, never for the first time on stage.

### config/station_w14.yaml

```yaml
station:
  id: W-14
  name: "Wheel & Tyre Fitment, Line 2"
  takt_seconds: 58

source:
  kind: file                                  # file | uvc | pylon  (Section 8)
  path: data/scenarios/cycle_correct_01.mp4
  # kind: uvc     → device: 0                   # a USB webcam, development only
  # kind: pylon   → serial: "40123456"          # production builds; settings from cameras.yaml
  fps: 30                                     # 15 in four-station mode
  process_width: 960
  process_height: 540

threads:                                      # logical CPUs: one per physical core 4..7
  capture: 8
  inference: 10
  verify: 12
  alert: 14
  priority_class: high                        # never realtime

perception:
  model_path: models/yolo11n_wheel.onnx
  model_manifest: models/manifest.json        # SHA-256 checked before load (I12)
  execution_provider: cuda
  gpu_mem_limit_mb: 1024                      # 400 in four-station mode
  warmup_runs: 20
  input_size: 640
  conf_commit: 0.65                           # at or above: commit
  conf_floor: 0.35                            # 0.35–0.65: ambiguous → adjudicate
  nms_iou: 0.45
  debounce_frames: 2
  crop_pool_size: 4                           # full-resolution frames kept for inspection-point crops
  crop_px: 96

profile:
  dir: profiles/wheel-fitment-rl              # procedure.json, coverage.json, narratives.yaml
  variant_tag: Station.W14.VariantCode        # selects a variant's procedure at cycle start
  release_tag: Station.W14.ReleaseRequest     # the release gate
  advisory_only: false                        # true during a cold start: verdicts shown, PLC never written

alert:
  backend: opcua
  endpoint: opc.tcp://host.docker.internal:4840
  namespace: 2
  prefix: Station.W14

clip_ring:
  name: Local\torquesense_W-14
  slots: 180
  slot_bytes: 262144
  jpeg_quality: 80

sidecar:
  url: http://127.0.0.1:8001
  timeout_ms: 1500
  failure_threshold: 3
  circuit_open_seconds: 60
  queue_depth: 16

io:
  websocket_port: 8080                        # W-15 adds 100 to all three ports, W-16 adds 200, …
  mjpeg_port: 8090
  control_port: 8095
  api_url: http://127.0.0.1:8000
  event_spool: data/spool/W-14.jsonl          # events survive an API outage

health:
  bucket_s: 300                               # 5 in demo mode
  heartbeat_ms: 500
  heartbeat_node: Station.W14.Heartbeat
```

### CMakePresets.json

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "base", "hidden": true, "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
      "cacheVariables": { "CMAKE_EXPORT_COMPILE_COMMANDS": "ON" }
    },
    {
      "name": "windows-release", "inherits": "base",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows" },
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CMAKE_C_COMPILER": "cl", "CMAKE_CXX_COMPILER": "cl",
        "VCPKG_MANIFEST_FEATURES": "plant-io;full",
        "SUNVERIFY_WITH_CUDA": "ON",
        "ONNXRUNTIME_ROOT": "$env{ORT_ROOT}"
      }
    },
    {
      "name": "windows-ci", "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_C_COMPILER": "cl", "CMAKE_CXX_COMPILER": "cl",
        "SUNVERIFY_WITH_CUDA": "OFF"
      }
    },
    {
      "name": "linux-asan", "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug", "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_CXX_FLAGS": "-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1",
        "SUNVERIFY_WITH_CUDA": "OFF"
      }
    },
    {
      "name": "linux-tsan", "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug", "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_CXX_FLAGS": "-fsanitize=thread -g -O1",
        "SUNVERIFY_WITH_CUDA": "OFF"
      }
    },
    {
      "name": "linux-headless", "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "VCPKG_MANIFEST_FEATURES": "plant-io",
        "SUNVERIFY_WITH_CUDA": "OFF",
        "SUNVERIFY_HEADLESS": "ON"
      }
    }
  ],
  "buildPresets": [
    { "name": "windows-release", "configurePreset": "windows-release" },
    { "name": "windows-ci",      "configurePreset": "windows-ci" },
    { "name": "linux-asan",      "configurePreset": "linux-asan" },
    { "name": "linux-tsan",      "configurePreset": "linux-tsan" },
    { "name": "linux-headless",  "configurePreset": "linux-headless" }
  ],
  "testPresets": [
    { "name": "windows-release", "configurePreset": "windows-release", "output": { "outputOnFailure": true } },
    { "name": "linux-tsan",      "configurePreset": "linux-tsan",      "output": { "outputOnFailure": true } }
  ]
}
```

Build from **Developer PowerShell for VS** so `cl.exe` is on `PATH`. The Ninja generator means the
Visual Studio version never appears in CMake. The `linux-*` presets run identically in WSL2 and in
GitHub Actions — renamed from `wsl-*` in earlier drafts because they are not WSL-specific.

### Jenkins — controller in Docker, agent on Windows

The controller is a container; the work happens on an agent on the laptop itself, because that is
where the GPU, MSVC, the vcpkg cache and the CUDA DLLs are.

```powershell
docker compose --profile ci up -d jenkins
docker compose exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

Open `http://localhost:8081`, install the suggested plugins, create your admin user, then:

1. **Manage Jenkins → System → Jenkins URL:** `http://localhost:8081/`.
2. **Manage Jenkins → Nodes → New node** `hsgpc-gpu`: permanent agent, label `hsgpc-gpu`, one
   executor, remote root `C:\dev\jenkins-agent`, launched by connecting it to the controller, with
   **Use WebSocket** ticked. Save the secret it shows into `C:\dev\jenkins-agent\secret`.
3. **Manage Jenkins → Credentials:** add a *Secret text* with ID `torquesense-api-key`.
4. **New Item → Pipeline** `torquesense-gpu`: *Pipeline script from SCM*, Git, your repository URL,
   branch `*/main`, script path `Jenkinsfile`.

Start the agent as **your own user** — never as Administrator, and not as a Windows service under
LocalSystem. It must inherit your `PATH` (ONNX Runtime and cuDNN DLLs), `VCPKG_ROOT`, the vcpkg
binary cache and `HF_HOME`, and a build step should never hold admin rights:

```powershell
# scripts\start-jenkins-agent.ps1
New-Item -ItemType Directory -Force C:\dev\jenkins-agent | Out-Null
Invoke-WebRequest http://localhost:8081/jnlpJars/agent.jar -OutFile C:\dev\jenkins-agent\agent.jar
java -jar C:\dev\jenkins-agent\agent.jar -url http://localhost:8081/ -name hsgpc-gpu `
     -secret "@C:\dev\jenkins-agent\secret" -webSocket -workDir C:\dev\jenkins-agent
```

Locking GPU clocks needs Administrator, so the pipeline does not lock them. It records the clocks,
temperature and power source in `conditions.json` beside every result instead, and marks the build
UNSTABLE on battery. The 150 ms budget is about twice the expected 70 ms, so only real throttling
fails it — which is a result worth having.

### GPU clock lock — before every measurement

```powershell
# PowerShell as Administrator
nvidia-smi -pm 1                                   # may report "not supported" on Windows; harmless
nvidia-smi --lock-gpu-clocks=1200,1200
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv
# after measuring:
nvidia-smi --reset-gpu-clocks
```

Persistence mode is a Linux feature and is often reported unsupported on Windows; the clock lock is
what matters. If `--lock-gpu-clocks` itself is refused by the laptop's driver, measure anyway with
clocks unlocked and report the clocks observed alongside the histogram — a disclosed condition is
still a valid measurement; an undisclosed one is not.

---

## 29. Integration

One rule governs everything here: **every process boundary is a schema, and every schema is
generated into code.** Hand-written structs on two sides of a boundary drift; generated ones fail
to compile.

### The contract, generated

```powershell
# contracts/ holds state, anomaly, adjudication, health and host_health schemas (*.schema.json)
npx quicktype -s schema contracts\*.schema.json -l c++ --namespace sunverify::contracts `
  --code-format with-struct --include-location global-include -o control-app\include\contracts.hpp
npx json2ts -i "contracts/*.schema.json" -o web\src\contracts\
```

`quicktype` emits C++ structs with nlohmann/json serialisers; `json-schema-to-typescript` emits the
dashboard's types. Run both from one script; CI fails if the generated files differ from what is
committed.

These structs are used **only on the io thread**. The hot path carries `StateVector`, a fixed-size
POD, and the io thread converts it to the contract type. JSON never touches a pinned thread.

### The shared clip ring — one layout, two languages

The engine writes JPEG frames; the sidecar reads them for adjudication crops and clip assembly. No
image ever crosses HTTP. The layout is fixed and documented, because a Python reader parses it with
`struct` offsets:

```
Offset     Size        Field
0          4           magic 'TSCR'
4          4           version = 1
8          4           slots = 180
12         4           slot_bytes = 262144
64         8           write_seq          (own cache line; producer-owned)
128        180 × 32    slot table:  seq u64 · ts_ns u64 · frame_seq u64 · len u32 · pad u32
8192       180 × 256K  payloads
Total      47,194,112 bytes  (~45 MiB)
```

Each slot is a **seqlock**. The writer marks the slot odd while writing and even when done; a reader
that sees the sequence change, or sees it odd, discards the copy and retries.

```cpp
// ipc/src/SharedClipRing.cpp — producer, called from the capture thread
bool SharedClipRing::publish(std::span<const std::byte> jpeg,
                             std::uint64_t ts_ns, std::uint64_t frame_seq) noexcept {
  if (jpeg.size() > kSlotBytes) return false;                 // never truncate, never block
  const std::uint64_t n = header_->write_seq.load(std::memory_order_relaxed);
  SlotMeta& m = slots_[n % kSlots];
  m.seq.store(2 * n + 1, std::memory_order_relaxed);          // odd: write in progress
  std::atomic_thread_fence(std::memory_order_release);        // odd seq visible before payload
  std::memcpy(payload(n % kSlots), jpeg.data(), jpeg.size());
  m.ts_ns = ts_ns; m.frame_seq = frame_seq;
  m.len   = static_cast<std::uint32_t>(jpeg.size());
  m.seq.store(2 * n + 2, std::memory_order_release);          // even: payload complete
  header_->write_seq.store(n + 1, std::memory_order_release); // publish
  return true;
}
```

The Windows backend maps it with `CreateFileMappingW` under the name `Local\torquesense_W-14`.
Python's standard `mmap` module can open a named Windows mapping directly through its `tagname`
argument — no extension module needed:

```python
# sidecar/clipring.py
import mmap, struct

SLOTS, SLOT_BYTES = 180, 262_144
TABLE, PAYLOAD, TOTAL = 128, 8192, 8192 + 180 * 262_144

class ClipRing:
    def __init__(self, station: str):
        self.m = mmap.mmap(-1, TOTAL, tagname=f"Local\\torquesense_{station}")
        assert self.m[0:4] == b"TSCR", "engine not running, or layout mismatch"

    def write_seq(self) -> int:
        return struct.unpack_from("<Q", self.m, 64)[0]

    def read(self, n: int) -> bytes | None:
        base = TABLE + (n % SLOTS) * 32
        for _ in range(3):                                    # seqlock retry
            s1, ts, fseq, ln = struct.unpack_from("<QQQI", self.m, base)
            if s1 & 1:                                        # writer mid-copy: retry
                continue
            if s1 != 2 * n + 2:                               # slot holds a different frame:
                return None                                   # not yet written, or overwritten
            off = PAYLOAD + (n % SLOTS) * SLOT_BYTES
            data = bytes(self.m[off: off + ln])
            if struct.unpack_from("<Q", self.m, base)[0] == s1:
                return data
        return None
```

> **An honest platform assumption.** The Python reader has no explicit memory fences. It is correct
> on x86-64, where loads are not reordered with other loads, and the seqlock re-check catches torn
> copies. On an ARM edge device the reader must move to C++ with acquire loads. This is noted in
> `docs/architecture.md` rather than discovered on a Jetson.

### ONNX Runtime — fail loudly, then warm up

```cpp
// perception/src/YoloOnnxDetector.cpp
std::expected<YoloOnnxDetector, Error>
YoloOnnxDetector::create(const Config& c, Ort::Env& env) {
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  so.SetIntraOpNumThreads(1);                      // the inference thread owns this session

  OrtCUDAProviderOptions cuda{};
  cuda.device_id              = 0;
  cuda.gpu_mem_limit          = std::size_t{c.gpu_mem_limit_mb} << 20;   // 1024 or 400
  cuda.arena_extend_strategy  = 1;                 // kSameAsRequested: no greedy doubling
  cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda.do_copy_in_default_stream = 1;
  try {
    so.AppendExecutionProvider_CUDA(cuda);
  } catch (const Ort::Exception& e) {
    // Typical on this machine: the CUDA 12 ORT asset looking for cudart64_12.dll,
    // or cudnn64_9.dll missing from PATH.
    spdlog::critical("CUDA EP failed to load: {} — use the gpu_cuda13 ONNX Runtime asset "
                     "and check `where cudnn64_9.dll`", e.what());
    return std::unexpected(Error::kCudaUnavailable);
  }

  YoloOnnxDetector d{env, c, so};
  // Warm-up: the first runs pay for cuDNN's exhaustive algorithm search and lazy
  // allocation — often ~1 s. They must never land in a latency histogram.
  for (int i = 0; i < c.warmup_runs; ++i) d.runDummy();
  const double median_ms = d.timeDummyRuns(10);
  if (median_ms > 40.0) {
    spdlog::critical("Median inference {:.1f} ms after warm-up — expected ~14 ms. "
                     "Nodes are probably running on CPU, or the GPU is throttled.", median_ms);
    return std::unexpected(Error::kTooSlow);
  }
  return d;
}
```

Earlier drafts checked `Ort::GetAvailableProviders()`. That only lists providers compiled into the
library, not whether they load — so it passes on exactly the misconfiguration it was meant to catch.
The exception plus a timed warm-up is the check that actually works (Appendix A).

### The sidecar — Florence-2 adjudication

```python
# sidecar/florence.py
import io, torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

class FlorenceAdjudicator:
    MODEL = "florence-community/Florence-2-base"
    TASK  = "<CAPTION_TO_PHRASE_GROUNDING>"

    def __init__(self, memory_fraction: float = 0.30):
        torch.cuda.set_per_process_memory_fraction(memory_fraction)   # share the GPU politely
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            self.MODEL, torch_dtype=torch.float16).to("cuda").eval()   # pinned; never device_map="auto"
        self.proc = AutoProcessor.from_pretrained(self.MODEL)
        self._warm()

    @torch.inference_mode()
    def ground(self, crop: Image.Image, phrase: str) -> list[list[float]]:
        crop = crop.convert("RGB")                                     # RGBA crops crash the processor
        x = self.proc(text=self.TASK + phrase, images=crop,
                      return_tensors="pt").to("cuda", torch.float16)   # casts pixel_values only
        ids = self.model.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"],
                                  max_new_tokens=128, num_beams=1, do_sample=False)
        text = self.proc.batch_decode(ids, skip_special_tokens=False)[0]
        out = self.proc.post_process_generation(text, task=self.TASK,
                                                image_size=(crop.width, crop.height))
        return out[self.TASK].get("bboxes", [])

    def bolt_is_torqued(self, crop: Image.Image, u: float, v: float, r: float) -> bool:
        """True if a grounded 'yellow marked bolt head' box is centred within r px of
        the CAD-projected bolt position (u, v) in crop coordinates."""
        for x0, y0, x1, y1 in self.ground(crop, "yellow marked bolt head"):
            if ((x0 + x1) / 2 - u) ** 2 + ((y0 + y1) / 2 - v) ** 2 <= r * r:
                return True
        return False

    def _warm(self):
        self.ground(Image.new("RGB", (256, 256)), "bolt")
```

`max_new_tokens=128, num_beams=1` rather than the tutorial's 1024 and 3: a bolt crop needs few
tokens, and greedy decoding roughly triples throughput. `.to("cuda")` rather than
`device_map="auto"`: with the detector already resident, automatic placement can silently spill
weights to the CPU and turn a 300 ms call into three seconds.

> **Validate the adjudicator, do not assume it.** Florence-2 was never trained on your yellow
> witness marks; grounding "yellow marked bolt head" is zero-shot. Render 200 deliberately ambiguous
> crops (motion blur, partial occlusion, glancing light) with known labels and report its agreement
> rate in `docs/VALIDATION.md`. If it agrees less often than the detector alone, the adjudicator is
> making things worse and should be turned off — which the config allows without code changes.

### The engine's sidecar client — the circuit breaker

`ReasoningClient` lives on the io thread, owns a bounded drop-oldest queue of depth 16, and opens a
circuit after three consecutive failures or timeouts for sixty seconds. While the circuit is open,
ambiguous bolts simply stay in their last committed state until the detector resolves them. Nothing
about an open circuit can produce an alert; it can only delay a state commit.

### Integration checkpoints

| Hour (of 40) | Checkpoint | Proves |
|---|---|---|
| 21 | Stub engine publishes canned `StateVector` JSON over the WebSocket into all four dashboard zones | every interface exists; integration is never a cliff |
| 26 | The real procedure engine drives the soft PLC; Node-RED lamp turns red on the adjacent-order scenario | engine → OPC-UA → container round trip |
| 30 | Sidecar reads the clip ring, writes an MP4 to Garage, anomaly row has a `clip_key` | shared memory across processes and languages |
| 33 | Ambiguous scenario triggers an adjudication; the anomaly card shows `adjudicated: true` | the CV + LLM loop is live |
| 38 | Four engines on `/line` | multi-station mode inside VRAM budget |
| 41 | Close an engine; its Node-RED lamp turns amber within 2.5 s | heartbeat → PLC watchdog, end to end |
| 45 | The seeded drifting station is amber on `/fleet`; the attention list names lens contamination | drift job and playbook, end to end |
| 47 | Stop the API for a minute; restart it; no fleet-agent record is missing or out of order | spool and in-order flush |
| 52 | `k8s-smoke` green in GitHub Actions | the images and manifests work in a real cluster |

---

## 30. Testing

Eighteen layers. Each answers a different question. The latency gate and the two-number accuracy report
outrank the rest, because they are the claims you will say out loud.

| Layer | Tool | Proves | Runs |
|---|---|---|---|
| Unit | GoogleTest | ring buffer, frame pools, interpreter, pattern solvers, narrator, profile loading | laptop, CI |
| **Reference model** | pytest, `reference/tests` | 29 tests: one interpreter on four profiles, every violation kind, pattern solvers, the resolution check, lens choice and link budget | laptop, CI |
| **Differential** | GoogleTest + reference | the C++ engine and the reference produce identical violation lists on every profile's scenarios (I16) | laptop, CI |
| Profile gate | `profile_gate.py` | coverage clean, scenarios exact, recall reported, canary signed (I14) | CI, per profile |
| Concurrency | GoogleTest under **TSan** | SPSC ring and clip ring are race-free | WSL2, CI (Linux) |
| Memory | GoogleTest under ASan + UBSan | no overflow, use-after-free, UB | WSL2, CI (Linux) |
| Allocation | counting allocator | zero allocations in `tick()` over 10,000 frames (I1) | laptop, CI |
| Scenario | 10 rendered videos + `.expected.json` | every fault caught; both edge cases **not** flagged | laptop |
| **Latency** | `latency_test` + HdrHistogram | evidence latency p99 ≤ 150 ms (I4) | laptop, GPU, AC power |
| **Accuracy** | `training/eval.py` | synthetic and real mAP, reported separately (I10) | laptop |
| Reproducibility | pytest | same seed → same labels, PSNR ≥ 45 dB (I9) | laptop |
| Sidecar + API | pytest, Florence mocked in CI | schema validation, circuit breaker, event spool replay | laptop, CI |
| End to end | Playwright | harness selects a fault → anomaly card appears; HMI floods red | laptop, CI (stub engine) |
| **Fail-visible** | soft PLC + asyncua stand-in engine | AMBER before the first beat, clear while beating, back within 2.5 s of the last beat (I13) | laptop, CI (Linux) |
| Drift detection | pytest | EWMA quiet in control, fast on a shift, direction-aware, floored, hysteretic, restart-safe | laptop, CI |
| Fleet agent | `go test -race` | GPU parsing, in-order spool flush, spool cap; cross-compiles for Windows and Linux | CI |
| Orchestration | kind + `tests/k8s/smoke.py` | engine → API → PLC inside a cluster; scaling the engine to zero turns the lamp amber | CI |
| GPU gates | Jenkins on HSGPC | latency, accuracy, scenarios and the nightly soak, with conditions recorded | laptop |

### The procedure engine — table-driven

```cpp
// procedure/tests/ProcedureEngineTest.cpp
// Protects: I6 (written before ProcedureEngine), I4 (evaluation must stay O(1) per frame).
struct Case { const char* name; std::vector<int> torque_order; std::optional<Violation> expect; };

class Sequences : public ::testing::TestWithParam<Case> {};

TEST_P(Sequences, MatchesExpectedVerdict) {
  const auto& c = GetParam();
  ProcedureEngine fsm{loadProfile("profiles/wheel-fitment-rl")};
  StateVectorBuilder sv;                       // starts with 5 seated bolts
  std::optional<Violation> got;
  for (int bolt : c.torque_order)
    for (int f = 0; f < 3; ++f)                // 3 frames: survives the 2-frame debounce
      if (auto v = fsm.evaluate(sv.torque(bolt).frame()); v.isViolation()) got = v.violation();
  EXPECT_EQ(got, c.expect) << c.name;
}

INSTANTIATE_TEST_SUITE_P(W14, Sequences, ::testing::Values(
  Case{"star",               {1,3,5,2,4}, std::nullopt},
  Case{"adjacent",           {1,2,3,4,5}, Violation{.observed = 2, .expected = 3}},
  Case{"resume_from_two",    {1,3,2},     Violation{.observed = 2, .expected = 5}},
  Case{"double_torque_ok",   {1,1,3,5,2,4}, std::nullopt}
  /* … 21 more, including timeouts, occlusion, second person … */
));
```

### Zero allocation — the counting allocator

```cpp
// perception/tests/support/CountingAllocator.cpp — linked only into the allocation test binary
#include <atomic>
#include <cstdlib>
#include <new>
std::atomic<std::size_t> g_allocs{0};
thread_local bool g_counting = false;

void* operator new(std::size_t n) {
  if (g_counting) g_allocs.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc{};
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
```

```cpp
TEST(Pipeline, NoAllocationInSteadyState) {      // I1
  Pipeline p{testConfig(), std::make_unique<ReplayDetector>("data/scenarios/cycle_correct_01.json")};
  for (int i = 0; i < 100; ++i) p.tick();          // warm-up: pools, reserves, first cycle
  g_allocs = 0; g_counting = true;
  for (int i = 0; i < 10'000; ++i) p.tick();
  g_counting = false;
  EXPECT_EQ(g_allocs.load(), 0u);
}
```

This binary is excluded from the ASan presets: ASan replaces `operator new` itself, and the two
replacements conflict. That is expected, and the exclusion is commented in CMake so nobody "fixes" it.

### Latency — the gate

```powershell
# after locking clocks (Section 28), on AC power
.\build\windows-release\tests\latency_test.exe `
    --scenario data\scenarios\fault_interrupt_resume.mp4 `
    --runs 100 --budget-evidence-p99-ms 150 --histogram docs\latency.hgrm
```

The test prints p50/p99/p99.9 for both engine and evidence latency, the GPU clocks and temperature
observed during the run, and exits non-zero if evidence p99 exceeds the budget. The `.hgrm` file goes
into `docs/` and its percentiles into `VALIDATION.md`.

### Reproducibility and accuracy

```python
# synth/tests/test_render_reproducible.py — I9
def test_same_seed_same_render(tmp_path):
    a, b = render(seed=7, frames=10, out=tmp_path/"a"), render(seed=7, frames=10, out=tmp_path/"b")
    assert a.labels == b.labels                         # labels must be identical
    for fa, fb in zip(a.images, b.images):
        assert psnr(fa, fb) >= 45.0                     # pixels: near-identical, not bit-exact
```

```powershell
python training\eval.py --model models\yolo11n_wheel.onnx `
    --synthetic data\synth\heldout --real data\real_val --out docs\accuracy.json
```

`eval.py` refuses to write a single combined figure. Its output has two blocks and a `limits`
field stating that `bolt_torqued` has no real-set coverage.

### Differential — the reference is the specification

Every profile's scenario fixtures are replayed as committed state and tag events through both the
Python reference interpreter and the C++ engine. Their violation lists must match exactly — kind,
step, inspection point and order. A disagreement fails CI and names the first event at which the two
diverge.

```bash
python -m tsref.replay profiles/door-trim-harness-fl/scenarios --emit build/events.jsonl
./build/linux-asan/tests/differential --profile profiles/door-trim-harness-fl \
    --events build/events.jsonl --expect-from-reference
```

Both tools are built in hours 20–26. The reference model's own suite ran while this revision was
written: 29 tests, all passing, after its first run caught a design error — unordered state families
(Section 6).

### Fail-visible — the heartbeat test

The soft PLC is tested on its own, with asyncua standing in for the engine. This is the test that
was run while writing this revision, against open62541 1.4.21:

```python
# tests/plc/test_heartbeat.py — the script form; the pytest version asserts ≤ 2.5 s
import asyncio, time
from asyncua import Client, ua

async def main(url="opc.tcp://127.0.0.1:4840"):
    async with Client(url=url) as c:
        node = lambda n: c.get_node(f"ns=2;s=Station.W14.{n}")
        print("before first beat -> Amber:", await node("Andon.Amber").read_value())
        hb = node("Heartbeat")
        for i in range(1, 8):                               # the engine, beating every 400 ms
            # open62541 rejects a write that carries a status code unless the node grants
            # StatusWrite; asyncua sends one by default, so send the bare value
            await hb.write_value(ua.DataValue(ua.Variant(i, ua.VariantType.UInt32),
                                              StatusCode=None, SourceTimestamp=None, ServerTimestamp=None))
            await asyncio.sleep(0.4)
        print("while beating     -> Amber:", await node("Andon.Amber").read_value())
        t0 = time.monotonic()                               # the engine dies here
        while not await node("Andon.Amber").read_value():
            await asyncio.sleep(0.05)
        print(f"engine stopped    -> Amber after {time.monotonic() - t0:.2f} s")

asyncio.run(main())
```

```
before first beat -> Amber: True
while beating     -> Amber: False
engine stopped    -> Amber after 1.79 s
```

The last beat was written 0.4 s before the clock started, so AMBER arrived 2.2 s after the last
heartbeat: the 2 s rule plus the 250 ms check interval. The engine's own client never trips the
status-code issue — `UA_Client_writeValueAttribute` sends a bare value, and was tested writing the
heartbeat the same way.

### Drift — the control-chart tests

Seven tests in `api/tests/test_ewma.py`, all with fixed seeds so they are deterministic: no ticket
over 5,000 in-control buckets; a 2σ drop flagged within 12 buckets; a rise on a "bad when falling"
chart ignored; a flat baseline floored rather than given zero spread; a drift episode that clears
with hysteresis and can fire again; a reset to baseline on redeploy; and a chart restored from its
JSON state that behaves identically to the original. The first draft used a two-hour baseline, and
the in-control test failed — Section 18 explains why that failure was the correct outcome.

### Scenario and soak

The ten scenario videos each have a sibling `.expected.json` naming the violation type and the
frame window in which it must fire. `ctest -R Scenario` runs all ten. A separate
`soak_test --minutes 10 --stations 4` runs the line mode, samples `nvidia-smi` every second, and
fails if VRAM exceeds 5 GB (I7) or any station drops below 14 fps.

### Continuous integration

```yaml
# .github/workflows/ci.yml — passes actionlint
name: ci
on: [push, pull_request]
permissions: { contents: read }

jobs:
  linux-sanitizers:
    runs-on: ubuntu-24.04
    strategy: { matrix: { preset: [linux-asan, linux-tsan] } }
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y clang ninja-build
      - uses: lukka/run-vcpkg@v11
      - run: cmake --preset ${{ matrix.preset }} && cmake --build --preset ${{ matrix.preset }}
      - run: ctest --test-dir build/${{ matrix.preset }} --output-on-failure --repeat until-fail:5

  windows-msvc:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ilammy/msvc-dev-cmd@v1
      - uses: lukka/run-vcpkg@v11
      - run: cmake --preset windows-ci && cmake --build --preset windows-ci
      - run: ctest --test-dir build/windows-ci --output-on-failure

  fleet-agent:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version-file: fleet-agent/go.mod }
      - working-directory: fleet-agent
        run: |
          test -z "$(gofmt -l .)"
          go vet ./...
          go test -race ./...
          GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o /dev/null .
          GOOS=linux   GOARCH=amd64 CGO_ENABLED=0 go build -o /dev/null .

  python:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r sidecar/requirements-ci.txt -r api/requirements.txt pytest
      - run: pytest sidecar/tests api/tests      # includes the EWMA drift tests; Florence is mocked

  web:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22 }
      - run: cd web && npm ci && npx vitest run
      - run: cd web && npx playwright install --with-deps chromium && npx playwright test

  contracts:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: ./scripts/gen-contracts.sh && git diff --exit-code

  k8s-smoke:
    needs: [linux-sanitizers, fleet-agent]
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build images
        run: |
          docker buildx build --load -t torquesense/engine:ci      -f docker/engine.Dockerfile      --cache-from type=gha,scope=engine --cache-to type=gha,mode=max,scope=engine .
          docker buildx build --load -t torquesense/api:ci         -f docker/api.Dockerfile         --cache-from type=gha,scope=api    --cache-to type=gha,mode=max,scope=api    .
          docker buildx build --load -t torquesense/fleet-agent:ci -f docker/fleet-agent.Dockerfile --cache-from type=gha,scope=agent --cache-to type=gha,mode=max,scope=agent .
      - uses: helm/kind-action@v1
        with: { cluster_name: ts }
      - run: for i in engine api fleet-agent; do kind load docker-image torquesense/$i:ci --name ts; done
      - run: kubectl apply -k deploy/k8s/overlays/kind
      - run: |
          kubectl -n torquesense rollout status statefulset/timescaledb --timeout=180s
          kubectl -n torquesense wait --for=condition=available deployment --all --timeout=300s
          kubectl -n torquesense rollout status daemonset/fleet-agent --timeout=120s
      - run: pip install httpx asyncua && python tests/k8s/smoke.py
      - if: failure()
        run: kubectl -n torquesense get pods -o wide && kubectl -n torquesense logs -l app.kubernetes.io/part-of=torquesense --tail=100 --prefix
```

No GPU runs in hosted CI. The `windows-ci` preset builds with `SUNVERIFY_WITH_CUDA=OFF` so MSVC
compiles every line except the ONNX Runtime adapter; the Linux jobs carry the sanitizers; the
`k8s-smoke` job builds the three images, side-loads them into a kind cluster, applies the `kind`
overlay and runs `tests/k8s/smoke.py`. The workflow above passes `actionlint` cleanly.

The manifests are checked before they ever reach a cluster: both overlays render with Kustomize and
pass `kubeconform -strict` against Kubernetes 1.36 (12 objects for `kind`, 11 for `edge`). The
Dockerfiles were written but not built while drafting this revision — there was no Docker daemon
available — so their first build is the `k8s-smoke` job's first run.

### The GPU pipeline — Jenkins on the laptop

What hosted CI cannot run, Jenkins runs on HSGPC: the build with CUDA, the scenario suite, the
latency gate, the accuracy report and, nightly, the four-station soak.

```groovy
// Jenkinsfile: the GPU gates, on HSGPC.
// GitHub Actions runs everything that needs no GPU. These stages need the RTX 3060, so they
// run on a Jenkins agent on the laptop. The controller polls GitHub; nothing on the laptop is
// exposed to the internet, and builds run only from main, never from fork pull requests.
pipeline {
  agent { label 'hsgpc-gpu' }
  options {
    timeout(time: 2, unit: 'HOURS')
    disableConcurrentBuilds()            // one GPU: concurrent builds would corrupt latency figures
    buildDiscarder(logRotator(numToKeepStr: '30'))
    timestamps()
  }
  triggers {
    pollSCM('H/15 * * * *')              // the laptop pulls; no inbound webhook
    cron('H 2 * * *')                    // nightly run adds the four-station soak
  }
  environment {
    VCVARS  = 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat'
    PY      = 'C:\\dev\\torquesense\\.venv\\Scripts\\python.exe'
    API_KEY = credentials('torquesense-api-key')
  }
  stages {
    stage('Conditions') {
      steps {
        // On battery the GPU clocks down and p99 roughly doubles. Record the conditions and
        // mark the build UNSTABLE rather than trust numbers taken on battery.
        catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
          powershell '''
            $b  = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1
            $ac = if ($b) { [bool]$b.PowerOnline } else { $true }
            $g  = nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,driver_version --format=csv,noheader
            @{ ac_power = $ac; gpu = "$g"; at = (Get-Date).ToString("o") } | ConvertTo-Json | Set-Content conditions.json
            if (-not $ac) { Write-Error "On battery: latency figures from this run are not comparable"; exit 3 }
          '''
        }
      }
    }
    stage('Data')  { steps { bat '"%PY%" -m dvc pull' } }   // scenarios, real set, model: local DVC store
    stage('Build') {
      steps { bat 'call "%VCVARS%" && cmake --preset windows-release && cmake --build --preset windows-release' }
    }
    stage('Unit and scenarios') {
      steps { bat 'call "%VCVARS%" && ctest --preset windows-release --output-junit "%WORKSPACE%\\ctest.xml"' }
      post  { always { junit testResults: 'ctest.xml' } }
    }
    stage('Latency gate') {
      steps {
        bat 'build\\windows-release\\tests\\latency_test.exe --scenario data\\scenarios\\fault_interrupt_resume.mp4 --runs 100 --budget-evidence-p99-ms 150 --histogram docs\\latency.hgrm'
      }
    }
    stage('Accuracy report') {
      steps {
        bat '"%PY%" training\\eval.py --model models\\yolo11n_wheel.onnx --synthetic data\\synth\\heldout --real data\\real_val --out docs\\accuracy.json'
      }
    }
    stage('Soak, four stations') {
      when  { triggeredBy 'TimerTrigger' }
      steps { bat 'build\\windows-release\\tests\\soak_test.exe --minutes 10 --stations 4 --vram-limit-gb 5 --min-fps 14' }
    }
  }
  post {
    always { archiveArtifacts artifacts: 'conditions.json, docs/latency.hgrm, docs/accuracy.json', allowEmptyArchive: true }
  }
}
```

The file passes `npm-groovy-lint` with no errors; the only warnings are long command lines.
Archiving `conditions.json` beside every histogram is what makes a latency number from this pipeline
reviewable: a figure taken at 65 °C on AC power is a different claim from one taken throttled.

**Why Jenkins, and not a GitHub self-hosted runner.** A self-hosted runner would also give hosted CI
the laptop's GPU. GitHub recommends against using them on public repositories, because a pull request
from a fork can run arbitrary code on the runner — here, on your own laptop. Jenkins polling `main`
never builds fork code and needs no inbound connection. It is also the CI server most likely to
already exist inside a manufacturing or supply-chain IT department.

---

## 31. Deployment

Three tiers. You ship Tier 0, you share Tier 1, you pitch Tier 2.

```mermaid
flowchart TB
  subgraph T0["Tier 0 — the demo · HSGPC · Wi-Fi off"]
    E0["torquesense.exe ×1 or ×4<br/>RTX 3060 · cores 4–7"] --> S0["softplc.exe :4840"]
    E0 --> SC0["sidecar · Florence-2"]
    E0 --> D0[("Docker · 4 GB cap<br/>TimescaleDB · Garage · API · Node-RED")]
    E0 --> W0["Vite :5173<br/>dashboard · Edge kiosk HMI"]
  end
  subgraph T1["Tier 1 — public showcase · synthetic data only"]
    V1["Vercel · static dashboard in replay mode<br/>recorded session + clips"]
    G1["GitHub · CI badge · VALIDATION.md"]
  end
  subgraph T2["Tier 2 — production · RKE2 · one GPU edge node per station"]
    CAM["Basler a2A1920-51gcBAS<br/>GigE · PoE · own NIC port"] --> E2["engine pod · Recreate<br/>pinned to its station's node"]
    AG2["fleet-agent DaemonSet<br/>one per node"] -.-> E2
    E2 -- "OPC-UA SignAndEncrypt · heartbeat" --> PLC2["Line PLC · Andon · conveyor<br/>heartbeat watchdog"]
    NR2["Nutrunner controller"] -- "per-rundown result" --> E2
    E2 --> H2[("Plant historian / MES<br/>in-plant object store")]
  end
  T0 ~~~ T1
  T1 ~~~ T2
  classDef key fill:#E6F1F2,stroke:#0E7C86;
  class E0,E2 key;
```

*Figure 6 — Deployment tiers. The same engine source runs in Tier 0 and Tier 2; the Linux build that
runs the sanitizers in Tier 0 is the build that would run on a station edge PC. Tier 2 adds the one
input the prototype lacks — the nutrunner's torque result — joined to the camera's bolt identity.*

### Tier 0 — demo runbook

```powershell
# scripts/start-demo.ps1 — run from C:\dev\torquesense in Developer PowerShell
param([int]$Stations = 1)
$ErrorActionPreference = "Stop"
$env:API_KEY = (Select-String -Path .env -Pattern '^API_KEY=(.*)$').Matches[0].Groups[1].Value

docker compose up -d                                    # the platform; Jenkins stays in profile "ci"
docker compose ps                                       # all four healthy or running

Start-Process .\build\windows-release\softplc\softplc.exe
.\.venv\Scripts\Activate.ps1
Start-Process powershell -ArgumentList '-NoExit','-Command',
  '.\.venv\Scripts\uvicorn.exe sidecar.main:app --host 127.0.0.1 --port 8001'
while (-not (Invoke-RestMethod http://127.0.0.1:8001/health -ErrorAction SilentlyContinue)) { Start-Sleep 1 }

$ids = @("W-14","W-15","W-16","W-17")[0..($Stations-1)]
for ($i = 0; $i -lt $ids.Count; $i++) {
  $id = $ids[$i]; $control = 8095 + 100 * $i            # each station's ports are offset by 100
  Start-Process .\build\windows-release\control-app\torquesense.exe `
    -ArgumentList "--config config\station_$($id.Replace('-','').ToLower()).yaml"
  Start-Process .\build\fleet-agent\fleet-agent.exe `
    -ArgumentList "-station $id -engine http://127.0.0.1:$control/health -api http://127.0.0.1:8000/v1/health/host -disk C:\dev"
}

Start-Process powershell -ArgumentList '-NoExit','-Command','cd web; npm run dev'
Start-Sleep 4
Start-Process msedge -ArgumentList '--kiosk','http://127.0.0.1:5173/hmi/W-14',
  '--edge-kiosk-type=fullscreen','--user-data-dir=C:\dev\tools\edge-kiosk'
Start-Process msedge -ArgumentList 'http://127.0.0.1:5173/'
```

Microsoft Edge is used for the kiosk because it ships with Windows 11 — one less install. The
separate `--user-data-dir` keeps the kiosk window from attaching to your normal browser session.
Press `Alt+F4` to leave kiosk mode.

### Tier 1 — public showcase

A judge or recruiter should be able to see the system without your laptop. Build the dashboard in
**replay mode**: the WebSocket client is swapped for a reader that plays back a recorded session.

```powershell
# record one session during a rehearsal
.\build\windows-release\control-app\torquesense.exe --config config\station_w14.yaml `
    --record web\public\replay\session.jsonl
# copy the referenced clips from Garage into web\public\replay\clips\

cd web
$env:VITE_REPLAY = "1"; npm run build
npx vercel --prod
```

The replay is labelled as one on screen. Only synthetic renders are published — there is no real
worker video to protect, which is a side benefit of the software-only build.

### Tier 2 — production on RKE2

Specified, and partly exercised. The base manifests and the `edge` overlay pass strict schema
validation, and the `kind` overlay runs in CI; the plant cluster itself is not built. This
subsection is the station's view; Section 32 names the platform around it.

| Choice | Why |
|---|---|
| RKE2, one cluster per plant: servers in the server room, each station's edge PC an agent | NVIDIA documents its GPU Operator on RKE2; a running station keeps verifying if the server room is unreachable |
| One Deployment per station, `strategy: Recreate` | The camera and GPU are exclusive. A rolling update would start the new pod while the old one still holds them |
| `nodeSelector: torquesense.io/station: w-14` | W-14's engine runs on W-14's PC and nowhere else |
| Fleet agent as a DaemonSet | Exactly one watcher per machine, by construction |
| Station config and compiled profile in a hashed ConfigMap; family models in the image | Recompiling a profile — a new CAD revision or a new manifest — changes the hash, which rolls that station's pod: the change deploys itself |
| Images from an in-plant registry, pinned tags | Nothing is pulled from the internet on a production line |

```yaml
# deploy/k8s/base/engine-w14.yaml (excerpt)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: engine-w14
  labels: { app.kubernetes.io/name: engine, torquesense.io/station: w-14 }
spec:
  replicas: 1
  strategy: { type: Recreate }          # never RollingUpdate: exclusive camera and GPU
  selector:
    matchLabels: { app.kubernetes.io/name: engine, torquesense.io/station: w-14 }
  template:
    metadata:
      labels: { app.kubernetes.io/name: engine, torquesense.io/station: w-14 }
    spec:
      securityContext: { runAsNonRoot: true, runAsUser: 10001 }
      containers:
        - name: engine
          image: torquesense/engine:ci
          args: ["--config", "/etc/torquesense/station.yaml"]
          ports: [{ name: ws, containerPort: 8080 }, { name: control, containerPort: 8095 }]
          readinessProbe: { httpGet: { path: /health, port: control }, periodSeconds: 2 }
          livenessProbe:
            httpGet: { path: /health, port: control }
            initialDelaySeconds: 20     # covers the ONNX warm-up
            periodSeconds: 5
          volumeMounts: [{ name: config, mountPath: /etc/torquesense, readOnly: true }]
      volumes:
        - name: config
          configMap: { name: station-w14 }   # Kustomize appends a content hash to this name
---
# deploy/k8s/overlays/edge/engine-gpu.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: engine-w14
spec:
  template:
    spec:
      nodeSelector: { torquesense.io/station: w-14 }
      containers:                         # no runtimeClassName: the GPU Operator injects GPUs via CDI
        - name: engine
          resources:
            limits: { nvidia.com/gpu: 1, memory: 2Gi }
```

The fleet agent's DaemonSet takes its node name from the downward API and, on GPU nodes, reads
the GPU from the node's own dcgm-exporter (`-gpu-source=dcgm`) through a Service with
`internalTrafficPolicy: Local`, without claiming a GPU. The node-local Service matters: a normal
Service would load-balance across every node's exporter and report some other station's GPU. With
no `nvidia-smi` in its container the agent's image is distroless *static*. The SM clock ceiling for
`sm_clock_ratio` is a flag set at commissioning from `nvidia-smi -q -d CLOCK`, because dcgm-exporter
does not export a maximum clock.

Deploying by hand, and the canary from Section 18 — on a plant cluster the same steps are pull
requests that Flux applies (Section 32):

```bash
kubectl label node edge-w14 torquesense.io/station=w-14
kubectl apply -k deploy/k8s/overlays/edge

# canary: one station first
kubectl -n torquesense set image deployment/engine-w15 engine=registry.plant.local/torquesense/engine:1.5.0
kubectl -n torquesense rollout status deployment/engine-w15

# after one shift on /fleet: promote to every station…
kubectl -n torquesense set image deployments -l app.kubernetes.io/name=engine \
    engine=registry.plant.local/torquesense/engine:1.5.0
# …or roll back the canary
kubectl -n torquesense rollout undo deployment/engine-w15
```

The rest of Tier 2 is unchanged from v4.0: OPC-UA with `SignAndEncrypt` and certificates to the
line PLC; the nutrunner controller's per-rundown torque and angle joined to the camera's bolt
identity, so a single record says *bolt 4, out of sequence, 62 Nm against 103 Nm specified*; events
and clips to the plant's own historian and object store, never to an external cloud. Its credibility
comes from Tiers 0 and 1 already sharing the engine, the contracts, the tag names — and now the
container images.

### Go-live checklist — the night before

| Check | How |
|---|---|
| Model frozen and tagged | `git tag demo-v1`; no retraining after this |
| Both mAP figures in `VALIDATION.md` match `docs/accuracy.json` | `python training\eval.py --check` |
| Latency gate passes on this laptop today | `latency_test` after locking clocks |
| All ten scenarios fire correctly, both edge cases silent | `ctest -R Scenario` |
| Florence weights cached; demo runs with Wi-Fi off | Disable Wi-Fi; run `start-demo.ps1 -Stations 4` |
| Docker image tags pinned | No `latest` left in `docker-compose.yml` |
| Firewall rule for `softplc.exe` exists | Node-RED lamps respond |
| Four-lug CAD recompiles to 1, 3, 2, 4 | Run the compiler once |
| Fail-visible works | Close one engine window; its lamp is amber within 2.5 s; restart it |
| Fleet charts are out of baseline | `/fleet` shows no dotted "learning" cells — the 24-minute demo-mode baseline was captured in rehearsal and persisted |
| Every fleet agent reporting | `/fleet` shows host health for each station |
| Fallback recording present and plays | `data\demo_fallback.mp4` open in a second window |
| Projector colour check | Red, amber, green distinguishable from the back row |
| No secrets committed | `git log -p \| Select-String -Pattern "password|secret|token"` returns nothing |
| AC adapter packed | Not a joke. On battery, p99 doubles. |

---

## 32. Production platform and hosting

Tier 2 in Section 31 says what runs on a station. This section says where everything runs once the
system leaves the laptop, and names one tool per layer. Every pick serves one of three needs the
plant has and the hackathon did not: **nothing pulled from the internet at runtime**, **nothing
inbound to the shop floor**, and **every change through a reviewed pull request**.

### Where everything runs

| Environment | What runs there | Hosted on |
|---|---|---|
| Development | everything, Tier 0 | HSGPC, this laptop |
| Continuous integration | builds, sanitizers, Go and Python tests, the kind smoke test; image signing | GitHub Actions |
| GPU gates | latency, accuracy, scenarios, nightly soak | Jenkins on HSGPC |
| Public showcase | the replay dashboard, Tier 1 | Vercel |
| Station | the engine and the fleet agent, beside its cameras | one industrial edge PC per station |
| Plant | the control plane, API, TimescaleDB, Garage, dashboard, registry, Git mirror | three servers in the plant's server room |

```mermaid
flowchart LR
  subgraph GH["GitHub — outside the plant"]
    PR["pull request"] --> CI["Actions: test · build<br/>cosign sign"] --> GHCR[("GHCR<br/>signed images")]
  end
  subgraph DC["Plant server room — RKE2 control plane on 3 servers"]
    HB[("Harbor<br/>pulls GHCR on a schedule · Trivy scan")]
    GIT[("Git mirror")]
    FX["Flux<br/>pulls Git · decrypts SOPS"]
    KY["Kyverno<br/>admits only cosign-signed images"]
    DB[("CloudNativePG<br/>TimescaleDB · 3 instances")]
    OBJ[("Garage<br/>3 nodes · replication 3")]
    WEB["API · dashboard<br/>Traefik · cert-manager · oauth2-proxy"]
    OBS["kube-prometheus-stack · Loki"]
  end
  subgraph LINE["Shop floor — station PCs join as RKE2 agents"]
    ST1["W-14 · RTX 2000 Ada<br/>engine pod · fleet agent<br/>GPU Operator · dcgm-exporter"]
    CAMS["Basler a2A1920-51gcBAS<br/>one NIC port each"]
    PLC["line PLC"]
  end
  GHCR -. "pull" .-> HB
  GIT --> FX --> KY
  HB --> ST1
  CAMS --> ST1
  ST1 -- "OPC-UA" --> PLC
  ST1 --> WEB --> DB
  WEB --> OBJ
  ST1 -. "metrics · logs" .-> OBS
  classDef key fill:#E6F1F2,stroke:#0E7C86;
  class FX,HB,ST1 key;
```

*Figure 7 — Production. The arrows that cross into the plant all start inside it: Harbor pulls
images, Flux pulls Git, station PCs connect out to the control plane. Nothing on the internet can
reach the shop floor, and a station keeps verifying if the server room drops off the network — a
running pod does not need its control plane.*

### The register — one pick per layer

| Layer | Pick | Why this one |
|---|---|---|
| Station edge PC | x86 industrial PC: one low-profile GPU slot, one slot for the camera card, 32 GB RAM, NVMe | Same architecture as the laptop, so the Linux build that runs the sanitizers is the build that ships |
| Station GPU | NVIDIA **RTX 2000 Ada Generation** — 16 GB, 70 W, low-profile | Fits an industrial chassis without extra power; Ada is supported by the CUDA 13 line already in use. A Jetson would mean an ARM build and a JetPack-pinned CUDA: a second toolchain for no gain here |
| Camera network | **4-port PoE+ GigE NIC**, one port per camera | The link budget allows one camera per gigabit port (Section 8) |
| Node OS | **Ubuntu Server 24.04 LTS** | Long support window; check it is on the supported-OS list of the GPU Operator release you deploy before ordering hardware |
| Provisioning | **Ansible** | One playbook: OS autoinstall, RKE2 join, NIC jumbo frames, socket buffer sysctls, node labels. A replaced PC is a playbook run, not a day |
| Kubernetes | **RKE2** 1.36, one cluster per plant: three servers in the server room, every station PC an agent | NVIDIA documents its GPU Operator for RKE2 explicitly; RKE2 1.36 ships containerd 2.3 and is security-hardened by default |
| GPUs | **NVIDIA GPU Operator** | Driver, container toolkit, device plugin and dcgm-exporter as one managed unit. From release 25.10 it injects GPUs through CDI, so pods need no runtime class |
| GitOps | **Flux** | Runs inside the cluster and *pulls* desired state, so nothing pushes into the plant. Decrypts SOPS natively |
| Secrets | **SOPS with age** | Secrets live encrypted in the same Git repository as the manifests and are reviewed the same way |
| Registry | **Harbor**, in the server room | Pull-based replication from GitHub Container Registry on a schedule, with Trivy scanning. The line never touches the internet to start a pod |
| Image signing | **cosign** in GitHub Actions, verified by **Kyverno** at admission | An image that was not built and signed by CI cannot run on a station |
| Ingress | **Traefik** — RKE2's default from 1.36 | ingress-nginx was retired upstream in March 2026; start new clusters on what is maintained |
| Certificates | **cert-manager** with the plant's internal CA | Every endpoint on HTTPS; renewal is automatic |
| Dashboard sign-in | **oauth2-proxy** in front of the dashboard, against the plant's OIDC identity provider | Engineers sign in with plant credentials; this system stores no passwords |
| Database | **CloudNativePG** with the TimescaleDB extension image, three instances | The operator manages failover and backups; TimescaleDB mounts as an extension image, which needs image volumes — stable and on by default from Kubernetes 1.36 |
| Database backups | CloudNativePG's Barman Cloud backups to Garage, with point-in-time recovery | Backups land in the same in-plant object store; restore is drilled quarterly |
| Object storage | **Garage**, three nodes, replication factor 3 | The prototype's store, now with no single point of failure |
| Platform observability | **kube-prometheus-stack** and **Loki**, with **Grafana Alloy** collecting from every node; dcgm-exporter for GPUs | Node and pod health, logs, GPU telemetry. Station health stays in TimescaleDB (Section 18) — this layer watches the platform, not the inspection |
| Showcase hosting | **Vercel**, replay mode | Unchanged from Tier 1 |

### How a release reaches a station

1. A pull request merges to `main`. GitHub Actions runs the full suite, builds the three images,
   signs them with cosign and pushes them to GHCR.
2. Harbor's scheduled pull replication brings the signed images into the plant and scans them.
3. A second pull request changes the image tag in `deploy/k8s/overlays/plant/` for **one station** —
   the canary. Flux applies it; Kyverno admits it only if the signature verifies; the station's pod
   restarts on the new version (`Recreate`).
4. After one shift on `/fleet`, judged against its peers (Section 18), a third pull request
   promotes the tag to every station — or reverts the second.

Profiles travel the same road. A recompiled profile is a pull request that changes the station's
hashed ConfigMap, and the station rolls itself. Every change to what a station checks has an author,
a reviewer and a commit — which is also the audit trail a quality system needs.

### The station pod on real hardware

Three things change from the kind overlay.

| Change | Why |
|---|---|
| `hostNetwork: true` on the engine pod | GigE Vision cameras are found by broadcast on the camera ports, which are host interfaces |
| The image adds the pylon runtime | `PylonSource` is compiled in production builds only |
| GPU via `nvidia.com/gpu: 1`; no `runtimeClassName` | the GPU Operator's CDI injection |

The fleet agent reads GPU figures from the node's own dcgm-exporter through a Service with
`internalTrafficPolicy: Local`, so each agent reports its own GPU. It no longer needs `nvidia-smi` in
its container, so its image is distroless *static*. Both overlays render with Kustomize and pass
`kubeconform -strict` against Kubernetes 1.36.

### Running it — day two

| Task | How |
|---|---|
| Kubernetes upgrades | RKE2 minor versions one at a time: server nodes first, then station agents during a planned stop |
| GPU driver upgrades | through the GPU Operator; canary one station first, exactly as for a model |
| Certificates | cert-manager renews; Alertmanager warns 30 days before any certificate the plant issued by hand expires |
| Backups | nightly base backup plus continuous WAL to Garage; a restore into a scratch namespace every quarter |
| A station PC dies | swap the hardware, run the Ansible playbook, label the node; its Deployment schedules back onto it |
| Remote support | the plant's existing remote-access path — this system adds none |

---

## 33. Caching strategy

**Cache computation, never authority.** Every key is content-addressed, so nothing can go stale and
there is no invalidation logic to get wrong.

### Where caching pays, and where it does not

| Cache | Value | Why |
|---|---|---|
| ONNX session + warm-up | ✅ **required** | Creating a session and running cuDNN's exhaustive algorithm search costs about a second. Done once at startup, never per frame. |
| Florence-2 model load | ✅ **required** | Loading weights takes several seconds. Load once at sidecar start, warm up once. |
| Rendered dataset by seed | ✅ **high** | Re-rendering 4,000 frames is 30 minutes. Keyed by seed, generator version, and glTF hash, so a CAD change produces a new folder automatically. |
| YOLO decode cache on Colab | ✅ **high** | `cache='disk'` decodes each JPEG once rather than every epoch. |
| Adjudication results | ⚠️ **modest** | Keyed by crop hash. Helps only when the same ambiguous frame is asked about twice — retries and scenario replays. Do not claim it improves throughput. |
| Verdicts, alerts, anomaly rows | ❌ **never** | These are facts about one cycle at one time. A cached verdict is a wrong verdict. |
| A cache server (Redis) | ❌ **cut** | Nothing here needs one. On 16 GB, another service is a cost with no measurable gain. |

### What is cached, where, for how long

| Layer | Key | Lifetime | Invalidated by |
|---|---|---|---|
| ONNX session, CUDA arena | process | until engine restart | restart after a new model |
| Florence-2 weights in VRAM | process | until sidecar restart | restart |
| Florence-2 weights on disk | `HF_HOME=C:\dev\hf-cache` | indefinite | manual |
| Adjudication results | SHA-256 of crop bytes · model id · phrase | 256-entry in-process LRU | eviction |
| Rendered frames | `synth/{glTF-sha}-{generator-ver}-s{seed}/` | until generator or CAD changes | a new key simply creates a new folder |
| Compiled procedure | `procedure_{station}_{cad_revision}.json` | until the CAD revision changes | new revision, new file |
| EWMA chart state | `(station, metric)` in `health_chart_state` | until a redeploy | a `model_version` or `cad_revision` change resets it to baseline |
| vcpkg builds | `C:\dev\vcpkg-cache` | indefinite | vcpkg manages it |
| MJPEG frames in the browser | — | **never cached** | `Cache-Control: no-store` on `:8090` |

### The render manifest is the cache key

```python
# synth/generate.py — the output folder name is the cache key
import hashlib, json, pathlib
GENERATOR_VERSION = "3"
def output_dir(gltf: pathlib.Path, seed: int) -> pathlib.Path:
    sha = hashlib.sha256(gltf.read_bytes()).hexdigest()[:12]
    d = pathlib.Path("data/synth") / f"{sha}-g{GENERATOR_VERSION}-s{seed}"
    if (d / "manifest.json").exists():
        print(f"cache hit: {d}"); raise SystemExit(0)    # same CAD, same code, same seed
    return d
```

Change the CAD and the hash changes; change the generator and bump `GENERATOR_VERSION`. Either way
the old folder stops matching. Nothing is ever deleted to "invalidate" it.

### The demo fallback

If anything fails on stage, `data\demo_fallback.mp4` is a screen recording of a perfect run, already
open in a second window. It is the one cache of a *result* this system permits — because it is
labelled as a recording and shown as one.

---

# Appendices

## Appendix A — Corrections to earlier drafts

Each of these was stated in an earlier version of this design. Each is wrong or superseded, and the
reason matters more than the fix.

| # | Earlier claim | Corrected | Why |
|---|---|---|---|
| 1 | MinIO for object storage | **Garage** | MinIO stopped publishing community Docker images in October 2025 and archived its repository in 2026. `docker pull minio/minio` no longer gets maintained images. |
| 2 | CUDA 12.1 exactly, TensorRT 8.6 | **CUDA 13.2, no TensorRT** | Your installed toolkit is supported everywhere now; TensorRT's five milliseconds are not needed and cost an exact version pin. |
| 3 | Qwen2.5-VL on Colab through a tunnel | **Florence-2-base, local** | Fits in 6 GB beside the detector; removes the tunnel as a demo-day risk. |
| 4 | `transformers==4.44.2`, plus `timm`, `einops` | **Current `transformers`**, neither extra | 4.44 predates the native `Florence2ForConditionalGeneration` class; the extras were for the remote-code checkpoint. |
| 5 | Check `Ort::GetAvailableProviders()` for CUDA | **Catch the append exception, then time a warm-up** | `GetAvailableProviders` lists compiled-in providers, not loadable ones, so it passes on the very failure it should catch. |
| 6 | Florence-2 needs ~1.6 GB for weights | **~0.46 GB weights; ~1.2–1.5 GB as a process** | 230M parameters × 2 bytes. The process total includes the CUDA context and activations. |
| 7 | Four stations at 30 fps under 40% GPU | **Four stations at 15 fps, measured** | The earlier figure was not measured. Four processes time-slice one GPU. |
| 8 | "≈ 25–30 ms" alert latency | **Engine ≈ 30 ms, evidence ≈ 65 ms** | The earlier figure omitted the debounce frame. Quote both, with the clock defined. |
| 9 | "6 GB will thrash" for YOLO training | **It fits; Colab is for parallelism and heat** | YOLO11n at batch 16 trains comfortably on 6 GB. |
| 10 | I9: byte-identical renders | **Identical labels, PSNR ≥ 45 dB** | GPU path tracing is not guaranteed bit-exact across driver versions. |
| 11 | Star solver for even N | **Fallback to first unvisited start** | The earlier version looped forever on N = 8. |
| 12 | Qt 6 operator HMI | **Kiosk route of the React app** | A second UI toolchain on a 16 GB machine for no rubric gain. |
| 13 | MediaMTX for live video | **MJPEG from the engine** | The clip ring already holds JPEG frames; streaming them costs no extra encode and no extra service. |
| 14 | OPC-UA write on the verify thread | **Dedicated alert thread** | It is blocking I/O, which the hot-path rules forbid on the verify thread. |
| 15 | Scenario implied the system catches 62 Nm vs 103 Nm | **It catches sequence, not torque value** | A camera cannot see clamp load. Torque value stays the nutrunner's job until Tier 2 joins the two. |
| 16 | Presets named `wsl-asan`, `wsl-tsan` | **`linux-asan`, `linux-tsan`** | The same presets run in GitHub Actions; they were never WSL-specific. |
| 17 | Python 3.11 | **Python 3.12, latest patch** | You already have 3.12, and the cu130 wheel line covers it. |
| 18 | Set the soft PLC's `serverUrls` to `host.docker.internal` so Docker clients can resolve it | **Leave `serverUrls` at its default** | Tested against open62541 1.4.21. The server already echoes each client's own URL in GetEndpoints; asyncua, the open62541 client and node-opcua all connected while it ran under an unresolvable hostname. And in 1.4 `serverUrls` also sets the bind address: a name that is not a local interface leaves the server with no socket, and it exits. The v4.0 advice would have broken the soft PLC. |
| 19 | A procedure engine shaped around one procedure: bolt rising edges, one star sequence | **A generic interpreter over profiles** | The problem statement names three assemblies; the v4 engine could check one of them. |
| 20 | Every frame processed at 960×540 | **Detector on the downscale; inspection-point crops from a full-resolution pool** | The new resolution check put the wheel's bolt face at 12.8 px at 960 px wide — below the 16 px threshold. |
| 21 | The star solver started wherever the angle sort began | **Start at member 1, run in the numbering's direction** | It produced 2, 5, 3, 1, 4 for the wheel: a valid star, but not the documented 1, 3, 5, 2, 4. |
| 22 | `observed_bolt`, `expected_bolt`, `TORQUE_SEQUENCE_VIOLATION` | **`observed_ip`, `expected_ip`, `profile`, `profile_version`; six generic violation kinds** | A door clip is not a bolt. |
| 23 | Evidence latency ≈ 65 ms | **≈ 70 ms, an estimate** | The crop-classification stage adds a few milliseconds; `latency_test` will settle it. |
| 24 | k3s for Tier 2 | **RKE2** | NVIDIA's GPU Operator documents RKE2 explicitly and not k3s; one plant cluster with station agents replaces one cluster per line. |
| 25 | The fleet agent reads `nvidia-smi`, injected by `NVIDIA_DRIVER_CAPABILITIES=utility` | **The node's dcgm-exporter, through a node-local Service** | Under the GPU Operator's CDI injection a pod that claims no GPU gets no driver utilities. The GPU Operator already runs dcgm-exporter on every GPU node. |
| 26 | `runtimeClassName: nvidia` on the engine pod | **Removed** | From GPU Operator 25.10, GPUs are injected through CDI from the resource request alone. |

## Appendix B — Coding-agent kit deltas

The `AGENTS.md` kits for Claude Code, Cursor and VS Code Copilot predate this document. Apply these
edits so the agents build what this spec says:

| Area | Change |
|---|---|
| Machine | Add CUDA 13.2, Python 3.12, Windows 11 Home, project root `C:\dev\torquesense` |
| Inference | Replace TensorRT everywhere with ONNX Runtime `gpu_cuda13` + CUDA EP; delete `.trt` artefacts, `trtexec` steps and the engine-freshness checklist item |
| Sidecar | Replace Qwen/vLLM/Colab with local Florence-2; delete SM 7.5, xformers and tunnel guidance |
| Storage | Replace MinIO with Garage in every rule, compose file and command |
| Threads | Add the alert thread; the OPC-UA write is forbidden on the verify thread |
| Invariants | I4 restated as **evidence** latency; I7 restated as VRAM budgets; I9 restated as labels + PSNR; add I11 |
| HMI | Replace Qt rules with "operator HMI is the `/hmi` route; accepts no input" |
| Presets | `wsl-*` → `linux-*`; add `windows-ci` |
| Video | Replace MediaMTX with MJPEG on `:8090` |
| Hooks | The Claude Code guard hook's allowed `#ifdef _WIN32` paths are unchanged: `ipc/` and `control-app/src/Affinity.cpp` |
| Invariants | Add I12 (model manifest) and I13 (fail visible) |
| Operations rules | `fleet-agent/`: standard library only, `gofmt`, `go vet`, `go test -race`. `deploy/k8s/`: check with `kustomize build … \| kubeconform -strict`. `Jenkinsfile`: lint with `npm-groovy-lint`. Soft PLC: never set `serverUrls`. |
| Drift job | `api/fleet/ewma.py` parameters change only with a re-run of the simulation recorded in an ADR |
| Profiles | Engine code may not name an assembly, a part or a family; everything specific lives in `profiles/`. A change to the interpreter lands with the matching change to `reference/tsref` and passes the differential test. |
| Capture and platform | `IFrameSource` is the only way frames enter; pylon code lives behind `SUNVERIFY_WITH_PYLON` in `control-app/src/sources/`. Capture arithmetic changes land with `reference/tsref/capture.py`. `deploy/k8s/`: no `runtimeClassName`; validate against Kubernetes 1.36. |

## Appendix C — Portfolio and interview mapping

### Resume bullets — update the numbers only after measuring

Your current resume quotes **45 ms p99** and describes a **remote** inference sidecar. Both are now
out of date. Once `latency_test` has run on this laptop, replace them with the measured figures, in
this shape:

> Built a five-thread C++20 pipeline (capture → inference → verification → alert → I/O) for
> real-time wheel-fitment verification, passing frames over cache-line-aligned lock-free SPSC ring
> buffers from a pre-allocated pool — zero heap allocation in the steady state, proven by a
> counting-allocator test — and measured **[N] ms p99 evidence latency** against a 1-second
> requirement under locked GPU clocks.

> Compiled the inspection procedure from CAD instead of hand-writing rules: a depth-first traversal
> of the STEP assembly tree with Open CASCADE extracts fastener geometry, a star-pattern solver
> derives the tightening order (property-tested for N = 4–12), and a deterministic state machine
> validates the observed sequence in O(1) per frame; the same CAD model generates the
> auto-labelled training set through BlenderProc.

> Designed a cross-process, cross-language shared-memory clip ring (Win32 named mapping, per-slot
> seqlocks, read from Python via `mmap`) so a local vision-language sidecar (Florence-2) can
> adjudicate ambiguous detections without copying frames or blocking the alert path; isolated it
> behind a circuit breaker, kept the build ASan/UBSan/TSan-clean in CI, and removed TensorRT after
> profiling showed its gain was unnecessary against the frame budget.

Do not put a number on the resume that `docs/VALIDATION.md` does not contain.

### For Point Wild (C++, DSA, OS)

| Their assessment area | Where it is in this project |
|---|---|
| Processes and threads | five threads, five processes, the reasons for each boundary (§13) |
| Synchronisation | SPSC memory ordering; seqlocks; why no mutex (§29, I2) |
| Memory management | frame pool, zero-allocation proof, VRAM budgets, arena caps (§12, §30) |
| IPC | named shared memory read from two languages (§29) |
| Scheduling | core pinning, priority classes, why not realtime, Windows timer resolution, EcoQoS (§13) |
| DSA | star solver, ring buffer, HdrHistogram, LRU adjudication cache |
| Debugging and optimisation | sanitizers, the latency clock definition, removing TensorRT by measurement |

### For Sun Fibo (CAD customisation, C++)

| JD line | Where it is |
|---|---|
| CAD API for features, bodies, properties | OCCT XCAF traversal of the STEP assembly — the same class of work as NX Open (§7) |
| DLL / class library development | `cad-extract`, `perception`, `procedure`, `ipc` as separate libraries behind interfaces |
| Assembly component automation | the procedure compiler reads the assembly and emits the station's procedure |
| Excel integration | `XlsxReporter` shift export, in C++ |
| Test scenarios and documentation | per-profile scenario matrices, `VALIDATION.md`, `functional-specification.md` |
| Two languages minimum | C++ core, Python sidecar and data pipeline, TypeScript front end |

### For Seagate (Software Development Intern, Supply Chain Operations and Services)

The group builds software that monitors storage fleets, improves operational visibility,
anticipates device issues, and supports redeployment, reliability and security decisions. The
operations track is the same work on a different fleet.

| What the role does | Where it is in this project |
|---|---|
| Monitor fleets; operational visibility | `/fleet`: telemetry from every station, six signals each drawn against that station's own normal (§18, §20) |
| Anticipate device issues | EWMA drift detection with limits set by simulation to a false-ticket budget; the anchor residual catches a knocked camera before detection fails (§18) |
| Redeployment and reliability decisions | Canary on one station, judged against its peers; promote or `kubectl rollout undo`; a fail-visible heartbeat (§18, §31) |
| Security | Model integrity checked before load; the agent listens on no port; certificate expiry as a health signal in Tier 2 (§18) |
| C/C++ and Python | C++ engine and soft PLC; Python API, drift job, data pipeline and tests |
| Automation scripts | `start-demo.ps1`, contract generation, the render pipeline, the drift job |
| CI/CD with Jenkins or GitLab | A `Jenkinsfile` for the GPU gates beside GitHub Actions for everything else (§30) |
| Containerisation and orchestration | Three Dockerfiles; Kustomize base and overlays; a kind smoke test in CI; RKE2 with the GPU Operator as the edge target, GitOps with Flux (§31, §32) |
| Linux/Unix | Linux sanitizer and headless builds, WSL2, Bash, the engine container |
| Git workflows and code review | Short-lived branches into `main`, a PR template that asks for the gate output, CI that fails on contract drift |
| Golang (a plus) | The fleet agent |
| Document learnings | `docs/VALIDATION.md`, `docs/adr/`, this specification |

**Timing.** The posting opened on 13 October 2026 and stays open for at least seven days. Apply with
the Seagate resume as it stands — it already describes TorqueSense as a monitoring and
anomaly-detection pipeline with Docker and CI. Do not wait for the operations track. If it is
finished before the interview, it becomes the thing you walk them through.

**Bullets to add once built** — replacing the third TorqueSense bullet on the Seagate resume, and
only after each part runs:

> Built the operations layer for a fleet of inspection stations: per-station health telemetry with
> one-sided EWMA control charts whose limits were set by simulation to at most one false
> maintenance ticket per station per month, flagging a degrading camera before it causes false
> alarms; a PLC-side heartbeat watchdog that turns a station amber within 2.5 s of its inspector
> stopping; and a dependency-free Go fleet agent with an in-order offline spool.

> Set up CI/CD across two systems: GitHub Actions for Windows (MSVC) and Linux builds, sanitizer
> runs, Go, Python and Playwright tests, and a kind-based Kubernetes smoke test of the container
> images; plus a Jenkins pipeline on a GPU workstation for the latency and accuracy gates that
> hosted CI cannot run.

The sentence for this interview:

> *"I treated a set of camera inspection stations as a device fleet: every station reports its
> health, a control chart learns each one's normal and flags drift before it turns into false
> alarms, a dead inspector shows amber in two seconds, and a new model goes to one station first
> and rolls back with one command."*

The one sentence that covers all three interviews:

> *"I built a real-time assembly verification engine in C++ that checks any assembly described as
> data; its inspection procedure and its training data are both compiled from the CAD — and every latency number in it was
> measured under controlled conditions, with the clock defined."*
