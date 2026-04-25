### Phase 1: Hardware & Base Implementation

* **Wiring:** Connect the 3 NodeMCUs, INA219 voltage sensors, and the 4-channel relay to the 18650 cells. Wire TP4056 chargers per cell so cells can be topped up between episodes without swapping.
* **Firmware:** Flash the ESP8266 boards to read INA219 data via I2C, drive a status LED on D4, and POST heartbeats to the Raspberry Pi every 5 seconds. Edit `AGENT_ID` per board before flashing.
* **Bridge:** Boot the FastAPI bridge server on the Raspberry Pi to expose relay, voltage, ultrasonic, and ledger endpoints.
* **Env Sanity Check:** Verify `AgentGridEnv`, the composable rubrics, and the tabular Q-learning trust model run locally without exceptions in pure-sim mode.
* **Ledger Stub:** Create `bridge/ledger_bridge.py` so `bridge/server.py` imports `CommitmentLedger` cleanly without the fallback warning.

### Phase 2: Data Generation & Pure Simulation

* **Synthetic Data:** Prompt an LLM API (GPT-4o or Claude) to generate 2,000 synthetic 10-step negotiation transcripts. This forms your JSON action grammar dataset, saved to `training/synthetic_traces/sft_data.jsonl`.
* **Sim Backend:** Run the environment in pure simulation mode (`HARDWARE_BRIDGE_URL` unset) to bypass hardware. The calibrated battery curves in `sim_backend.py` carry the load.
* **Baseline Run:** Execute `eval/baseline_random.py` to confirm the simulator handles battery drain, observation rendering, and rubric scoring correctly. Verify Survival/Task/Promise rubrics fire non-zero rewards.
* **Trust Model Hookup:** Confirm `TrustModel.record_settlement()` fires on each verified trade and `end_episode()` fires on reset. Observation must include the `TRUST MODEL` block with Q-values and UCB bounds.

### Phase 3: The Training Pipeline

* **Stage 1 SFT:** Run Unsloth on Colab. Fine-tune Llama-3.2-1B base model with LoRA (r=16, alpha=16, target `q_proj`/`v_proj`) on the synthetic transcripts until JSON output validity hits 90% on held-out traces. Push checkpoint to HF Hub.
* **Stage 2 GRPO:** Load the SFT checkpoint and run TRL's `GRPOTrainer` self-play in pure simulation. Three parallel model copies, one per agent. Reward signal comes from `rubrics.py` directly. Track promise-keep rate alongside total return; save best checkpoint by promise-keep rate, not raw return.
* **Stage 3 HITL:** Connect the environment to the real Raspberry Pi bridge. Run 200 hardware-in-the-loop episodes to capture real battery noise and relay timing. Fine-tune for 1 epoch on the HITL replay buffer to close the sim-to-real gap.
* **Ablation Kickoff:** In parallel with Stage 2, kick off a second GRPO run with the trust model fields zeroed out in the observation. This produces the data for the trust-model ablation plot.

### Phase 4: Testing & Calibration

* **Relay Calibration:** Run 10 relay fires at `amount=0.1` with 250ms duration. Read voltage delta from the INA219, compute `volts_per_energy_unit = mean(delta_v) / 0.1`, and update `bridge/calibration.json`. Update the duration formula in `bridge/hardware.py` to `duration = amount / volts_per_energy_unit * 0.08`.
* **E2E Validation:** Run the trained policy against the live hardware. Confirm physical relay clicks perfectly match the LLM's accepted trades and INA219 registers the expected voltage drop within tolerance.
* **Ablation Run:** Compare the with-trust-model policy against the zeroed-trust policy on the same held-out scenarios. Generate `eval/plots/trust_model_ablation.png` showing the gap.
* **Hash Chain Verification:** Sanity-test the SHA-256 hash chain in `ledger.py` by appending two entries and confirming `prev_hash` linkage is correct. Required for the auditability claim.

### Phase 5: Demonstration & Submission

* **Data Vis:** Generate the required plots. Plot the baseline, sim-trained, and HITL-trained reward curves on one axis (`eval/plots/three_curves.png`). Commit ablation plot. Verify axes are labeled.
* **Scripted Replay:** Execute `eval/replay_demo_scenario.py` and record the 90-second pitch video triggering the ultrasonic sensor. Capture terminal transcript alongside the physical rig.
* **Deployment:** Push the `agentgrid_env/` directory and `openenv.yaml` to a Hugging Face Space. Verify it builds and is discoverable without hardware.
* **Documentation:** Finalize the README to explicitly hit the hackathon's 40/30/20/10 judging criteria. Link the HF Space, the unlisted YouTube video, and the HF mini-blog.

---

task division-

## Phase 1

### The Contract (The Common Ground)

Your absolute common ground is the HTTP REST API hosted on the Raspberry Pi. This enforces the client/server separation required by OpenEnv.

* **Software Perspective:** You don't care about I2C addresses or GPIO pins; you only care that hitting `POST /relay/fire` works and `GET /voltage/{agent}` returns a valid reading.
* **Hardware Perspective:** Gautam doesn't care about LLMs, GRPO, or the tabular trust model; he just needs to ensure the FastAPI server executes the JSON payloads and pushes accurate sensor data.

### Your Tasks (Phase 1 — Software)

You can build and test your entire stack without touching a single wire.

* **Env Core:** Finish `agentgrid_environment.py`, `rubrics.py`, and `trust_model.py`. The env is ~95% done — focus on integration testing, not structural work.
* **Sim Backend:** Rely heavily on `sim_backend.py`. This calibrated battery simulator allows you to run your environment in pure simulation mode (`HARDWARE_BRIDGE_URL=None`).
* **Trust Model Wiring:** Confirm `AgentGridEnv.__init__` instantiates one `TrustModel` per agent, that `step` calls `record_settlement(peer, action, verified_kept)` on each settled trade, and that `end_episode()` fires when `done` is True.
* **Observation Verification:** Start the env server and call `get_observation` manually. Confirm output contains YOUR STATE, PEERS, TRUST MODEL, INBOX, PENDING OFFERS, LEDGER, and the action menu.
* **Ledger Stub:** Create `bridge/ledger_bridge.py` re-exporting `CommitmentLedger` from `agentgrid_env.server.ledger`. Removes the fallback import warning.
* **Testing:** Run `eval/baseline_random.py` locally. If it completes an episode and the rubrics fire non-zero rewards, your layer is done.

### Gautam's Tasks (Phase 1 — Hardware)

Gautam's goal is to make the physical rig reachable over the network.

* **Wiring & Firmware:** Connect the 3 NodeMCUs to the INA219s (I2C addresses 0x40/0x41/0x44 via A0/A1 solder bridges) and flash `nodemcu_agent.ino`. The boards need to read voltage, drive the LED on D4, and send heartbeats to `/health`.
* **Relay Routing:** Wire the 18650 cells through the 4-channel relay module. GPIO 17 (A↔B), GPIO 27 (A↔C), GPIO 22 (B↔C). Test each relay with a raw `GPIO.output` script before any HTTP integration.
* **Ultrasonic & LEDs:** Wire HC-SR04 (TRIG=GPIO 23, ECHO=GPIO 24) and the three white LEDs with 220Ω resistors.
* **Charging Rig:** Mount TP4056 modules per cell so cells can be topped up between episodes without swapping.
* **Bridge Server:** Spin up the FastAPI server (`bridge/server.py`) on the Pi and test it using raw `curl` commands from a laptop.

### The Handshake (Phase 1 — Integration)

Your workflows merge when the physical network is established.

1. Gautam confirms that running `curl http://<pi-ip>:7000/health` returns a 200 OK status.
2. You plug that IP into your `HARDWARE_BRIDGE_URL` environment variable.
3. You run `baseline_random.py` against the live hardware to verify the relays physically click when actions resolve.

---

## Phase 2

Phase 2 is Data Generation and Pure Simulation. This phase is 100% on your plate. Gautam can keep soldering the rig or take a breather.

Here is your exact software checklist:

### 1. Generate the SFT Dataset

You need 2,000 synthetic negotiation transcripts to teach your Llama-3.2-1B models the JSON action schema before you start RL.

* Write a quick Python script hitting the GPT-4o or Claude API (~₹100 budget).
* Use a system prompt like: "You are playing Agent A in a 3-agent energy negotiation. Generate a realistic 10-step transcript where agents negotiate using these JSON actions: [paste action schema from env spec]."
* Cover all six action types: `broadcast`, `offer`, `accept`, `execute_task`, `renege`, `idle`.
* Save the output to `training/synthetic_traces/sft_data.jsonl`.

### 2. Start the Simulation Server

You are going to test the environment using the calibrated battery simulator (`sim_backend.py`), completely ignoring the hardware.

* Ensure the `HARDWARE_BRIDGE_URL` environment variable is unset or empty.
* Start the server: `uvicorn agentgrid_env.server.app:app --reload --port 8000`

### 3. Run the Baseline Test

Fire up the random policy against your simulated environment to verify the core logic works.

* In a new terminal, run: `python eval/baseline_random.py`
* **Pass condition:** The script completes episodes without crashing, and you see non-zero rewards logged from your Survival, Task, and Promise rubrics.

### 4. Verify Trust Model Integration

* Inspect a rendered observation and confirm the `TRUST MODEL` block appears with `Q_accept`, `Q_trust_pay`, `UCB`, and `N` fields per peer.
* Hand-test `trust_model.py` with a fake settlement stream: feed three `verified_kept=True` followed by two `verified_kept=False` events for one peer; confirm Q-value tracks toward the running mean.

### 5. Verify Ledger Hash Chain

* Append two entries via `CommitmentLedger`. Confirm entry 2's `prev_hash` equals entry 1's `this_hash`.
* This is required for the auditability claim in the README.

Once these pass cleanly, the environment is locked and you are ready to start the actual Unsloth training pipeline.

---

## Phase 3

Phase 3 is the core training pipeline. This all happens in Colab or Hugging Face Spaces. Gautam is finalizing the rig in parallel.

### 1. SFT Warmup (`01_sft_warmup.ipynb`) — ~30 min on free-tier T4

* Take the 2,000 synthetic JSON transcripts you generated in Phase 2.
* Use Unsloth to fine-tune a Llama-3.2-1B base model with LoRA (r=16, alpha=16, target_modules=`["q_proj", "v_proj"]`, 4-bit load).
* Use TRL's `SFTTrainer` for the supervised pass.
* **Goal:** Teach the LLM to consistently output valid JSON negotiation actions. You need a >90% validity rate on held-out traces before moving on.
* Push the checkpoint to HF Hub.

### 2. GRPO Self-Play (`02_grpo_selfplay.ipynb`) — ~3 hours on T4

* Load your SFT checkpoint.
* Start `AgentGridEnv` in pure sim mode (no `HARDWARE_BRIDGE_URL`).
* Run 3 copies of the model in self-play. Use TRL's `GRPOTrainer` with `reward_model=None` since the env provides rewards directly via the rubric scorer.
* Track per-episode: total return, **promise-keep rate**, task completion rate, JSON validity rate.
* Save best checkpoint by promise-keep rate, not just total return — this is the metric the trust-model story hangs on.

### 3. Ablation Run (parallel to step 2, no extra foreground time)

* Kick off a second GRPO training job with the same config, but zero out the trust model fields in the observation rendering.
* Runs alongside the main job in a separate Colab tab.
* This is the data source for `eval/plots/trust_model_ablation.png` in Phase 4.

### 4. Hardware-in-the-Loop (`03_hitl_finetune.ipynb`) — ~1 hour [HARDWARE]

* **Prereqs:** Bridge server running on Pi. Calibration completed (Phase 4 step 1). `HARDWARE_BRIDGE_URL` set on the env server.
* Load GRPO checkpoint.
* Run 200 episodes against the live bridge. This exposes the agents to real battery voltage drops, WiFi jitter, and physical relay timing.
* Capture INA219 readings into `training/synthetic_traces/hitl_curves.json`.
* Fine-tune for 1 epoch on the HITL replay buffer.

---

## Phase 4

This is where your simulated training meets Gautam's physical rig. Hardware and software synchronize tightly here, so the split is explicit.

### Your Tasks (Phase 4 — Software)

* **Pull Calibration Data:** Once Gautam fires the test relays, pull `delta_v` readings from the INA219 endpoint. Compute `volts_per_energy_unit = mean(delta_v) / 0.1`.
* **Update Calibration File:** Write the new `volts_per_energy_unit` to `bridge/calibration.json`. Update the relay duration formula in `bridge/hardware.py` line 91 from the placeholder `duration = amount * 2.5` to `duration = amount / volts_per_energy_unit * 0.08`.
* **End-to-End Validation:** Set `HARDWARE_BRIDGE_URL` to the Pi's IP and run the first episode of the HITL-trained policy. Confirm relay clicks match accepted JSON trades and voltage drops match expected amounts within tolerance.
* **Ablation Plot:** Pull the two GRPO training curves (with-trust vs zeroed-trust) and generate `eval/plots/trust_model_ablation.png`. Label axes clearly. The expected story: with-trust adapts faster to peer reneges.
* **Hash Chain Smoke Test:** Append two ledger entries during a real episode and confirm the SHA-256 chain links correctly. This proves the auditability claim.

### Gautam's Tasks (Phase 4 — Hardware)

* **Manual Calibration Fires:** Trigger 10 relay test fires at `amount=0.1` (using a base 250ms duration). Log the timestamp of each fire so the software side can correlate INA219 deltas.
* **Cell Reset:** Recharge cells to a known baseline voltage between calibration runs using the TP4056 modules.
* **Live Episode Support:** During the E2E validation run, watch the rig and confirm the LEDs respond and relays click on schedule. Flag any stuck-closed relays immediately (INA219 will detect this; abort the episode).
* **Ultrasonic Smoke Test:** Wave a hand at the HC-SR04. Confirm the bridge `/sensor/ultrasonic` endpoint returns a low distance reading that maps to an urgency spike for Agent C.

### The Handshake (Phase 4 — Integration)

1. Gautam confirms 10 calibration fires complete and posts the timestamps.
2. You pull the corresponding `delta_v` readings, compute the calibration constant, and commit the updated `bridge/calibration.json`.
3. You run the HITL-trained policy end-to-end. Pass condition: when Agent A accepts Agent B's offer in the JSON output, Gautam's relay must physically click and the INA219 must register the expected voltage drop.
4. Both of you watch the ablation plot get committed to confirm the trust model is pulling its weight. If the gap is flat, that goes in the README as a negative result; do not hide it.

---

## Phase 5

This is where you secure the points. Execution here matters as much as the code. The video shoot is the only real coordination point with Gautam.

### Your Tasks (Phase 5 — Software)

#### 1. Generate the Evidence (Plots)

You need hard proof of learning for the 20% "Reward Improvement" judging criteria.

* Confirm `eval/plots/three_curves.png` (Baseline vs. Sim-GRPO vs. HITL-GRPO) is committed with labeled axes.
* Confirm `eval/plots/trust_model_ablation.png` is committed.
* Both must be embedded in the README as PNGs, not links.

#### 2. Deploy to Hugging Face Spaces

This satisfies a core OpenEnv submission requirement.

* Create a new Hugging Face Space.
* Push your `agentgrid_env/` directory and the valid `openenv.yaml` file.
* Verify it builds and is discoverable. The Space must run end-to-end in pure-sim mode (no hardware required for judges to reproduce).

#### 3. Write the README

Treat the README as the grading rubric. Map it directly to the judges' scorecard:

* **Environment Innovation (40%):** Explicitly state that the reward function reads from a voltmeter. Name the hybrid architecture: "LLM for language, tabular Q-learning for trust, INA219 voltage as ground truth."
* **Storytelling (30%):** Write a tight 3-step narrative — Setup, the "hand wave" demo, and the resulting trust curves.
* **Reward Improvement (20%):** Embed both PNG plots directly in the markdown.
* **Pipeline (10%):** Document that all 5 rubrics are wired with weights, that `verify_sim` runs on every settled energy trade, and that `trust_model.end_episode()` runs on every reset.
* **Links:** Include the HF Space URL, the YouTube video link, and your HF mini-blog post.

#### 4. Write the HF Mini-Blog

Use the long-form pitch line: *"AgentGrid is the only OpenEnv submission where the reward function reads from a voltmeter, and trust is a tabular Q-function that a 1B LLM learns to read."*
Cover: the hybrid architecture, the ablation result, the demo moment.

### Gautam's Tasks (Phase 5 — Hardware)

* **Charge All Cells:** Top up all three 18650s to a known starting voltage before the demo recording. Have a spare charged cell on standby in case one dies mid-pitch.
* **Rig Setup for Camera:** Position the rig so the camera captures all three LEDs, the relay module clicking, and the ultrasonic sensor in the same frame.
* **Hand Wave Cue:** During the recording, physically trigger the HC-SR04 sensor on cue (simulating the urgency spike for Agent C). Practice timing: the spike should land while Agent C's transcript line is on screen.

### The Handshake (Phase 5 — Recording the 90-Second Pitch)

1. You start the env server with `HARDWARE_BRIDGE_URL` set, then run `python eval/replay_demo_scenario.py`.
2. Gautam films the rig from a fixed angle. The video must capture the terminal showing the negotiation transcript alongside the physical relay clicking and Agent C's LED brightening on the urgency spike.
3. Rehearse 3 times before the real take. Know which LED corresponds to which agent.
4. Upload the final cut as **unlisted** to YouTube. Do **not** commit the video file to the repo — link only.
5. Lock the repo once the four submission steps (plots committed, HF Space deployed, README written, video linked) are verified.

### Hard Cuts If Behind Schedule (in order)

1. Drop the HITL fine-tune notebook — ship sim-only. Demo still works because relays fire on accepted offers regardless.
2. Drop the trust model ablation plot — ship with the trust model wired but no ablation evidence (1-point hit on training evidence).
3. Drop end-of-episode MC reconciliation in `trust_model.py` — keep only in-episode Q-updates. Saves ~30 minutes; story basically unchanged.
4. Drop UCB from the observation, keep only Q-values — loses the explore/exploit angle but keeps the trust learning story.
5. Drop the 4-channel relay — use one relay (A↔B only), one less trade direction.

Do **NOT** drop the in-episode Q-updates or the voltmeter-derived reward. Those are the core of the v2.1 framing; without them, the submission collapses to "another text-only negotiation env."
