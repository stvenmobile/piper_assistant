# Piper Assistant

**Piper** is a local voice assistant running on an **NVIDIA Jetson Orin NX (16 GB)**: it listens
for its name, answers simple questions instantly on the device, hands everything else to a language model on the local network, and speaks the reply with a natural neural voice. No cloud services are involved apart from a weather lookup.

This repository is Piper's **runtime** - the program that runs on the Jetson. It is being
extended to give Piper a body: **[piper-watch](https://github.com/stvenmobile/piper-watch)**, a small robot head with a camera and a glowing light ring, so Piper can **see** who is there, **turn to look** at them, **recognise** people it knows, and **show** what it is doing and how it "feels" (section 6).

Piper's research side - curiosity, choosing what to study and checking whether studying worked - continues in **[curious-george](https://github.com/stvenmobile/curious-george)**, which will later connect to this runtime (section 9).

---

## 1. Status

| Part                                                                              | Status                                     |
| --------------------------------------------------------------------------------- | ------------------------------------------ |
| Voice loop: wake word, speech-to-text, quick local answers, LLM replies, speech   | **Working**                                |
| Two speech engines (Kokoro on the GPU, Piper on the CPU), chosen in `config.yaml` | **Working**                                |
| Per-person profiles loaded into the conversation                                  | Working (person found by "my name is ...") |
| Unit tests (config, quick responder, weather cache, state, audio devices)         | **Working** - `pytest`                     |
| SP-200 speakerphone (microphone + speaker with echo cancellation)                 | Next: set the device names                 |
| piper-watch head: light ring states, head link                                    | Planned - roadmap phase 1                  |
| Pan motor and face tracking                                                       | Planned - phases 2-3                       |
| Face recognition (who is there) replacing "my name is ..."                        | Planned - phase 3                          |
| curious-george connection (idle-time curiosity)                                   | Planned - phase 5                          |

---

## 2. Hardware

| Part                                                                  | Role                                                                                                                                  |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **NVIDIA Jetson Orin NX 16 GB** in a Seeed **reComputer J4012**       | runs everything here (Ubuntu 22.04, JetPack 6.1, CUDA 12.6)                                                                           |
| **SP-200 USB speakerphone** (4-mic array, hardware echo cancellation) | microphone and speaker; sits on the desk wherever is convenient. Until it is configured, a USB microphone and a USB speaker are used. |
| **Ollama** server on the local network (`llama3.2:latest`, the 3B model)                | the language model for anything the quick responder can't answer                                                                      |
| **piper-watch** (planned)                                             | the robot head: camera, light ring, pan motor - sits on top of the reComputer                                                         |

---

## 3. How a conversation works

```text
 microphone ─► listener ─► wake word? ─► quick responder ──(match)──────────────► speech
               (energy                     (time, date,                               ▲
                detection +                 weather, hello,                           │
                Whisper)                    goodbye, status)                          │
                                               │ no match                             │
                                               ▼                                      │
                                     supervisor (LangGraph) ─► Ollama on the LAN ─────┘
                                     + persona (system_dna.md)
                                     + person's profile (profiles/<name>.md)
                                     + date, time, cached weather
```

- **Listener** (`piper_audio/listener.py`): measures the room's noise at start-up, records
  whenever the sound level rises above it, and transcribes with **faster-whisper** (`base.en`,
  CPU, int8). While idle, only utterances containing "Piper" (or Whisper's favourite mishearing,
  "paper") wake it.
- **Quick responder** (`piper_brain/quick_responder.py`): answers the date, time, weather,
  greetings, goodbyes and status checks in milliseconds, without the language model. "Goodbye,
  Piper" ends the conversation; "shut down" stops the program.
- **Supervisor** (`piper_brain/supervisor.py`): a LangGraph flow that adds Piper's persona, the
  current person's profile, the date and the (cached) weather, keeps the last 8 messages of the
  conversation, and asks Ollama for a short spoken reply.
- **Speech** (`piper_audio/`): one of two engines, chosen with `audio.tts_engine`:

|               | Kokoro-82M (`kokoro`)            | Piper TTS (`piper`)       |
| ------------- | -------------------------------- | ------------------------- |
| Runs on       | GPU (CUDA)                       | CPU                       |
| Sound         | very natural, human-like prosody | clean, slightly synthetic |
| Default voice | `af_heart`                       | `en_US-hfc_female-medium` |
| Memory        | ~330 MB                          | < 60 MB                   |

### States

`src/main.py` moves through four states. They will drive the light ring (section 6.3).

```text
            "Hey Piper" (+ chime)                 request
  IDLE ─────────────────────────► ENGAGED ────────────────► PROCESSING ──► SPEAKING
   ▲                                 ▲  │                                      │
   │  20 s of silence, or "goodbye"  │  └──────────── follow-up turn ◄─────────┘
   └─────────────────────────────────┘
```

---

## 4. Repository layout

```text
piper_assistant/
├── config.yaml              settings (every one has a default in piper_brain/config.py)
├── system_dna.md            Piper's persona and voice rules
├── profiles/                one markdown profile per person (e.g. steve.md)
├── daily_journal.md         activity log written by the runtime
├── src/
│   ├── main.py              the runtime: state machine and conversation loop
│   ├── piper_audio/         listener (Whisper), Kokoro and Piper speakers, devices.py
│   ├── piper_brain/         config, supervisor, quick responder, tools, state, journal
│   ├── piper_memory/        research memory: SQLite store, embeddings, concept graph, seeds
│   ├── piper_research/      the research loop and its scheduler (section 5.5)
│   ├── piper_geometry/      research: residual-stream extraction (continues in curious-george)
│   └── piper_tools/         research: Obsidian vault and reading-dashboard builders
├── tests/                   unit tests (pytest)
├── tools/audio/             hand-run hardware checks: audio diagnostic, voice auditions,
│                            Kokoro benchmark (see tools/audio/README.md)
└── requirements.txt, requirements-dev.txt, pyproject.toml
```

Planned packages for piper-watch: `src/piper_head/` (head link client and service) and
`src/piper_vision/` (vision service) - see section 6.

---

## 5. Setup and running

### 5.1 Environment

```bash
cd ~/piper_assistant
source .venv/bin/activate
pip install -r requirements.txt     # torch / kokoro use the Jetson's CUDA builds
```

### 5.2 Configuration

Everything has a default in `src/piper_brain/config.py`; `config.yaml` only needs what differs. The environment variables `PIPER_OLLAMA_URL`, `PIPER_LLM_MODEL` and `PIPER_VOICE_MODEL` override the file.

```yaml
assistant:
  engaged_timeout_seconds: 20.0     # back to IDLE after this much silence
  max_conversation_turns: 8         # messages kept in the conversation
llm:
  base_url: "http://192.168.1.150:11434"
  model: "llama3.2:latest"
audio:
  mic_device_hint: "pnp"            # any unique part of the device name (see 5.3)
  speaker_device_hint: "usb2.0"
  tts_engine: "kokoro"              # "kokoro" (GPU) or "piper" (CPU)
  kokoro_voice: "af_heart"          # af_heart, af_bella, am_adam, am_michael
weather:
  location: "Matthews,NC"
  cache_minutes: 10
```

### 5.3 Choosing the microphone and speaker (SP-200)

List the audio devices the Jetson sees, then put any unique part of the speakerphone's name in both `audio.mic_device_hint` and `audio.speaker_device_hint` (the SP-200 is the microphone and the speaker in one):

```bash
python3 src/piper_audio/devices.py
```

### 5.4 Running

```bash
python3 src/main.py
```

Type `q` and Enter to stop cleanly, or say "shut down".

### 5.5 Research

In its scheduled hours (`research.windows`, default `20:00-08:00` local time) Piper researches
the topics in her memory (`piper_memory`). Each cycle she:
- picks a topic, favouring stale, thin and evenly-argued ones;
- picks or writes a question;
- reads Wikipedia and keeps the passages closest to the question;
- has the research model (`research.model`, default `qwen3:30b-a3b` on the PC's Ollama) extract findings.

A finding is kept only if its supporting quote really appears in the page. A separate judge,
with the model's thinking mode on, then rates each finding:
- its **relevance** to the topic: core, background, or off-topic (dropped);
- its **stance** toward the thesis: supports or challenges only if a proponent or critic would
  cite it, otherwise neutral, with a one-line reason.

The finding is stored with the page's permanent link (by revision), its certainty
(established / reported / speculative), and its links in the concept graph. Every few cycles she reflects: where the evidence stands, plus remarks to
say aloud. When the window closes she writes an overnight summary.

`start_piper.sh` starts the service. To load topics and try it by hand:

```bash
cp research_seeds.example.yaml research_seeds.yaml      # then write your own topics
cd src && python3 -m piper_memory.seeds && cd ..
python3 src/piper_research/service.py --now --once      # one cycle now, whatever the time
python3 src/piper_research/service.py --now --cycles 5 --topic "How migrating birds navigate"
```

Two maintenance commands:
- `--rejudge 500` runs the judge over findings stored before it existed. The service also does a
  few after each cycle (`research.rejudge_per_cycle`).
- `--tidy` prunes the open questions to `research.max_open_questions` per topic and marks the old
  cross-topic flags as reviewed.

To change the hours, set `research.windows` in `config.yaml` (several windows are allowed, and
a window may cross midnight) or set `PIPER_RESEARCH_WINDOWS="01:00-08:00,13:00-14:00"`.

### 5.6 Tests

The unit tests need no audio hardware, GPU or models, so they run on any machine:

```bash
pip install -r requirements-dev.txt
python3 -m pytest
```

---

## 6. Integration with piper-watch (planned)

**piper-watch** is Piper's head. It sits on top of the reComputer and turns left and right (about ±100°) on a lazy-Susan bearing, driven by a stepper motor and belt. Its round face has the **camera lens** in the middle, framed by a **24-LED light ring** behind a diffuser. An **ESP32-S3** in the base drives the motor and the ring; the camera, the ESP32 and the SP-200 all plug into the Jetson's USB ports. There are deliberately no eyes or mouth: the ring is the face, and Piper's voice comes from the speakerphone.

The head adds two things to Piper: **seeing** (the camera) and **showing** (the light ring and where the head points).

### 6.1 Architecture

Face tracking has to run continuously - the head keeps following you while Piper listens, thinks and speaks - so it can't live inside the conversation loop, which waits on each step. The work is split into three processes on the Jetson that talk over local sockets:

```text
                         ┌────────────────────────── Jetson ───────────────────────────┐
  C920X camera ─USB─►    │  VISION SERVICE (piper_vision)                              │
                         │   capture → face detection → tracking → recognition         │
                         │      │ LOOK (pan target)            │ presence events       │
                         │      ▼                              ▼                       │
                         │  HEAD LINK (piper_head)        ASSISTANT (src/main.py)      │
  ESP32-S3 ◄─USB serial─ │   owns the ESP32's port   ◄──── FACE (state, mood,          │
  (motor + light ring)   │   heartbeat, reconnects          attention)                 │
                         │                                                             │
  SP-200 ◄──────USB──────│  ◄── microphone / speaker ──►  ASSISTANT                    │
                         └─────────────────────────────────────────────────────────────┘
```

- **Head link** - a small service that is the only program talking to the ESP32. It forwards pan targets from vision and face states from the assistant, sends a heartbeat (the ESP32 stops the motor and dims the ring if the Jetson goes quiet), and reports the head's position back.
- **Vision service** - owns the camera, runs face detection and recognition on the GPU
  (TensorRT), steers the head, and tells the assistant who is present.
- **Assistant** (this program) - unchanged in structure; it reports its state to the head link and asks vision who is there.

Each part can be restarted or tested on its own: the light ring can be developed with just the ESP32 on the bench, and vision without the motor.

### 6.2 Visual input: what seeing adds

| Today                                                       | With piper-watch                                                                                                                         |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| The person is identified only if they say "my name is ..."  | Piper **recognises faces** and loads that person's profile automatically, so it can greet people by name and remember context per person |
| Piper doesn't know anyone is there until it hears its name  | Piper **notices someone arriving** and turns to look; it can offer a greeting, or simply show attention on the ring                      |
| A conversation ends after 20 s of silence or "goodbye"      | It can also end when the person **walks away**, and stay open while they are still there thinking                                        |
| No sense of who is speaking when several people are present | The head turns to the face it is attending to, and the ring's attention arc points at them                                               |

Recognition stays on the Jetson: face prints are stored locally in `faces/` (git-ignored), only after someone gives their name, and "Piper, forget me" deletes a person.

**Meeting people** (`src/piper_skills/meet_person.py`, built): Piper only helps people she knows by name.

- A stranger in view is asked their name straight away; no wake word is needed. Until she has it, she answers nothing else ("first I need to know who you are").
- She checks the name ("Steve? Did I get that right?"). A correction ("no, it's Stephen") or a spelling ("S-T-E-V-E") also works.
- She then asks them to look at her while vision stores five looks at their face. She also starts a profile for them in `profiles/<name>.md`.
- Someone who won't say is told they can say goodbye. After "goodbye" she leaves them alone until they walk away and come back.
- Someone she knows gets "Welcome back, Steve! How can I help you today?", but not again if they were there in the last 10 minutes.
- "Who am I?" and "forget me" also work.
- Without the vision service, she behaves as before.
- Settings: `assistant.require_known_person`, `assistant.welcome_back_minutes`, and the `vision.recognize` / `match_threshold` / `enroll_*` keys.

### 6.3 State display and mood expression

The ring is how Piper shows what it is doing. The basic states map directly onto the
assistant's existing states:

| Assistant state                 | Ring                                                           | Head                              |
| ------------------------------- | -------------------------------------------------------------- | --------------------------------- |
| IDLE, nobody around             | slow, dim warm glow; "sleeping" ember after a long quiet spell | rest pose, occasional look-around |
| IDLE, someone present           | soft glow with a brighter **attention arc** pointing at them   | follows their face                |
| ENGAGED (listening)             | the whole ring **breathes** slowly                             | faces the speaker                 |
| PROCESSING (thinking)           | a short **comet** runs around the ring                         | holds                             |
| SPEAKING                        | a gentle **pulse** while the speech plays                      | faces the listener                |
| Error (e.g. Ollama unreachable) | **amber** flashes                                              | -                                 |

On top of the state, a **mood** colours how it is shown - hue, brightness and tempo - without ever turning the ring into a cartoon:

| Mood               | How it shows                                                           | Where it comes from                                                        |
| ------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Neutral**        | soft warm white                                                        | default                                                                    |
| **Warm / pleased** | warmer, a little brighter; a welcome swell when a known person arrives | recognising someone; a friendly exchange                                   |
| **Curious**        | slow cool-tinted shimmer                                               | idle-time exploration (curious-george, section 9); an interesting question |
| **Uncertain**      | dimmer, slower breathing                                               | low-confidence speech recognition; "I'm not sure" replies                  |
| **Concerned**      | amber tint                                                             | errors, lost network, the head blocked                                     |
| **Sleepy**         | very dim ember                                                         | long inactivity, late at night                                             |

The mood comes from simple, inspectable signals - recognition events, error conditions, time of day, idle activity - plus an optional one-word mood tag the language model can add to its reply (stripped before the text is spoken). It decays back to neutral on its own.

### 6.4 Messages

The head link and the ESP32 exchange newline-delimited JSON over the ESP32's USB serial port (easy to read while debugging). The draft message set, shared with the piper-watch firmware:

| Direction      | Message     | Example                                                          |
| -------------- | ----------- | ---------------------------------------------------------------- |
| Jetson → ESP32 | `HEARTBEAT` | `{"t":"HEARTBEAT","seq":812}`                                    |
| Jetson → ESP32 | `FACE`      | `{"t":"FACE","state":"listening","mood":"warm","attention":-20}` |
| Jetson → ESP32 | `LOOK`      | `{"t":"LOOK","pan":-18.5,"speed":60,"mode":"track"}`             |
| Jetson → ESP32 | `CONFIG`    | `{"t":"CONFIG","pan_limits":[-100,100],"max_brightness":64}`     |
| ESP32 → Jetson | `STATUS`    | `{"t":"STATUS","pan":-17.9,"moving":true,"homed":true}`          |
| ESP32 → Jetson | `EVENT`     | `{"t":"EVENT","what":"watchdog"}`                                |

The assistant and vision talk to the head link over a local socket with the same messages, so any of them can be driven by hand from a test script.

---

## 7. Roadmap

0. **Clean-up** - *done*: config-driven settings, working TTS-engine choice, weather cache,
   one config loader, unit tests.
1. **Head link and light ring** (ESP32 + ring on the bench): message protocol, head link
   service, ring states from the assistant's state changes. *Milestone: the ring reacts to the conversation.*
2. **Motion** (printed base, NEMA17 + belt): smooth stepper control, homing, soft limits,
   watchdog, a manual pan test tool.
3. **Vision** (camera mounted): detection and tracking steering the head; recognition and enrolment; who-is-present events replacing "my name is ..."; a dashboard page with the camera view. *Milestone: Piper follows you and knows who you are.*
4. **Conversation quality**: a dedicated wake-word model (openWakeWord), streaming speech (start speaking before the whole reply is generated), talking over Piper using the SP-200's echo cancellation, all services started at boot (systemd).
5. **Curiosity**: connect curious-george (section 9).

---

## 8. Configuration notes

- `system_dna.md` holds Piper's persona and voice rules; it is read at start-up.
- `profiles/<name>.md` is loaded into the conversation for that person (file name in lower case, e.g. `steve.md`).
- `daily_journal.md` records state changes, local answers and LLM replies, grouped by date.

---

## 9. Related projects

- **[piper-watch](https://github.com/stvenmobile/piper-watch)** - the robot head: hardware
  design, CAD, and the ESP32-S3 firmware (motor, light ring, protocol).
- **[curious-george](https://github.com/stvenmobile/curious-george)** - research into machine curiosity: choosing what to study and measuring whether studying it taught the model anything. It exposes its tools through an **MCP server**; the plan is for this runtime to use them while Piper is idle (choose a topic, study it, log what was learned), to talk about what it has been curious about, and to show it on the ring ("curious" mood). Where study runs
  (Jetson, PC GPU or Colab) and how learned adapters reach Ollama are still open.
