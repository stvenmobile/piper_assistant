# Piper Assistant

**Piper** is a local voice assistant running on an **NVIDIA Jetson Orin NX (16 GB)**: it listens
for its name, answers simple questions instantly on the device, hands everything else to a language model on the local network, and speaks the reply with a natural neural voice. No cloud services are involved apart from a weather lookup.

This repository is Piper's **runtime** - the program that runs on the Jetson. Piper has a body:
**[piper-watch](https://github.com/stvenmobile/piper-watch)**, a small desktop robot whose head pans and
tilts, with a camera and a face screen (animated eyes over a speaking mouth), so Piper **sees** who is
there, **turns to look** at them, **recognises** people she knows, and **shows** what she is doing and
how she "feels" (section 6). The robot is assembled and working; the head's printed parts are a
prototype and the final 5" face screen is on order.

Piper's research side - curiosity, choosing what to study and checking whether studying worked - continues in **[curious-george](https://github.com/stvenmobile/curious-george)**, which will later connect to this runtime (section 9).

---

## 1. Status

| Part                                                                              | Status                                     |
| --------------------------------------------------------------------------------- | ------------------------------------------ |
| Voice loop: wake word, speech-to-text, quick local answers, LLM replies, speech   | **Working**                                |
| Two speech engines (Kokoro on the GPU, Piper on the CPU), chosen in `config.yaml` | **Working**                                |
| Per-person profiles loaded into the conversation                                  | Working (person found by "my name is ...") |
| Research: daily rhythm, verified findings, synthesis with reasoning trails, mastery | **Working** - section 5.5                |
| Research dashboard (reports, insights and their trails, topics and mastery)        | **Working** - section 5.6                  |
| Unit tests (config, audio, state, research memory and loop, synthesis, rhythm, mastery, dashboard) | **Working** - `pytest`    |
| SP-200 speakerphone (microphone + speaker with echo cancellation)                 | **Working**                                |
| piper-watch robot: face screen (eyes + mouth), pan/tilt servos, camera, head link  | **Working** - prototype head (section 6)   |
| Face following (pan/tilt) and face recognition, meeting new people                | **Working** - section 6.2                  |
| Final head: 5" face screen, camera brow band, printed head v7                     | Next - parts on order                      |
| Research moved to curious-george (Colab), Piper focused on conversation           | Proposal - see the decision log (Claude project docs)           |

---

## 2. Hardware

| Part                                                                  | Role                                                                                                                                  |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **NVIDIA Jetson Orin NX 16 GB** in a Seeed **reComputer J4012**       | runs everything here (Ubuntu 22.04, JetPack 6.1, CUDA 12.6)                                                                           |
| **SP-200 USB speakerphone** (4-mic array, hardware echo cancellation) | microphone and speaker; sits on the desk wherever is convenient                                                                      |
| **Ollama** server on the local network (`qwen3:14b`, thinking off)                       | the language model for anything the quick responder can't answer                                                                      |
| **piper-watch** base: printed tub and lid                              | holds the reComputer, the 12 V power split and the servo adapter                                                                       |
| 2 x Feetech **STS3215** bus servos + **Waveshare Bus Servo Adapter (A)** | pan (ID 1) and tilt (ID 2); the adapter is on USB to the Jetson (CH343 chip), driven by the vision service                            |
| **CrowPanel Advanced 7" ESP32-P4** (1024 x 600) - the 5" (800 x 480) in the final head | the face: animated eyes over a spectrum-bar mouth (piper-watch `firmware/face_p4`); USB serial (CH340) to the Jetson |
| Logitech **C920** camera board                                        | vision (who is there, where to look); USB to the Jetson                                                                               |
| **QUANTUM** (PC, RTX 5070 12 GB)                                      | runs Ollama: `qwen3:14b` for conversation, `qwen3:30b-a3b` for research and "think carefully"                                         |
| **piper-mind** (small server)                                         | runs the research service and the dashboard (systemd units in `deploy/piper-mind/`)                                                    |

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
                                     + persona and purpose (system_dna.md)
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

`src/main.py` moves through four states. They drive the face (section 6.3).

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
├── system_dna.md            Piper's persona, her purpose (mastery) and voice rules
├── profiles/                one markdown profile per person (e.g. steve.md)
├── daily_journal.md         activity log written by the runtime (not in git: each machine keeps its own)
├── src/
│   ├── main.py              the runtime: state machine and conversation loop
│   ├── piper_audio/         listener (Whisper), Kokoro and Piper speakers, devices.py
│   ├── piper_brain/         config, supervisor, quick responder, tools, state, journal, research hold
│   ├── piper_head/          the head link (face screen), expressions, servos, protocol
│   ├── piper_vision/        the vision service: camera, face detection, tracking, recognition, following
│   ├── piper_skills/        spoken skills: meeting people, face tests
│   ├── piper_memory/        research memory: SQLite store, embeddings, concept graph, seeds
│   ├── piper_research/      research: the loop, synthesis, the daily rhythm, mastery (section 5.5)
│   ├── piper_dashboard/     web view of the research memory; its actions are requests (section 5.6)
│   ├── piper_geometry/      research: residual-stream extraction (continues in curious-george)
│   └── piper_tools/         research: Obsidian vault and reading-dashboard builders
├── tests/                   unit tests (pytest)
├── tools/audio/             hand-run hardware checks: audio diagnostic, voice auditions,
│                            Kokoro benchmark (see tools/audio/README.md)
├── tools/head_cli.py        drive the face by hand; tools/ollama_bench.py: time the chat models
├── deploy/piper-mind/       systemd units for the research service and dashboard
└── requirements.txt, requirements-dev.txt, pyproject.toml
```

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
  model: "qwen3:14b"
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

**Do not disturb:** `./start_quiet.sh` (or `./start_piper.sh --quiet`) starts everything except
the conversation. The microphone is never opened and Piper never speaks, so calls nearby can't
trigger her. Research, vision and the head run as usual, and `q` + Enter stops it.

### 5.5 Research

`./start_quiet.sh` starts Piper's research **rhythm** (`research.session_hours: 0`, the default):
research with no end, in a daily structure (`src/piper_research/rhythm.py`, `research.rhythm`):

| Time (local) | Phase |
|---|---|
| 05:00-07:00 | **Consolidate** - no new reading: reflect on every topic researched since yesterday, a synthesis pass over every topic, prune stale questions, review insights; on Sundays the **topic decisions** (rest a stuck topic, wake a rested one, and *propose* new topics and sub-topics - you add them). Topics ready for mastery ask for it. Spare time goes to the re-judging backlog |
| 07:00 | **Daily report** (and the weekly one on Sundays) - on the dashboard, and a short version Piper can say |
| 08:00-02:00 | **Pause**, every day (`pause_days`) - the PC's GPU is left to conversation (since 2026-11-24; was 10:00-17:00 on weekdays) |
| otherwise | **Explore** - research and synthesis cycles, with a checkpoint every hour |

Each research cycle, Piper researches the topics in her memory (`piper_memory`):
- picks a topic, favouring stale, thin and evenly-argued ones;
- picks or writes a question;
- reads Wikipedia, a Stanford Encyclopedia of Philosophy entry, and research papers - published
  journal papers from **OpenAlex** (the well-cited first) and **arXiv** preprints (labelled not
  peer-reviewed) - and keeps the passages closest to the question. OpenAlex is metered: without a
  free key (`OPENALEX_API_KEY` in `.env`) it allows about 100 searches a day, and Piper skips it
  until its allowance resets; arXiv is skipped for an hour if it says "too many requests";
- has the research model (`research.model`, default `qwen3:30b-a3b` on the PC's Ollama) extract findings.

A finding is kept only if its supporting quote really appears in the page. A separate judge,
with the model's thinking mode on, then rates each finding:
- its **relevance** to the topic: core, background, or off-topic (dropped);
- its **stance** toward the thesis: supports or challenges only if a proponent or critic would
  cite it, otherwise neutral, with a one-line reason.

The finding is stored with the page's permanent link (by revision), its certainty
(established / reported / speculative), and its links in the concept graph. Every few cycles she reflects: where the evidence stands, plus remarks to
say aloud.

**Synthesis** (`src/piper_research/synthesis.py`) is the second kind of cycle. It reads nothing new:
it takes a handful of findings that belong together - a specific concept shared by a few topics,
the strongest evidence on both sides of a thesis, or a finding and its nearest relatives in other
topics - and looks for relationships between them. The model reasons in numbered steps, each
citing the findings it uses, and proposes at most two **insights** (link, tension, hypothesis,
analogy, pattern), each with 1-2 new questions. Every citation is checked, each insight gets a
grounding check (unsupported ones are kept as *rejected*, so the trail shows what didn't survive),
and its questions become open questions tagged **derived**, with their own cap per topic. When
research later takes up a derived question, the new findings are weighed against the insight and
its confidence moves. Synthesis runs when a topic is saturating (its recent cycles find little
new), and at least once every `research.synth_every_max` cycles.

`start_piper.sh` starts the service. To load topics and try it by hand:

```bash
cp research_seeds.example.yaml research_seeds.yaml      # then write your own topics
cd src && python3 -m piper_memory.seeds && cd ..
python3 src/piper_research/service.py --now --once      # one cycle now, whatever the time
python3 src/piper_research/service.py --now --cycles 5 --topic "How migrating birds navigate"
```

Maintenance commands:
- `--rejudge 500` runs the judge over findings stored before it existed. The service also does a
  few after each cycle (`research.rejudge_per_cycle`).
- `--tidy` prunes the open questions to `research.max_open_questions` per topic and marks the old
  cross-topic flags as reviewed.
- `--delete-topic "Name"` removes a topic and everything that belongs to it (the dashboard's
  delete does the same). Back up `data/memory/piper_memory.db` before upgrading Piper, too: a new
  version may upgrade the memory's format, and the older code can't read it afterwards.

**Mastery** (`src/piper_research/mastery.py`) is the goal for every topic - in Piper's research
role and her DNA (`system_dna.md`). Each current topic has a mastery checklist: recent research
finds little new, a solid evidence base (`research.mastery_min_findings`), synthesis done, its
insight questions tested, your guidance explored, and a stated position. When every item is met,
Piper **asks for mastery**; you confirm it on the dashboard, and she writes the topic's **mastery
brief** (bottom line, the established core with its findings, live conjectures and disputes,
edge-case questions worth exploring later, what her insights added) and moves it to **Deeply
Explored**. That frees one of the `research.max_current_topics` (8) slots. If you say **not yet**,
you write what else needs exploring; she turns it into priority questions, researched first, and
the checklist waits for them. Topics close to mastery get a little more of her time.
Piper proposes new topics (and sub-topics, when derived questions keep pointing the same way);
you add them - or your own - into free slots from the dashboard. A deeply explored topic can be
reopened, and its findings stay available to synthesis.

The rhythm can also be started by hand: `python3 src/piper_research/service.py --session`;
`--session 2` instead researches for two hours, writes a summary and stops. For plain timed research
instead, set `research.windows` in `config.yaml` (several allowed; a window may cross midnight) or
`PIPER_RESEARCH_WINDOWS="01:00-08:00"`.

### 5.6 Dashboard

A small web dashboard (`src/piper_dashboard/`, standard library only) runs next to the research
service - on piper-mind (`piper-dashboard.service`) - on port 8080: open `http://<piper-mind-ip>:8080/`
from any device on the home network. (`start_piper.sh` can also start one, `dashboard.enabled`.)

| View | Shows |
|---|---|
| Now | what Piper is researching, the rhythm's phase, the live activity feed |
| Reports | the daily and weekly reports; insight numbers in them open the insight's trail |
| Insights | what synthesis concluded. Click one for its **trail**: why those findings were chosen (each back to its source quote and page revision), the numbered reasoning steps and their checks, the grounding check, the questions it raised and whether research took them up, what happened since, and the model's raw thinking (collapsed) |
| Topics | **Current** (the slots, Add topic, Piper's proposals) and **Deeply Explored** (mastery briefs, Reopen). Each current topic: stage, thesis, evidence balance, her position and remarks, open questions (derived ones marked), and its **mastery checklist** - with Confirm / Not yet when Piper asks for mastery |
| Findings | every finding, filterable by topic, stance, relevance and source (or just the derived questions), with its quote, the judge's reason and a link to the exact page revision |
| Sessions | one row per research session: cycles, new findings, off-topic, rejected, the session summary |
| Notable | contradictions and links between topics, with a "mark reviewed" button |
| Graph | the concept graph: click a concept for its links and the findings behind them |

The dashboard's writes (mark reviewed, delete or add a topic, confirm mastery / not yet, reopen,
accept a proposal) are requests: Piper's research service carries them out between steps.

Keep it on the home network: the topics are personal. It can also run on another machine against
a copied database: `python3 src/piper_dashboard/server.py --db path/to/piper_memory.db`.

### 5.7 Tests

The unit tests need no audio hardware, GPU or models, so they run on any machine:

```bash
pip install -r requirements-dev.txt
python3 -m pytest
```

---

## 6. piper-watch: Piper's body

> **Design history:** the 24-LED light ring, and after it the two round GC9A01 eyes with an SH1122
> OLED mouth, have been retired. The face is now one screen. The XIAO + OLED mouth still works
> (`head.mouth_port: auto`) but is not part of the active design. See the decision log (Claude project docs).

**piper-watch** is Piper's body: a printed base holding the reComputer, a turret that **pans** on a
pillow-block bearing, and a head that **tilts** - each moved by an STS3215 bus servo. On the head
are the **face screen** and the **camera**. The servos (through their USB adapter), the face screen
and the camera all plug into the Jetson's USB ports; the SP-200 speakerphone sits on the desk.

Today's head is a working prototype with the 7" CrowPanel on the bench. The final head carries the
**5" CrowPanel ESP32-P4** (800 x 480) with the C920's board as a brow band above it (camera above the
screen keeps it close to the drawn eyes); its parts are being designed (piper-watch `hardware/cad`).

### 6.1 Architecture

Face following has to run continuously - the head keeps following you while Piper listens, thinks and
speaks - so it can't live inside the conversation loop, which waits on each step. The work is split
into separate processes:

```text
                      ┌───────────────────────────── Jetson ─────────────────────────────┐
 C920 camera ──USB──► │ VISION SERVICE (piper_vision)                                    │
                      │   capture → face detection → tracking → recognition → follow     │
 servo adapter ◄─USB─ │      │ pan / tilt targets (servos.py)      │ who is present      │
 (pan + tilt)         │                                            ▼ (vision hub)        │
                      │ HEAD LINK (piper_head/link.py)        ASSISTANT (src/main.py)    │
 face screen ◄──USB── │   owns the face's serial port  ◄─────  MOUTH (mode, status,      │
 (ESP32-P4)           │   restores its state on reconnect        speech level)           │
 SP-200 ◄─────USB──── │  ◄── microphone / speaker ──►  ASSISTANT                         │
                      └──────────────────────────────────────────────────────────────────┘
        QUANTUM (Ollama: conversation, research)        piper-mind (research + dashboard)
```

- **Vision service** - owns the camera; detects faces (OpenCV YuNet), tracks the one Piper attends
  to, recognises people she has met (SFace face prints in `faces/`), and turns the head towards them
  through the servos (`vision.follow`: limits, gain, deadband, home). A live preview with boxes is
  served on `http://<jetson>:8081/`.
- **Head link** - the only program talking to the face screen. It passes MOUTH messages from the
  assistant and tools on as the screen's text commands, sets the mouth style, and restores the last
  mode and status whenever the screen (re)connects.
- **Assistant** (this program) - decides the face from its state and mood (`piper_head/expression.py`),
  asks vision who is there, and holds research on piper-mind during a conversation.

Each part can be restarted or tested on its own: the face with `tools/head_cli.py`, the servos with
`python3 src/piper_head/servos.py`, vision without the servos (`vision.follow.enabled: false`).

### 6.2 Seeing

| Without the robot                                           | With piper-watch                                                                                                                       |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| The person is identified only if they say "my name is ..."  | Piper **recognises faces** and loads that person's profile automatically, so she greets people by name and remembers context per person |
| Piper doesn't know anyone is there until it hears its name  | Piper **notices someone arriving** and turns to look                                                                                    |
| No sense of who is speaking when several people are present | The head turns to the face she is attending to                                                                                         |

Vision sees **faces only**: Piper can't describe objects or scenes (her DNA tells her so). Asking her
about something held up to the camera ("what am I holding?") is a proposal - a snapshot plus a vision
model in Ollama - not built.

Recognition stays on the Jetson: face prints are stored locally in `faces/` (git-ignored), only after someone gives their name, and "Piper, forget me" deletes a person.

**Meeting people** (`src/piper_skills/meet_person.py`): Piper only helps people she knows by name.

- A stranger in view is asked their name straight away; no wake word is needed. Until she has it, she answers nothing else ("first I need to know who you are").
- She checks the name ("Steve? Did I get that right?"). A correction ("no, it's Stephen") or a spelling ("S-T-E-V-E") also works.
- She then asks them to look at her while vision stores five looks at their face. She also starts a profile for them in `profiles/<name>.md`.
- Someone who won't say is told they can say goodbye. After "goodbye" she leaves them alone until they walk away and come back.
- Someone she knows gets "Welcome back, Steve! How can I help you today?", but not again if they were there in the last 10 minutes.
- "Who am I?" and "forget me" also work.
- Without the vision service, she behaves as before.
- Settings: `assistant.require_known_person`, `assistant.welcome_back_minutes`, and the `vision.recognize` / `match_threshold` / `enroll_*` keys.

### 6.3 The face: state and mood

The face screen (piper-watch `firmware/face_p4`) draws RoboEyes-style eyes over a spectrum-bar mouth,
with a small status symbol in a corner. `src/piper_head/expression.py` picks the mouth's mode and the
symbol from the assistant's state:

| Assistant state | Mouth                                                                          | Symbol                     |
| --------------- | ------------------------------------------------------------------------------ | -------------------------- |
| IDLE            | **blank** - the screen is dark (no burn-in)                                    | -                          |
| woken while research runs | **"Research in progress..."** while she pauses it (below)            | -                          |
| ENGAGED         | neutral (a smile for 3 s when a known face is welcomed)                        | listening: + / −           |
| PROCESSING      | neutral                                                                        | thinking: three dots       |
| SPEAKING        | talk; **happy talk** when the mood is "warm" (greetings, goodbyes, chat); **excited talk** when it is "excited" | none |
| LLM failed      | neutral                                                                        | confused: ??? for 4 s      |

The **mood** picks the talking mouth: `neutral` → talk (speech bars up to half height), `warm` →
happy talk (arched happy eyes, a laugh as she starts), `excited` → excited talk (bars at full height,
eyes a little wider). Nothing sets "excited" yet except `say(text, "excited")` and the test phrase.
Other moods (curious, uncertain, concerned, sleepy) were ideas for the light ring; they have no look
on the face yet.

Speech moves the mouth only in the three talk modes, so Piper never talks while frowning or surprised.
Modes: neutral, talk, happytalk, excitedtalk, smile, frown, surprise (big round eyes; the "O" pops open
and settles small), sleep, happysleep, research (the message), blank. The face goes blank by itself
after 5 minutes without any command.

**Pausing research for a conversation.** Research (on piper-mind) and conversation share one Ollama -
QUANTUM's 12 GB RTX 5070 - and it can't hold both models. Research runs only from 02:00 to 08:00
(section 5.5); when she is woken during those hours, the wake word holds research:

> **You:** Piper, can I ask you a question?
> **Piper:** Sure, give me a second to pause my research. *(face: "Research in progress...")* ... OK, I'm ready. How can I help?

The Orin asks piper-mind's dashboard to hold research (`POST /api/research/hold`, renewed every minute; `src/piper_brain/research_hold.py`). The research service drops any model call in progress within a second and waits (`src/piper_research/hold.py`); the Orin then unloads the research model on QUANTUM and loads the chat model. After "goodbye" or 20 s of silence the hold is released (`/api/research/release`), the screen goes dark and research carries on. A hold that isn't renewed expires after `research.hold_s` (300 s), so a crashed Orin can't leave research paused. Outside research hours she just answers. Settings: `research.mind_url`, `research.hold_s`.

**"Piper, think carefully: ..."** sends that question to the research model with its thinking on - ~10-15 s (plus loading it), better reasoned. Everything else uses the quick chat model.

Testing the face by hand (the head link must be running) and by voice:

```bash
python3 tools/head_cli.py mouth smile                  # any mode
python3 tools/head_cli.py mouth excitedtalk --level 200 # with speech levels (0-255)
python3 tools/head_cli.py mouth demo                   # its demo loop; "mouth demo-off" stops it
python3 tools/head_cli.py status thinking              # idle, listening, thinking, confused
```

| Say                                         | Piper                                                 |
| ------------------------------------------- | ----------------------------------------------------- |
| "Piper, smile" / "frown" / "look surprised" | shows that face for 5 s                               |
| "Piper, show me your faces"                 | every mouth mode, then every status symbol (~24 s)    |
| "Piper, test normal talk"                   | says a sentence with the normal talking mouth         |
| "Piper, test happy talk"                    | says a sentence with the happy talking mouth          |
| "Piper, test excited talk"                  | says a sentence with the excited talking mouth        |
| "Piper, go to sleep" / "Piper, wake up"     | the sleep face for 2 minutes, then dark / says she's awake |
| "Piper, think carefully: <question>"        | answers with the research model, thinking (slower)    |

(While Piper is running she re-sends her own face state every minute, so a `head_cli` change lasts until then or until her state changes.)

### 6.4 Messages and settings

Local clients (the assistant, tools) talk to the head link over `127.0.0.1:head.port` with
newline-delimited JSON; the link translates for the face screen, which speaks one-line text commands
(the same protocol as the old OLED mouth):

| From → to       | Message / command | Example |
| --------------- | ----------------- | ------- |
| client → link   | `MOUTH`           | `{"t":"MOUTH","mode":"happytalk","symbol":"idle","level":0.4}` |
| link → face     | `M<mode>` `I<status>` `L<0..255>` `D1`/`D0` `S<style>` `?` | `Mhappytalk`, `Iidle`, `L102` |
| face → link     | its state, on `?` | `mouth: LIVE, mode 2 happytalk, status 0 idle, style spectrum, contrast 160` |

The face also understands `G<x>,<y>` (look towards a point for 3 s) and `C<0..255>` (brightness);
piper_assistant doesn't send them yet. The servos have no messages of their own: the vision service
drives them directly over the adapter (Feetech's SCServo protocol).

```yaml
head:
  mouth_style: spectrum      # the face screen only has spectrum (the OLED mouth also has lips)
  mouth_port: /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0   # the face's CH340; "auto" finds only
                                                                 #   Espressif USB (the XIAO OLED mouth)
vision:
  follow:
    enabled: true
    pan_limits: [-60, 60]    # servo degrees: pan + = left, tilt + = the head up
    tilt_limits: [-16, 50]
    home: [0, 0]
```

---

## 7. Roadmap

0. **Clean-up** - *done*: config-driven settings, working TTS-engine choice, weather cache,
   one config loader, unit tests.
1. **Face and head link** - *done*: the head link, the face screen's expressions from the
   assistant's state and mood (the light ring and the separate eyes/OLED mouth were retired).
2. **Motion** - *done*: pan and tilt with two STS3215 servos, limits and home, a servo check tool.
3. **Vision** - *done*: detection and tracking steering the head; recognition and enrolment;
   who-is-present events replacing "my name is ..."; a live preview page.
4. **Final head** - *next*: the 5" face screen, the C920 as a brow band, the printed head v7
   (a smoked cover over screen and camera is to be tested).
5. **Conversation quality**: chat on `qwen3:14b` and a working research hold - *done*; next: a
   dedicated wake-word model (openWakeWord), streaming speech (start speaking before the whole reply
   is generated), talking over Piper using the SP-200's echo cancellation, research findings
   available to the conversation (she can't recall them yet).
6. **Research and curiosity**: research limited to 02:00-08:00 for now; proposal to move it to
   curious-george on Colab (section 9, the decision log (Claude project docs)).

---

## 8. Configuration notes

- `system_dna.md` holds Piper's persona, her purpose (mastery as the goal of her research) and
  voice rules; it is read at start-up.
- `profiles/<name>.md` is loaded into the conversation for that person (file name in lower case, e.g. `steve.md`).
- `daily_journal.md` records state changes, local answers and LLM replies, grouped by date (git-ignored).
- `config.yaml` is shared by every machine (the Jetson, piper-mind); change it on the PC, push, and
  pull everywhere. Settings for one machine only can use the environment variables in
  `src/piper_brain/config.py` (e.g. `PIPER_RESEARCH_ENABLED=false` on the Jetson).

---

## 9. Related projects

- **[piper-watch](https://github.com/stvenmobile/piper-watch)** - the robot head: hardware
  design, CAD, wiring, and the firmware (the face screen, the old OLED mouth, servo set-up).
- **[curious-george](https://github.com/stvenmobile/curious-george)** - research into machine curiosity: choosing what to study and measuring whether studying it taught the model anything. It exposes its tools through an **MCP server**; the plan is for this runtime to use them while Piper is idle (choose a topic, study it, log what was learned), to talk about what it has been curious about. Moving Piper's research itself there (a bigger model on a Colab GPU) is proposed. Where study runs
  (Jetson, PC GPU or Colab) and how learned adapters reach Ollama are still open.
