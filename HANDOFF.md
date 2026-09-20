# Handoff: offline school tutor (kiwix-ai-tutor)

Written 2026-09-19 for the session that starts implementation after a `/clear`. Read this first, then
`docs/plan/offline_tutor_implementation_plan.md`. You have no prior context; everything you need to
start is here or in `docs/plan/`.

**You are the orchestrator. Sub-agents write the code. Tests come first, always.**

---

## 1. What we are building

An offline tutoring app for a student with no internet. A local llama-server runs a small model; the
app retrieves evidence from Kiwix ZIM archives (offline Wikipedia, textbooks, Stack Exchange), feeds
it to the model as citations, and teaches rather than answers. Two model-facing tools only:
`research` and `calc`. Development happens on this Windows laptop; the shipping target is a Dell
with a GTX 1060 6 GB running Linux Mint.

Authoritative documents, copied into this repo so you do not depend on the user's Downloads folder:

| File | What it governs |
|---|---|
| `docs/plan/offline_tutor_implementation_plan.md` | **Primary.** Milestones, work packages, order of work, cross-platform rules. v1.1. |
| `docs/plan/offline_tutor_spec_v0.3.md` | Architecture, budgets, gates. Wins on budgets, caps, thresholds. |
| `docs/plan/offline_tutor_kiwix_reuse_plan.md` | Donor code and algorithms. Wins on algorithm choices. |

When the spec and the reuse plan disagree, stop and ask the user. That rule is from the plan (§10.11)
and it is not optional.

## 2. Working agreement for this project

### 2.1 TDD, strictly

Every work package runs red → green → refactor:

1. **Red.** A sub-agent writes failing tests from the acceptance criteria in the plan. No production
   code in this step. You review the tests before any implementation starts: tests that assert the
   wrong thing are worse than no tests.
2. **Green.** A different sub-agent makes them pass, touching only what is needed.
3. **Refactor / review.** A third sub-agent reviews the diff against the cross-platform rules
   (plan §5), the license rules (§10.7), and the "no generic RAG framework" rule (§10.9).

No production code is written without a failing test that demands it. If a work package's acceptance
criteria are too vague to test, that is a question for the user, not a license to improvise.

### 2.2 Sub-agents: Sonnet only

Delegate implementation to sub-agents with `model: "sonnet"` on every `Agent` call. This is a hard
constraint from the user.

- Use `subagent_type: "general-purpose"` with `model: "sonnet"`. For read-only sweeps, `Explore`
  with `model: "sonnet"` is fine.
- **Never use `subagent_type: "fork"`.** A fork always inherits the parent's model, which would run
  Opus and violate the constraint.
- Sub-agents do not share your context. Every prompt must be self-contained: name the files, quote
  the acceptance criteria, state the cross-platform rules that apply, and say what "done" means.
- Independent work packages run in parallel: send multiple `Agent` calls in one message. A0 and B1
  are explicitly parallel per the plan. Dependent work does not overlap.
- You (the orchestrator) run the tests yourself and read the diffs. Do not take a sub-agent's word
  that tests pass. Their report is a claim; the test run is the evidence.

### 2.3 What stays with you, not sub-agents

- Deciding a work package is done, and writing the `docs/` note that closes it.
- Anything touching the user's GitHub, the runtime at `C:\git\bonsai`, or the archives on `D:`.
- Milestone reports. M0, M2, M3, M4 and M5 go to the user before the next milestone starts.

## 3. Machine state, verified 2026-09-19

### 3.1 This repository

`C:\git\kiwix-ai-tutor`, git initialized, branch `main`, **zero commits**, remote `origin` =
`https://github.com/Keoian/kiwix-ai-tutor.git`. Only `HANDOFF.md` and `docs/plan/` exist.

The first commit should establish the layout from plan §2 and the dual-OS CI from §5. CI on both a
Windows and a Linux runner is required "from day one", so it belongs in the first or second commit,
not later.

### 3.2 The model runtime (read-only)

`C:\git\bonsai` is the working Bonsai 8B installation. **The plan declares it read-only.** If it
needs changes, copy it to `runtime/bonsai/` via `scripts/runtime_init.ps1` and change the copy.

Measured today, on this machine (i9-9980HK, Radeon Pro 5500M 8 GB, Windows 11, AMD driver
32.0.12019.1028). These numbers are WP-A0's reference data, already collected:

| Fact | Value |
|---|---|
| Model file | `C:\git\bonsai\models\Bonsai-8B-Q1_0.gguf`, 1.16 GB, 8.19B params |
| SHA-256 | `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54` |
| Server build | `b10716-ee2aeae98` (PrismML fork, Vulkan backend) |
| Launchers | `start-server-32k.ps1` (32k, q8_0 cache, 3,584 MiB VRAM), `start-server-65k.ps1` (65k, 6,064 MiB) |
| **Use the 32k launcher** | It is the plan's profile (§0.2) and the one that fits the Dell's 6 GB |
| Decode, short context | 31.9 tok/s |
| Prompt, short context | 175 t/s |
| Sampling defaults | temp 0.5, top-p 0.9, top-k 20 (model card) |
| Flags already set | `--jinja`, `--slots`, `-np 1`, `--path <webui>` |
| Thinking | Not a reasoning model; no thinking by default |

The server is an HTTP black box. The app must never link the runtime or branch on model name.

**The flash-attention trap, and it will bite WP-C1.** Vulkan flash attention has no accelerated path
on this GPU (upstream llama.cpp issue, not a Bonsai bug). With it on, prompt processing collapses as
context fills: 21 t/s at 4k depth, 11 t/s at 8k, and a 25,200-token prompt never finished in 40
minutes. With `-NoFlashAttn` the same prompt was read in 5 minutes at 82.7 t/s, but decode drops.

Consequences for the app:
- **Prefill of a large uncached prompt is brutally expensive here.** The plan already calls the
  append-only prompt layout (WP-C1) important; on this machine it is the difference between usable
  and not. Any test that rebuilds a long prompt from scratch will look like a hang.
- Keep integration tests well under 8k tokens of context unless the test is specifically about depth.
- The Dell's CUDA build will not share this pathology. Do not tune the app around it; just do not let
  it make your tests look broken.

Two operational cautions from today: a long-running deep-context prompt coincided with an AMD driver
popup, and killing `llama-server` mid-prompt is a plausible cause of a driver reset. Prefer graceful
shutdown. Do not run multi-hour GPU jobs unattended without telling the user.

### 3.3 Archives

`D:\kiwix` on an external 5 TB HDD, 65 archives, 509 GB. Present and confirmed today: Lumen
Learning courses (12 GB), `wikipedia_en_all_maxi_2023-10` (103 GB), `math.stackexchange`,
`matheducators.stackexchange`, `wikiversity`, `wikihow`, `gutenberg`, `khanacademy`, and the rest of
the Stack Exchange set.

**Missing, and both need downloading:** `wikipedia_en_simple_all_maxi` (~2 GB, the tier-1 archive
that all latency numbers depend on) and `wikibooks_en_all_maxi` (~4 GB). The `wikibooks_af` on the
drive is Afrikaans and is not a substitute. Tier 1 belongs on the `C:` SSD, which has 129 GB free.

**The user's internet is unreliable right now.** Their ISP is under a DDoS and they were downloading
over a phone hotspot. Ask before starting multi-GB downloads, use `curl -C -` so interrupted
transfers resume, and verify checksums after.

### 3.4 Tooling present

Python 3.12.10 (`python` and `py -3`), git 2.x with git-lfs 3.7.1, PowerShell 5.1, Git Bash, MSVC
2022 Build Tools, CMake, Ninja, Vulkan SDK 1.4.357, headless Edge for UI screenshots.

**Absent:** `uv`, `node`, `npm`, `gh`. No GitHub CLI and no API token, so anything GitHub-side beyond
`git push` needs the user. `python-libzim` wheel availability for 3.12 on both OSes is unverified and
is WP-B1's first check; the plan says report it as a blocker rather than work around it.

## 4. House rules

**Git and GitHub.**
- Commit as `Keoian <65092962+Keoian@users.noreply.github.com>`. Set it repo-locally:
  `git config user.name Keoian && git config user.email 65092962+Keoian@users.noreply.github.com`.
  Never commit with the user's personal email address.
- End commit messages with the attribution lines the session reminder gives you.
- **Ask before every push.** The user was surprised by a push earlier today even though they had
  asked for one. Authorization does not carry between pushes.
- Never commit: ZIM archives, GGUF models, `runtime/`, anything in `D:\kiwix`.

**Writing for the user.** They read the final message, not the tool calls. Lead with the outcome,
give measured numbers rather than adjectives, and say plainly when something was not verified. They
value being told what was inferred versus measured; they have corrected work on exactly that.

**Scope.** Build the work package in front of you. Do not build a generic RAG framework, do not add
ranking flags without the eval table that justifies them, and do not start a downstream work package
because you are blocked on the current one. Report the block.

## 5. Start here

Two work packages run in parallel and neither depends on the other.

### WP-A0 — Wire up the existing dev runtime (½ session)

Most of the measurement is already done in §3.2. What remains: capture the exact launch command,
port and flags into `config/dev.toml` and `scripts/serve_dev.ps1`, record the fork commit and
backend from the startup log, and write `docs/dev_runtime.md`. Do not modify `C:\git\bonsai`.

Tests first, even here: `config/dev.toml` gets a loader with a test that asserts the profile is
32,768 context with a q8_0 cache, and a test that the configured runtime path is read from config
rather than hard-coded.

### WP-A1 — Tool-call and template verification (1 session) — **gates M0**

20 scripted requests against the dev server with the `research` and `calc` schemas: 10 that should
produce a tool call, 10 that should not. Count parsed, malformed, and prose-shaped-like-a-call.
Verify `/tokenize` and `/apply-template`. If malformed exceeds 2%, the grammar-constrained path is
chosen instead, and that decision gets recorded.

This is measurement, not app code, but the counting harness is code and gets tests. Expect the 1-bit
model to be the weak link: its published tool-calling score (BFCL 65.7) is well below the ternary
model's 73.9, and the plan anticipates that Q1_0 may simply be worse at this. Malformed calls here
are data, not bugs to debug.

### WP-B1 — Donor inventory and vendoring (1–2 sessions)

Check out OpenZIM MCP at its latest real tag (the reuse plan's v3.3.4 does not exist; v2.5.3 is the
newest as of 2026-07-01). Write `docs/donor_inventory.md` mapping every reuse-plan reference to a
real path. Vendor the smallest cohesive units into `retrieval/zim/` with upstream MIT headers intact
and `THIRD_PARTY_NOTICES.md` updated in the same commit.

**Check `python-libzim` wheels on Windows for Python 3.12 before anything else.** If there is no
wheel, stop and report; do not build from source as a workaround.

Acceptance: `from tutor.retrieval.zim import archive` imports with only `libzim`, an HTML parser and
stdlib, on Windows.

### Suggested first moves

1. Read `docs/plan/offline_tutor_implementation_plan.md` in full. §5 and §10 are the rules you will
   be judged against.
2. Decide the repo skeleton and CI from plan §2 and §5, and get that first commit reviewed by the
   user before the code lands on top of it.
3. Launch WP-A0 and WP-B1 sub-agents in parallel, Sonnet, tests first.
4. Report at M0 before touching workstream C.

## 6. Open questions for the user

Ask these early; they change what gets built.

1. **Tier-1 archive.** Confirm downloading `wikipedia_en_simple_all_maxi` (~2 GB) and
   `wikibooks_en_all_maxi` (~4 GB), given the unreliable connection. Nothing in workstream B's
   acceptance criteria can be finished without the Simple Wikipedia archive.
2. ~~CI on both OSes~~ **Decided 2026-09-19: yes, use GitHub Actions.** The user approved it for this
   repo specifically, because the workflow is ours rather than inherited and it only runs unit tests.
   Keep it that way: one small workflow running the suite on `windows-latest` and `ubuntu-latest`,
   nothing that builds artifacts, publishes packages or pushes images. GitHub's runners have no GPU
   and no archives, so llama-server tests, real ZIM files and every performance gate stay on the
   user's machines. If Actions turns out to be disabled on the repo, that needs the user; there is
   no `gh` CLI here.
3. **Python version pin.** The plan defers to OpenZIM MCP's `.python-version`, subject to
   `python-libzim` wheels on both OSes. 3.12.10 is what is installed. Confirm at WP-B1.
