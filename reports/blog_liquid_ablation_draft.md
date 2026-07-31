# Blog Draft: "I Killed My Own Thesis With Data" (Liquid/CfC Ablation)

**Purpose:** BACKLOG I.6 #56 — "İngilizce teknik blog: 'I killed my own thesis with data' (1500-2000 kelime)". Unlike most of Part 6's deliverables, this one is fully writable NOW because the underlying evidence already exists (the 12-seed Liquid ablation is already measured, not pending a future run). Draft below; final polish/publication is a human decision (BACKLOG #56 is filed under I.6 Publication/Career).

---

## I Killed My Own Thesis With Data

When I set out to build MertFormer Titan, one of its core architectural bets was Liquid/CfC — continuous-time recurrent mixer layers threaded through a BitNet-quantized, sparse-MoE transformer. The pitch, at least in my head, was that a continuous-time recurrent inductive bias would give the model something attention alone doesn't have: a genuine notion of state that evolves smoothly through a sequence, rather than being reconstructed fresh at every position via attention weights.

It's a nice story. I ran the experiment to prove it. It didn't.

### The setup

I ran a 12-seed ablation at toy scale: same architecture, same data, same training budget, Liquid layers on vs. off, averaged over enough random seeds to not be fooled by a single lucky or unlucky run. This is about as basic as ablation methodology gets, and that's deliberate — I didn't want a result I'd later have to defend as "well, actually, if you squint at seed 7..."

### The result

No measured accuracy benefit. None. Across 12 seeds, the Liquid-enabled runs did not outperform the dense baseline on the metrics I was tracking. And the Liquid path was measurably slower — about 30% slower in wall-clock terms, which is not a rounding error, it's a real tax on every single training step.

I want to be precise about what this does and doesn't say, because it would be easy (and dishonest) to overstate it in either direction:

- It does NOT say Liquid/CfC mixers are useless as an idea. Toy-scale ablations are exactly that — toy scale. A continuous-time recurrent bias might behave completely differently at 3.67B parameters on real data than it does on a small ablation harness. I don't know, because I haven't run that experiment yet (it's on the list — a real 45K-scale run is the only way to find out for certain, and it's expensive enough that I'm not spending it carelessly).
- It DOES say that, as measured, right now, at the scale I could afford to test cheaply, the thing I built specifically because I believed it would help — didn't show any sign of helping, and it cost real training speed to include.

### Why write this down instead of quietly dropping it

Because the alternative — not publishing a negative result, quietly ripping the layers out, and never mentioning it — is exactly the kind of selective reporting that makes ML research results untrustworthy in aggregate. If I only ever tell you about the things that worked, you have no way to calibrate how much to trust the things I tell you worked. A negative result, honestly reported, with its scope explicitly bounded ("toy scale, inconclusive, no speed claim"), is worth more to anyone reading this than a cherry-picked win.

It's also, frankly, a decent proxy for whether I can be trusted to report results I don't like as carefully as results I do. I spent a good chunk of this project's history finding and fixing places where a "fix" was actually cosmetic — code that looked like it was doing something (a `getattr` with a fallback, a safety check gathering a hardcoded dummy tensor instead of a real value) but wasn't. Reporting an ablation result that undercuts my own architectural thesis is the same discipline pointed at myself instead of at old code.

### What happens next

The Liquid/CfC layers are still in the canonical architecture as of this writing — whether they stay in the actual 45K training run is an open, explicitly-flagged decision (not yet made, not buried), weighed against the real GPU-hour cost a ~30% slowdown represents on an expensive one-shot run. If they go in, it's with eyes open about what the toy-scale evidence actually says. If they come out, it's because the cost-without-measured-benefit case won at this stage, not because the idea was bad — just unproven, so far, at the scale that would have made the case for it.

I'd rather ship a smaller, well-evidenced claim than a bigger one I can't back up. That's the whole point of doing the ablation in the first place.

*(2026-07-31 update: an independent, small-scale external test found the CfC mixer running roughly an order of magnitude slower than attention in isolation, and flagged a mechanism — the mixer's sequential recurrence scales with sequence length while attention doesn't — that this repo hadn't written down before. It's not evidence about this model specifically, but it's a reason the ~30% figure above might be optimistic at the scale that matters. See `BACKLOG.md`/`ABLATION.md` for the full, hedged version before repeating a number from this paragraph anywhere.)*

---

*Word count: ~700 (draft — expand with concrete numbers/plots from ABLATION.md before publishing; target 1500-2000 per the backlog spec).*
