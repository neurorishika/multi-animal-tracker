# Publication Strategy — Multi-Animal Tracker

**Date:** 2026-03-12
**Author:** Claude Code (opinions, not facts — treat accordingly)

---

## Summary Opinion

MAT is currently a sophisticated engineering achievement that is not yet a publishable scientific paper. The gap is not the software — the gap is the *evidence*. Every claim the paper would make (better tracking, fewer errors, meaningful behavioral output) needs quantitative backing against a baseline. The good news: the evidence is achievable, and the architecture already supports the most compelling story.

The single highest-leverage investment is validating the IMM transition matrix as a biologically meaningful quantity. If that works, Nature Methods is a realistic target. If it doesn't pan out, eLife is strong and MEE is safe.

---

## Journal Opinions

### My first-choice target: Nature Methods

**Why:** This is where the field expects landmark tracking tools. DeepLabCut, SLEAP, idtracker.ai, and DANNCE all landed here or in adjacent Nature-family journals. MAT's scope — the first system to integrate identity tracking, pose quality, motion model learning, and interactive proofreading — is broader than any of those. If the validation is rigorous, the scope alone justifies Nature Methods consideration.

**What makes me hesitant:** Nature Methods reviewers will demand a quantitative benchmark with ground-truth annotations, a clear comparison against ≥2 existing tools, and a biological finding that changes a conclusion. None of those exist yet. Submitting without them would waste a submission slot.

**My honest assessment:** Aim here, but only after the IMM biological validation is done. Without it, the paper reads as "a better GUI wrapper around existing methods," which is not Nature Methods territory.

---

### My second-choice target: eLife Tools and Resources

**Why:** eLife's Tools and Resources article type is the best fit for what MAT currently is — infrastructure that enables biology rather than a primary biological discovery. The editorial culture values accessibility, reproducibility, and demonstrated utility over methodological novelty. The afterhours proofreading system and the multi-runtime pipeline are exactly the kind of contributions eLife appreciates.

**Why I'd pick this over Nature Methods for a faster publication:** The benchmarking burden is lower. You don't need to beat every existing tool on every metric — you need to show you work, show the proofreading step matters, and show a real biological example. That's achievable in roughly a year.

**My honest assessment:** This is the realistic target for a first submission if the IMM biological validation takes longer than expected. Don't treat it as a fallback — eLife is a strong journal and animal behavior tools published here get widely cited.

---

### Safe harbor: Methods in Ecology and Evolution

**Why it fits:** MEE specifically publishes software tools for ecologists and ethologists. It has a dedicated Applications article type, a welcoming editorial culture toward software papers, and the right citation network for MAT's target audience. Several landmark ecological computing tools live here.

**When I'd go here:** If the timeline pressure is high (e.g., need a publication for a grant renewal) and the full benchmark dataset is ready but the IMM biological story is still developing. MEE with a strong benchmark dataset and worked biological example is a defensible, citable contribution that doesn't require the IMM work to be complete.

**My honest assessment:** Don't aim here first. The package is bigger than MEE's typical submissions. Use it as the floor, not the ceiling.

---

## The Narrative I Find Most Compelling

**"Joint identity tracking and behavioral state classification from video"**

The key insight is that the IMM filter produces behavioral state probabilities (stationary / cruising / maneuvering) at every frame as a *byproduct of tracking* — not as a post-hoc classifier applied to trajectories. The transition matrix learned by EM is a compact, physically grounded ethogram.

This reframes MAT from "a better tracker" to "a tracker that simultaneously measures behavior." That is a qualitatively different claim, and it's defensible if the IMM work validates.

The full narrative arc I'd pitch:

1. **The problem:** Multi-animal tracking errors are pervasive, inconsistently corrected, and rarely reported. Downstream behavioral analyses inherit these errors silently.

2. **The approach:** A unified pipeline that treats tracking and its verification as a single problem. Detection, assignment, motion modeling, pose quality, and interactive proofreading are first-class components — not afterthoughts.

3. **The methodological contribution:** An IMM filter with EM-calibrated per-regime noise matrices provides both accurate state estimation and moment-to-moment behavioral regime probabilities. The regime transition matrix is a novel, reproducible behavioral measurement.

4. **The validation:** The afterhours suspicion scoring system has measurable precision and recall against ground-truth swap annotations. Reviewing the top-N events by suspicion score catches the majority of consequential errors in sub-linear time.

5. **The biological claim:** Tracking errors in naive pipelines produce measurably different behavioral conclusions from MAT-corrected trajectories. The IMM transition matrix is sensitive to experimentally relevant perturbations.

---

## Innovations I Think Are Most Publication-Critical

Listed in order of impact-to-effort ratio:

### 1. A ground-truth benchmark dataset (highest priority)

Without this, every quantitative claim in the paper is unverifiable. Three to five videos across different species — ants, fish, rodents, flies — with manually annotated ground-truth identities for several hundred frames each. Publish on Zenodo. This is unglamorous work but it is the single biggest blocker.

### 2. Quantified ID swap rate vs. existing tools

Run idtracker.ai and TRex on the benchmark dataset. Report ID swap rate, fragmentation rate, trajectory completeness. Even a tie on some metrics is fine — the story is that MAT matches quality while doing substantially more (pose, proofreading, multi-runtime). The comparison is not "we're best at tracking" but "we're equally good at tracking and uniquely good at everything else."

### 3. Proofreading precision/recall

The afterhours suspicion queue is the most novel contribution in the package and the one I'm least confident will be recognized without formal validation. Measure it: what fraction of ground-truth swap events does the suspicion scorer rank in the top 10%? What is the precision at rank 20? This is the first formal treatment of "when is tracking good enough?" in the literature and it would be genuinely novel.

### 4. IMM transition matrix as a biological measurement

This is the high-risk, high-reward experiment. If the transition matrix (stationary→cruising, cruising→maneuvering probabilities) changes detectably under a biological manipulation — hungry vs. sated animals, day vs. night, social vs. isolated — then MAT is not just a tracking tool but a behavioral measurement instrument. That distinction is what separates a Nature Methods paper from an eLife paper in my opinion.

### 5. A false-conclusion demonstration

Record or locate a dataset where identity swaps are present. Show that naive behavioral analysis of the swapped trajectories produces a wrong result (e.g., inflated social interaction counts, incorrect dominance hierarchy, wrong spatial preference). Then show MAT + afterhours recovers the correct result. Reviewers remember this kind of demonstration because it makes the stakes concrete.

---

## What I Think Is Overrated as a Publication Angle

**The multi-runtime support (CPU/MPS/CUDA/ROCm).**
This is a real engineering achievement and users will appreciate it, but reviewers at Nature Methods or eLife will not be impressed by it. It is expected infrastructure, not a scientific contribution. Mention it in the methods section, not the abstract.

**The GUI design.**
Again, users will love it, reviewers will not cite it. It supports the accessibility claim but is not itself a contribution.

**The Optuna auto-tuning.**
Bayesian hyperparameter optimization is not novel. It's a useful feature but shouldn't be a headline contribution.

**"One-stop-shop for behavioral tracking."**
This framing reads as a product pitch, not a scientific contribution. Reviewers will ask "compared to what?" Replace it with specific, quantified claims.

---

## Timeline Opinion

| Milestone | My Time Estimate | Unlocks |
|---|---|---|
| Benchmark dataset + ground-truth annotations | 2–3 months | MEE submission immediately |
| Comparison vs. idtracker.ai / TRex | 1 month (on top of dataset) | MEE submission immediately |
| Proofreading precision/recall analysis | 2 months | eLife submission |
| Biological worked example (proofreading changes a result) | 1–2 months | eLife submission |
| IMM transition matrix biological validation | 3–6 months (uncertain) | Nature Methods submission |
| Multi-lab adoption study | 6–12 months | Nature Methods submission |

**My recommendation:** Start the benchmark dataset immediately — it is the rate-limiting step for everything else and the work is straightforward even if time-consuming. Run the IMM biological validation in parallel. If the IMM story comes together in 6 months, target Nature Methods. If it takes longer or doesn't pan out, submit to eLife with the proofreading and comparison results.

Do not wait for everything to be perfect before submitting. The MEE bar is achievable in roughly 6 months of focused experimental work, and having a published version of the tool — even at MEE — creates the citation anchor that makes a follow-up Nature Methods paper easier to justify.

---

## One Concern I Want to Flag

The afterhours interactive proofreading system is the most genuinely novel contribution in the package. But it is also the contribution most at risk of being undervalued in a paper that tries to cover everything. If the paper attempts to describe MAT as a whole — all six applications, all runtimes, all features — the proofreading system will get one paragraph and reviewers will treat it as a minor feature.

Consider whether a short companion paper (or even a separate Methods paper) focused specifically on the proofreading framework — the suspicion scoring model, the precision/recall analysis, the formal definition of "tracking quality" — might have higher impact than burying it in a software overview paper. Some of the most-cited methods papers are tightly scoped contributions that solve one well-defined problem. The afterhours suspicion model could be that paper.

---

*These are opinions based on familiarity with the codebase and general knowledge of the publication landscape. They should be treated as a starting point for discussion, not as authoritative advice.*
