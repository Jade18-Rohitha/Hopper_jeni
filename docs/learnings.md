# The week I spent beating cuDNN on GQA backward — in plain terms

*A retrospective on the H200 flash-attention backward kernel. Written for myself, so future-me
remembers not just what worked but why the wrong turns were wrong.*

---

## The goal

Make the **backward pass** of grouped-query attention (GQA) run faster than cuDNN (what PyTorch's
`scaled_dot_product_attention` calls under the hood) on an H200 — across a 12-shape sweep of batch
sizes and head counts, at S=4096, D=128, bf16, causal. cuDNN is the number to beat; it's the same
hardware for both of us, so if it can hit a speed, so can I. That belief turned out to be the most
important thing I carried all week.

## Where I started — and the wall I hit

I spent the first several days trying to copy **cuDNN's structure**: three warpgroups per block, a
"combo" kernel where two warpgroups share one load of Q/dO to cut memory traffic in half, and a
persistent scheduler that keeps blocks resident. Each of these *sounded* right on paper — and each one
lost.

The combo kernel actually achieved what it promised: it cut the Q/dO memory traffic by 35%. And it was
**slower** — 5.9 ms vs the 3.2 ms I already had. That was the week's first hard lesson:

> **A memory-traffic win only pays off if you're memory-bound. I wasn't.** The kernel was
> *latency-bound* — the GPU was sitting idle 78% of the time waiting for data to arrive, not waiting for
> bandwidth. Saving bytes I wasn't short on bought nothing, and the bigger per-block workload made the
> idle *worse*.

I kept measuring the wrong scoreboard, too — TFLOP/s and bytes moved — when the number that actually
predicts speed here is **"what fraction of the time are the SMs busy?"** Once I started watching *that*,
the picture got honest.

## The pivot that worked — throw it all out, start naive

The turning point was deciding to **stop imitating cuDNN and start from nothing.** I wrote the dumbest
possible correct kernel — plain scalar math, 305 ms, embarrassingly slow — and then let the *profiler*,
and nothing else, tell me what to fix next. One change at a time. Measure. Keep it only if the SM-busy
number went up.

Eight profile-driven steps later I had **Vz2**: 305 ms → 2.8 ms. Every step was the profiler pointing at
the current bottleneck:
- switch the math to the tensor cores (wgmma)
- overlap the slow "write the gradient" step with the next block's compute instead of waiting for it
- double-buffer the shared memory so two things happen at once
- issue two matrix multiplies together so the tensor pipe never goes idle
- let the softmax math run *during* a matrix multiply instead of after it

The theme of every winning step was the same: **add overlap** — find something the GPU was waiting on,
and give it useful work to do during the wait. (Interestingly, *removing* synchronization barriers
usually did **not** help — if a barrier was already hidden behind overlap, deleting it changed nothing
and sometimes hurt.)

Vz2 was simpler than any of the fancy kernels I'd built earlier, and it **beat cuDNN on the big shapes**
(the high-batch ones I'd never once won). But it lost the small and mid-sized shapes.

## The insight that explained everything

Why did Vz2 win big batches but lose small ones? Because there are **two different ways to hide
latency**, and each is the other's weakness:

1. **Do more inside each block** (cuDNN's 3-warpgroup style). Works when there aren't many blocks to go
   around — it keeps a small number of blocks busy. Wins small workloads.
2. **Run more blocks at once** (Vz2's style — two blocks per SM interleaving). Needs *lots* of blocks to
   fill the machine. Wins big workloads.

Vz2 was pure style #2. On small and mid shapes there simply **weren't enough blocks** to fill all 132
SMs, so the machine sat half-empty. That wasn't a bug to fix inside the kernel — it was a **shortage of
work to hand out.**

## The breakthrough — make more blocks

The fix, once I saw it that way, was almost embarrassingly simple. GQA has several **query heads**
sharing each **key/value head** (say 3-to-1). Vz2 assigned one block per *key/value* head and looped
over the 3 query heads inside. So I cloned Vz2 into **Vj1** and changed exactly one thing: assign one
block per *query* head instead. That's **3× as many blocks**, and each block now does **one head** of
work instead of three (a shorter chain, so it finishes and frees up faster). Each block writes a partial
result, and a tiny follow-up kernel adds the partials together.

That one change re-filled the GPU **at every batch size** — not just the mid ones I was aiming at. Vj1
turned out to be the **fastest of all my kernels on 11 of the 12 shapes**, and it beat cuDNN on 11 of
12 — including **8×12, the one shape I had never beaten all week.** It collapsed my previous
"use-kernel-A-here, kernel-B-there" patchwork into essentially **one kernel that wins almost everywhere.**

## The last shape — deleting the tax

One shape, 4×24, still lost by about 1%. This time I didn't guess — I profiled. And the profiler pointed
somewhere I wasn't looking: not at the main kernel (which was fine), but at the **little helper kernel**
I'd added to make the per-head trick work.

Remember, per-head means several blocks each compute *part* of the same key/value gradient, and something
has to add those parts together. I'd done it the obvious way: each block writes its part to a scratch
buffer, then a second kernel reads them all back and sums them. The profiler showed that little sum
kernel was taking **69 microseconds and was memory-bound** — and the gap to cuDNN was only ~16
microseconds. **The tax I'd added to make per-head work was four times bigger than the gap.** I'd been
about to go optimize the main kernel, which wasn't even the problem.

The fix was to stop summing in a separate kernel and instead have each block **add its part directly into
the final answer as it finishes** — using a hardware feature (TMA-reduce) that does the addition inside
the memory system, overlapped with the real work, no scratch buffer and no second kernel at all. (This is,
it turns out, exactly how cuDNN does it.) It saved ~111 microseconds — *more* than the sum kernel cost,
because I also stopped writing the scratch buffer in the first place. 4×24 went from a 1% loss to a **3%
win.**

**That was the twelfth shape. Every single shape now beats cuDNN.** The thing that finally closed it
wasn't a cleverer kernel — it was *measuring* where the time actually went and deleting a step I never
needed to add.

## The things I'll carry forward

1. **Measure the bottleneck, don't guess it.** SM-busy% and "are there eligible warps to run" tell the
   truth; TFLOP/s and byte-counts lie about what's actually limiting you.
2. **Optimize the thing you're actually short on.** I was short on *latency hiding*, not bandwidth. The
   traffic-saving kernel was a beautiful solution to a problem I didn't have.
3. **Prefer changes that add overlap** over changes that remove synchronization. Overlap fills idle
   time; removing an already-hidden barrier fills nothing.
4. **Sometimes the fix is more work-items, not a smarter kernel.** Half my week was spent making each
   block cleverer when the real problem was that there weren't enough blocks. Splitting the work finer
   (per query head) beat every clever intra-block trick.
5. **Trust that the target is reachable.** cuDNN ran on the same H200. Every time I caught myself about
   to write "this is a hardware limit," it turned out to be a limit of the *structure I'd chosen*, not
   the chip. Never call a wall from a design you haven't finished testing.
6. **Believe the medians, not the runs.** Margins under ~3% are noise — cuDNN's own speed wobbles 2–4%
   run to run (it thermal-throttles under sustained load; my kernels don't). I only claim a win that
   survives 12 matched runs, and that discipline flipped several "wins" to losses and several "losses"
   to wins.
7. **Start from naive when you're stuck.** The single best decision of the week was throwing away days
   of sophisticated work and rebuilding from the dumbest correct version, letting the profiler lead.
8. **Profile the whole pipeline, not just the star kernel.** The final win was hiding in a tiny helper
   kernel I'd stopped thinking about. The main kernel was the obvious suspect and the wrong one. Time
   the *entire* launch sequence — the setup, the cleanup, the little glue steps — because the bottleneck
   loves to hide in the part you've mentally filed as "done."
9. **The cheapest step is the one you delete.** I closed the last gap not by making something faster but
   by removing a step entirely — the scratch-buffer-then-sum became a reduce-as-you-go. When a step is
   the bottleneck, ask whether it needs to exist before you ask how to speed it up.
