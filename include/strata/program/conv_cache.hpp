// include/strata/program/conv_cache.hpp - the conversation cache's retention policy: which checkpoint
// leaves when the cache is over its slot budget.
//
// The cache holds conversation checkpoints: snapshots of the running state (the GDN recurrence, the PLE
// history, the QSA indexer tails - `ConvCheckpoint` in generate.cpp) that a request resumes its prompt
// from instead of reading those tokens again.  The request path keeps only the checkpoints whose tokens
// are a prefix of the current prompt - any other checkpoint's positional cells have been overwritten,
// because the KV cache is one arena that holds one branch of history at a time - so at any moment the
// retained checkpoints are a CHAIN: sorted by length, each a prefix of the next.  That is a radix cache's
// tree collapsed onto the one branch of history the session can hold.
//
// The chain's root is therefore the deepest point every request so far has shared - in practice the end
// of the system prompt, which every new chat of the same client mounts through - and it is exactly the
// node a radix cache keeps alive while its leaves rotate.  First-in-first-out kept it only by accident of
// being first: after `prompt_cache` newer checkpoints it was gone, and the next chat read the whole
// prefix again at prefill speed.  So:
//
//   * the root is PINNED: dropping it frees one slot's ~118 MB and costs every future conversation that
//     shares the prefix a full re-read of it (a 30K system prompt is ~30 s at ~1,000 tok/s);
//   * every other slot rotates by least recent use.  A checkpoint's stamp advances when it is created,
//     when a new checkpoint lands on its length, and when a request mounts through it.
//
// With a budget below two slots there is no room to keep root and leaf apart, so the pin switches off and
// the oldest checkpoint leaves - the previous behaviour, one slot = the newest point only.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::program::conv_cache {

/// The index in `stamps` of the chain item to drop once the chain holds more than `cap` items.  `stamps`
/// are the items' last-use stamps; the caller owns the chain and erases the returned index.  Pure and
/// deterministic so conv_cache_test.cpp can walk the scenarios by hand.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap) {
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the pin is off, the oldest leaves
    size_t v = 1;                     // the root (0) is pinned; least recent use among the rest
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// --prompt-cache-tail: the extra checkpoint near the prompt's end (`tail[i]`) is the first to go - it serves
/// only a branch of the last request - unless it is the newest item (the one just saved: `n > 1` and the newest
/// stamp).  With no such item this is the plain policy.  Never the root.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail) {
    if (tail != nullptr && n >= 2 && cap >= 2) {
        size_t newest = 0;
        for (size_t i = 1; i < n; ++i)
            if (stamps[i] > stamps[newest]) newest = i;
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && i != newest) return i;
    }
    return eviction_victim(stamps, n, cap);
}

/// The pin=N shared prefixes (`pinned[i]`): they are never the victim, whatever their stamp - the queries that branch
/// from one keep coming back to it, and a least-recently-used rule would drop it while they still do.  The root and
/// the tail rule are as above.  When every other item is pinned too, the policy falls back to the oldest pinned one:
/// the cache still keeps to its `cap` (the caller limits how many pins it takes, so this is a backstop).
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail, const bool* pinned) {
    if (pinned == nullptr) return eviction_victim(stamps, n, cap, tail);
    if (cap < 2 || n < 2) return 0;
    size_t newest = 0;
    for (size_t i = 1; i < n; ++i)
        if (stamps[i] > stamps[newest]) newest = i;
    if (tail != nullptr)
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && !pinned[i] && i != newest) return i;
    size_t v = n;                    // the least recently used among the unpinned non-root items
    for (size_t i = 1; i < n; ++i)
        if (!pinned[i] && (v == n || stamps[i] < stamps[v])) v = i;
    if (v != n) return v;
    v = 1;                           // all pinned: the oldest of them (never the root, index 0)
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// The periodic grid: the next multiple of `every` strictly past `at`. The periodic checkpoints of
/// a conversation sit at ABSOLUTE positions (every, 2*every, ...), not at `resume + k*every` per
/// request: a chat that grows in turns far shorter than `every` never reads `every` tokens past
/// any resume, so the resume-relative rule laid no mid-conversation rung at all - the deepest
/// periodic stayed where the first long read left it (17,128 in the 2026-11-08 log) while the
/// conversation grew to 82,191, a prompt that re-rendered history at 41,918 fell back to it, and
/// the engine re-read 75,105 tokens. On the grid every read that crosses a multiple - the first
/// long prompt, a big paste, a post-divergence re-read - lays a rung at its first chunk boundary
/// past it (a read that fits in one chunk lands that boundary on the prompt's end, whose state the
/// cut path saves anyway), so the ladder keeps refreshing as the conversation grows.
/// Undefined for `every` <= 0; the callers gate on it.
inline int64_t next_periodic_at(int64_t at, int64_t every) {
    return (at / every + 1) * every;
}

/// --prompt-cache-before-turn N (opt-in): where a request's extra checkpoint goes - N tokens before its turn
/// boundary `turn_at`, inside what the request reads (past `read_from`). The turn checkpoint is what the next turn
/// resumes from, but a client that re-renders the END of its last message on a retry (zoocode refreshes the
/// environment_details timestamp after a stop or a restart) diverges before it, and the resume fell back to the
/// previous turn's boundary: the 2026-11-09 log diverged at 47,327 with the turn at 47,437 and read 2,517 tokens
/// again from 45,010. A point N before the boundary still matches when the rewrite is less than N deep. The engine
/// keeps one alive, outside the --prompt-cache budget and never the victim, so the rest of the chain rotates as
/// without it. -1 = none: N <= 0, no turn boundary, or the point is not past `read_from`.
inline int64_t before_turn_checkpoint_at(int64_t read_from, int64_t turn_at, int64_t distance) {
    if (distance <= 0 || turn_at <= 0 || read_from < 0) return -1;
    const int64_t at = turn_at - distance;
    return at > read_from ? at : -1;
}

/// The periodic ladder's slot budget: the spill's resume set (spill_resume_checkpoints,
/// conversation_file.hpp) needs the chain to hold the root, the two deepest `turn` boundaries
/// and the end-of-prompt checkpoint - four slots - so what is left of `cap` protects the
/// mid-conversation periodic rungs, at least one from `cap` 5 and none below (the smallest
/// chains keep today's plain rotation).
inline int64_t periodic_budget(int64_t cap) { return cap >= 5 ? cap - 4 : 0; }

/// The periodic ladder: `periodic[i]` marks the chain's periodic checkpoints (the caller passes
/// the --prompt-cache-tail item as periodic=false - the tail rule still drops it first). Up to
/// `budget` of them act as pinned: a stale turn or end-of-prompt leaf leaves before a rung does,
/// so the mid-conversation fallback survives the rotation (a rung's stamp goes stale the turn
/// after it is created - no request mounts through it again while the tail cluster rotates, so
/// plain least-recent-use ate the ladder the 2026-11-08 log's chain lacked). Over the budget the
/// ladder thins from the top: the rung directly below the newest periodic leaves - never the
/// root at index 0, never the newest rung itself (the spill's resume set takes it) - so the kept
/// rungs spread back from the end towards the root and a re-render d tokens deep re-reads about
/// d plus the rung gap, instead of falling through to the root. No periodic flags or a budget of
/// 0: the plain policy, unchanged.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail,
                              const bool* pinned, const bool* periodic, int64_t budget) {
    if (periodic == nullptr || budget <= 0) return eviction_victim(stamps, n, cap, tail, pinned);
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the pin is off, the oldest leaves
    auto pin = [&](size_t i) { return (pinned != nullptr && pinned[i]) || periodic[i]; };
    size_t newest = 0;                // the newest stamp (the item just saved is never the victim)
    for (size_t i = 1; i < n; ++i)
        if (stamps[i] > stamps[newest]) newest = i;
    if (tail != nullptr)              // --prompt-cache-tail: the branch-only item leaves first
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && !pin(i) && i != newest) return i;
    size_t rungs = 0, top = n;        // the periodic rungs, and the ladder's newest position
    for (size_t i = 0; i < n; ++i)
        if (periodic[i]) { ++rungs; top = i; }
    if ((int64_t) rungs > budget)
        for (size_t i = top; i-- > 1;)   // over budget: the rung directly below the newest leaves
            if (periodic[i]) return i;
    size_t v = n;                     // least recent use among the unprotected rest
    for (size_t i = 1; i < n; ++i)
        if (!pin(i) && (v == n || stamps[i] < stamps[v])) v = i;
    if (v != n) return v;
    v = 1;                            // everything protected: the oldest of them, never the root
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// #2409: what a conversation restore can allocate on a card. It writes into session buffers that already exist; the one
/// thing it may create is the calling thread's private copy stream for that card (conversation_copy.hpp), which it takes
/// from the pool of finished threads when there is one, and the first launch of the two small residency kernels loads
/// their code. A stream and an event cost well under 2 MiB where it was measured (conversation_copy_streams_test prints
/// it); 64 MiB covers both many times over and is far below the reserves (--vram-reserve-mib 700), so the check can only
/// fire on a card that is really full.
constexpr uint64_t kRestoreFirstUseMib = 64;

/// Whether a parked snapshot can be restored without a first-use allocation failing: every card for which the restore
/// would have to create its copy stream (`stream_ready[i]` false) must have `need_bytes` free. A card that already has a
/// stream needs nothing, so it never refuses - unlike a floor on `--vram-reserve-mib`, which a loaded card is always under
/// (#2409's review). A missing sample (`free_bytes` null) does not refuse.
inline bool restore_fits(const uint64_t* free_bytes, const uint8_t* stream_ready, size_t n, uint64_t need_bytes) {
    if (free_bytes == nullptr || stream_ready == nullptr || need_bytes == 0) return true;
    for (size_t i = 0; i < n; ++i)
        if (!stream_ready[i] && free_bytes[i] < need_bytes) return false;
    return true;
}

/// --prompt-cache-spread: the item whose removal leaves the smallest gap for its distance from the newest, so the kept
/// points thin out geometrically towards the root and an edit d tokens before the end reads about d again.  `lens`:
/// the chain's token counts.  Never the root, the newest or a pinned item; the tail rule first, as above.
inline size_t spread_victim(const int64_t* lens, const uint64_t* stamps, size_t n, int64_t cap, const bool* tail,
                            const bool* pinned) {
    if (n >= 3 && cap >= 2) {
        auto candidate = [&](size_t i) { return !(pinned && pinned[i]); };
        if (tail != nullptr)
            for (size_t i = 1; i + 1 < n; ++i)
                if (tail[i] && candidate(i)) return i;
        size_t v = n;
        double best = 0;
        for (size_t i = 1; i + 1 < n; ++i) {
            if (!candidate(i)) continue;
            const double gap = double(lens[i + 1] - lens[i - 1]) / double(lens[n - 1] - lens[i - 1]);
            if (v == n || gap < best) { v = i; best = gap; }
        }
        if (v != n) return v;
    }
    return eviction_victim(stamps, n, cap, tail, pinned);
}

/// --conversation-spill-gain N: the live point a disk backup would add, measured on the chain the
/// session holds.  The deepest `turn` checkpoint (the boundary a re-rendering client's next prompt
/// still contains); without one, the deepest checkpoint that is not `end_of_prompt` (that one carries
/// the effort tail the next prompt never re-sends); without any, the live length.  `lens[i]` is
/// checkpoint i's token count, `is_turn[i]` / `is_eop[i]` its kind.
inline int64_t spill_gain_live_point(const int64_t* lens, const bool* is_turn, const bool* is_eop, size_t n,
                                     int64_t live_len) {
    int64_t turn = 0, other = 0;
    for (size_t i = 0; i < n; ++i) {
        if (is_turn[i] && lens[i] > turn) turn = lens[i];
        if (!is_eop[i] && lens[i] > other) other = lens[i];
    }
    return turn > 0 ? turn : other > 0 ? other : live_len;
}

/// --conversation-spill-gain N: skip the disk backup of a conversation the session keeps while the disk
/// already delivers a resume point (`t_disk` > 0) less than N tokens behind the live point `t_live`
/// (spill_gain_live_point).  What a crash then costs is at most that gap, read again.  N <= 0 (the
/// default) never skips: every backup writes, as before.  A disk point at or past the live one skips too
/// (nothing would be added).
inline bool spill_gain_skips(int64_t t_live, int64_t t_disk, int64_t gain) {
    return gain > 0 && t_disk > 0 && t_live - t_disk < gain;
}

}  // namespace strata::program::conv_cache
