// CPU-only tests for the disk-tier spill store (include/strata/core/conversation_disk_store.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model. Mirrors the style of
// src/core/conversation_file_test.cpp: a single main(), anonymous-namespace sample builders,
// a check() helper that exits on failure, and a scratch dir under fs::temp_directory_path().
#include "strata/core/conversation_disk_store.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

#if !defined(_WIN32)
#include <unistd.h>
#define jsetenv(n, v) ::setenv(n, v, 1)
#define junsetenv(n) ::unsetenv(n)
#else
#define jsetenv(n, v) _putenv_s(n, v)
#define junsetenv(n) _putenv_s(n, "")
#endif

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// Build a small SavedConversation with the same shape as conversation_file_test.cpp's sample(),
// but smaller (we want fast tests, not 17 MiB files). The disk store's logic is independent of
// the payload size; we only need round-trip identity and prefix-match determinism.
ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn = bytes_of(99, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(31, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

SavedConversation sample(size_t live_tokens = 1000) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(live_tokens, 1);
    s.checkpoints.push_back(checkpoint(10, 2));
    s.checkpoints.push_back(checkpoint(500, 3));
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = layer == 2 ? 0 : 9; kv.idx_dim = 128;
        kv.k = pattern(4096 + layer, uint8_t(layer));
        kv.v = pattern(4096 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(layer == 1 ? 0 : 512, uint8_t(layer + 20));
        kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(layer == 2 ? 0 : 9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used && b.stage_parts.empty();
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}

// sample() shifted to a different token range: a DIFFERENT conversation of the same client
// (distinct history, same system root shape). Used by the reload tests that must keep several
// conversations side by side — under the same-conversation dedupe, plain sample(100/200/300)
// all describe ONE growing conversation and only the newest would survive.

// ── Delta-journal builders: capture-consistent images (cells = tokens rounded up to a page,
// every part exactly per-cell bytes), so a journal record can chain onto them. Mirrors the
// layout math of conversation_snapshot.cpp's layout() for a small fp16 K/V.
constexpr int64_t JPage = 64, JHeads = 2, JHeadDim = 64, JIdxDim = 32, JIdxBlock = 4;

int64_t jcells(size_t tokens) { return ((int64_t) tokens + JPage - 1) / JPage * JPage; }

ConversationBuffer jfill(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed * 7u);
        return true;
    });
    return b;
}

ConversationCheckpoint jckpt(const std::vector<int32_t>& ids, uint8_t seed) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.gdn = {uint8_t(seed), uint8_t(seed + 1)};
    c.ple = {uint8_t(seed + 2)};
    c.tails = {uint8_t(seed + 3), uint8_t(seed + 4), uint8_t(seed + 5)};
    c.dead = {uint8_t(seed + 6)};
    c.block_pos = {uint8_t(seed + 7), uint8_t(seed + 8)};
    c.used = seed;
    return c;
}

SavedConversation conv_image(size_t tokens, uint8_t seed) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(200 + i);
    s.layer_lo = 0; s.layer_hi = 40;
    s.cvec = false;
    std::vector<int32_t> ids;
    for (size_t i = 0; i < tokens; ++i) ids.push_back(int32_t(3000 + i));
    s.live = jckpt(ids, seed);
    s.live.imgs = {{7, 0xabcd0000ull + seed}};
    s.checkpoints.push_back(jckpt(
        std::vector<int32_t>(ids.begin(), ids.begin() + std::ptrdiff_t(tokens / 2)), uint8_t(seed + 50)));
    const int64_t cells = jcells(tokens);
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 1; kv.cells = cells; kv.heads = JHeads; kv.head_dim = JHeadDim;
        kv.page_size = JPage;
        kv.pooled_rows = layer < 2 && tokens > 0 ? (int64_t) tokens / JIdxBlock + 1 : 0;
        kv.idx_dim = JIdxDim;
        kv.k = jfill(size_t(cells) * size_t(JHeads * JHeadDim * 2), uint8_t(layer * 10 + seed));
        kv.v = jfill(size_t(cells) * size_t(JHeads * JHeadDim * 2), uint8_t(layer * 10 + seed + 1));
        kv.k_scale = layer < 2 ? jfill(size_t(cells) * size_t(JHeads * 2), uint8_t(layer * 10 + seed + 2))
                               : ConversationBuffer{};
        kv.v_scale = layer < 2 ? jfill(size_t(cells) * size_t(JHeads * 2), uint8_t(layer * 10 + seed + 3))
                               : ConversationBuffer{};
        kv.pooled = layer < 2 && tokens > 0
                        ? jfill(size_t(kv.pooled_rows) * size_t(JIdxDim * 4), uint8_t(layer * 10 + seed + 4))
                        : ConversationBuffer{};
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// conv_image with EXPLICIT ids (cases 54/55's re-render models need their own tokens):
// capture-consistent, the same shape conv_image builds.
SavedConversation ids_conv_image(const std::vector<int32_t>& ids, uint8_t seed) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(200 + i);
    s.layer_lo = 0; s.layer_hi = 40;
    s.cvec = false;
    s.live = jckpt(ids, seed);
    const size_t tokens = ids.size();
    const int64_t cells = jcells(tokens);
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 1; kv.cells = cells; kv.heads = JHeads; kv.head_dim = JHeadDim;
        kv.page_size = JPage;
        kv.pooled_rows = layer < 2 && tokens > 0 ? (int64_t) tokens / JIdxBlock + 1 : 0;
        kv.idx_dim = JIdxDim;
        kv.k = jfill(size_t(cells) * size_t(JHeads * JHeadDim * 2), uint8_t(layer * 10 + seed));
        kv.v = jfill(size_t(cells) * size_t(JHeads * JHeadDim * 2), uint8_t(layer * 10 + seed + 1));
        kv.k_scale = layer < 2 ? jfill(size_t(cells) * size_t(JHeads * 2), uint8_t(layer * 10 + seed + 2))
                               : ConversationBuffer{};
        kv.v_scale = layer < 2 ? jfill(size_t(cells) * size_t(JHeads * 2), uint8_t(layer * 10 + seed + 3))
                               : ConversationBuffer{};
        kv.pooled = layer < 2 && tokens > 0
                        ? jfill(size_t(kv.pooled_rows) * size_t(JIdxDim * 4), uint8_t(layer * 10 + seed + 4))
                        : ConversationBuffer{};
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// The next turn of `base`: `extra` more tokens; fresh bytes from the completed-page boundary
// of base's length on (exactly what a capture re-sends), everything below it identical.
SavedConversation conv_grow(const SavedConversation& base, size_t extra, uint8_t seed) {
    SavedConversation s = base;
    const size_t old_tokens = s.live.ids.size();
    for (size_t i = 0; i < extra; ++i) s.live.ids.push_back(int32_t(3000 + old_tokens + i));
    const std::vector<ConversationImageKey> base_imgs = s.live.imgs;   // the pictures stay
    s.live = jckpt(s.live.ids, seed);
    s.live.imgs = base_imgs;
    s.checkpoints.clear();
    s.checkpoints.push_back(jckpt(
        std::vector<int32_t>(s.live.ids.begin(),
                             s.live.ids.begin() + std::ptrdiff_t(s.live.ids.size() / 2)),
        uint8_t(seed + 50)));
    const int64_t whole = (int64_t) (old_tokens / JPage) * JPage;
    const int64_t cells = jcells(s.live.ids.size());
    const int64_t pooled_keep = (int64_t) old_tokens / JIdxBlock;
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv& kv = s.kv[layer];
        kv.cells = cells;
        kv.pooled_rows = layer < 2 ? (int64_t) s.live.ids.size() / JIdxBlock + 1 : 0;
        auto extend = [&](ConversationBuffer& buf, size_t size, size_t at) {
            buf.resize(size);   // keeps [0, at): the base's completed pages
            if (size > at)
                buf.visit(at, size - at, [&](uint8_t* p, size_t c, size_t) {
                    for (size_t i = 0; i < c; ++i) p[i] = uint8_t(i * 17u + seed + layer);
                    return true;
                });
        };
        extend(kv.k, size_t(cells) * size_t(JHeads * JHeadDim * 2), size_t(whole) * size_t(JHeads * JHeadDim * 2));
        extend(kv.v, size_t(cells) * size_t(JHeads * JHeadDim * 2), size_t(whole) * size_t(JHeads * JHeadDim * 2));
        if (layer < 2) {
            extend(kv.k_scale, size_t(cells) * size_t(JHeads * 2), size_t(whole) * size_t(JHeads * 2));
            extend(kv.v_scale, size_t(cells) * size_t(JHeads * 2), size_t(whole) * size_t(JHeads * 2));
            extend(kv.pooled, size_t(kv.pooled_rows) * size_t(JIdxDim * 4), size_t(pooled_keep) * size_t(JIdxDim * 4));
        }
    }
    return s;
}

// A 2-GPU spill's shape: the main holds layers [0, 40), one stage holds [40, 48) — the same
// live state (one capture, two devices), different K/V bytes per device. What
// spill_live_to_disk builds for a layer split on two cards.
SavedConversation conv_stage_of(size_t tokens, uint8_t seed) {
    SavedConversation s = conv_image(tokens, seed);
    s.layer_lo = 0; s.layer_hi = 40;
    SavedConversation st = conv_image(tokens, uint8_t(seed + 9));
    st.layer_lo = 40; st.layer_hi = 48;
    st.live = s.live;                     // the capture's one live state
    st.checkpoints = s.checkpoints;
    s.stage_images.push_back(std::move(st));
    return s;
}

// The next turn of a 2-GPU spill: the main and the stage each grow `extra` tokens from the
// same completed-page boundary (each device's own buffers), sharing the new live state.
SavedConversation conv_stage_grow(const SavedConversation& base, size_t extra, uint8_t seed) {
    SavedConversation s = conv_grow(base, extra, seed);   // grows the main's ids and K/V
    s.stage_images.clear();
    SavedConversation st = conv_grow(base.stage_images[0], extra, uint8_t(seed + 1));
    st.layer_lo = base.stage_images[0].layer_lo;
    st.layer_hi = base.stage_images[0].layer_hi;
    st.live = s.live;
    st.checkpoints = s.checkpoints;
    s.stage_images.push_back(std::move(st));
    return s;
}

// The client's re-render: the new history keeps only the anchor checkpoint's prefix (the
// base's live ids are NOT a prefix of the result) with a fresh shorter turn on top. The K/V
// below the anchor's completed-page boundary is the base's own (the same conversation's
// cells); above it, fresh bytes.
SavedConversation conv_rerender(const SavedConversation& base, size_t new_tail, uint8_t seed) {
    const std::vector<int32_t> anchor = base.checkpoints.back().ids;
    const size_t keep = anchor.size();
    const std::vector<ConversationImageKey> base_imgs = base.live.imgs;
    SavedConversation s = base;
    s.live.ids = anchor;
    for (size_t i = 0; i < new_tail; ++i) s.live.ids.push_back(int32_t(9000 + i));
    s.live = jckpt(s.live.ids, seed);
    s.live.imgs = base_imgs;
    s.checkpoints.clear();
    s.checkpoints.push_back(jckpt(
        std::vector<int32_t>(s.live.ids.begin(),
                             s.live.ids.begin() + std::ptrdiff_t(s.live.ids.size() / 2)),
        uint8_t(seed + 50)));
    const int64_t whole = (int64_t) (keep / JPage) * JPage;
    const int64_t cells = jcells(s.live.ids.size());
    const int64_t pooled_keep = (int64_t) keep / JIdxBlock;
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv& kv = s.kv[layer];
        kv.cells = cells;
        kv.pooled_rows = layer < 2 ? (int64_t) s.live.ids.size() / JIdxBlock + 1 : 0;
        auto cut = [&](ConversationBuffer& buf, size_t size, size_t at) {
            buf.resize(at);      // keep the base's completed pages of the anchor prefix
            buf.resize(size);    // grow to the new layout
            if (size > at)
                buf.visit(at, size - at, [&](uint8_t* p, size_t c, size_t) {
                    for (size_t i = 0; i < c; ++i) p[i] = uint8_t(i * 23u + seed + layer);
                    return true;
                });
        };
        cut(kv.k, size_t(cells) * size_t(JHeads * JHeadDim * 2), size_t(whole) * size_t(JHeads * JHeadDim * 2));
        cut(kv.v, size_t(cells) * size_t(JHeads * JHeadDim * 2), size_t(whole) * size_t(JHeads * JHeadDim * 2));
        if (layer < 2) {
            cut(kv.k_scale, size_t(cells) * size_t(JHeads * 2), size_t(whole) * size_t(JHeads * 2));
            cut(kv.v_scale, size_t(cells) * size_t(JHeads * 2), size_t(whole) * size_t(JHeads * 2));
            cut(kv.pooled, size_t(kv.pooled_rows) * size_t(JIdxDim * 4), size_t(pooled_keep) * size_t(JIdxDim * 4));
        }
    }
    return s;
}

// A capture whose deepest checkpoint sits at `ckpt_tokens` (the real park shape: the checkpoint
// is the last turn boundary, a few hundred ids below the live end — the reporting run's park
// 5814 held 22728 under 23143).
SavedConversation conv_at(size_t tokens, size_t ckpt_tokens, uint8_t seed) {
    SavedConversation s = conv_image(tokens, seed);
    s.checkpoints.clear();
    s.checkpoints.push_back(jckpt(
        std::vector<int32_t>(s.live.ids.begin(), s.live.ids.begin() + std::ptrdiff_t(ckpt_tokens)),
        uint8_t(seed + 50)));
    return s;
}

// The reporting run's next park: the history keeps [0, cut) of the base and is re-rendered
// above it (the client strips the assistant's thinking), so the base's ids are NOT a prefix of
// the result and its checkpoint is NOT contained — only the common prefix survives, landing
// BELOW the old checkpoint (measured: 29 ids below it, every consecutive pair of the run). The
// K/V below the completed-page boundary of `cut` is the base's own cells (the same
// conversation's bytes, what a capture re-sends); above it, fresh bytes. `ckpt_at` places the
// new capture's deepest checkpoint (0: at half) — a real park's checkpoint is its last turn
// boundary, near the end, which a chain's next park cuts just below.
SavedConversation conv_lcp_turn(const SavedConversation& base, size_t cut, size_t new_tail,
                                uint8_t seed, size_t ckpt_at = 0) {
    const std::vector<ConversationImageKey> base_imgs = base.live.imgs;
    SavedConversation s = base;
    s.live.ids.assign(base.live.ids.begin(), base.live.ids.begin() + std::ptrdiff_t(cut));
    for (size_t i = 0; i < new_tail; ++i) s.live.ids.push_back(int32_t(9000 + i));
    s.live = jckpt(s.live.ids, seed);
    s.live.imgs = base_imgs;
    const size_t ckpt_pos = ckpt_at > 0 ? std::min(ckpt_at, s.live.ids.size()) : s.live.ids.size() / 2;
    s.checkpoints.clear();
    s.checkpoints.push_back(jckpt(
        std::vector<int32_t>(s.live.ids.begin(), s.live.ids.begin() + std::ptrdiff_t(ckpt_pos)),
        uint8_t(seed + 50)));
    const int64_t whole = (int64_t) (cut / JPage) * JPage;
    const int64_t cells = jcells(s.live.ids.size());
    const int64_t pooled_keep = (int64_t) cut / JIdxBlock;
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv& kv = s.kv[layer];
        kv.cells = cells;
        kv.pooled_rows = layer < 2 ? (int64_t) s.live.ids.size() / JIdxBlock + 1 : 0;
        auto cut_buf = [&](ConversationBuffer& buf, size_t size, size_t at) {
            buf.resize(at);      // keep the base's completed pages of the kept prefix
            buf.resize(size);    // grow to the new layout
            if (size > at)
                buf.visit(at, size - at, [&](uint8_t* p, size_t c, size_t) {
                    for (size_t i = 0; i < c; ++i) p[i] = uint8_t(i * 23u + seed + layer);
                    return true;
                });
        };
        cut_buf(kv.k, size_t(cells) * size_t(JHeads * JHeadDim * 2),
                size_t(whole) * size_t(JHeads * JHeadDim * 2));
        cut_buf(kv.v, size_t(cells) * size_t(JHeads * JHeadDim * 2),
                size_t(whole) * size_t(JHeads * JHeadDim * 2));
        if (layer < 2) {
            cut_buf(kv.k_scale, size_t(cells) * size_t(JHeads * 2), size_t(whole) * size_t(JHeads * 2));
            cut_buf(kv.v_scale, size_t(cells) * size_t(JHeads * 2), size_t(whole) * size_t(JHeads * 2));
            cut_buf(kv.pooled, size_t(kv.pooled_rows) * size_t(JIdxDim * 4),
                    size_t(pooled_keep) * size_t(JIdxDim * 4));
        }
    }
    return s;
}

// The 2-GPU next park with the same divergence: the main and the stage each keep the base's
// completed pages below `cut`, sharing the re-rendered live state.
SavedConversation conv_stage_lcp(const SavedConversation& base, size_t cut, size_t new_tail,
                                 uint8_t seed) {
    SavedConversation s = conv_lcp_turn(base, cut, new_tail, seed);
    s.stage_images.clear();
    SavedConversation st = conv_lcp_turn(base.stage_images[0], cut, new_tail, uint8_t(seed + 1));
    st.layer_lo = base.stage_images[0].layer_lo;
    st.layer_hi = base.stage_images[0].layer_hi;
    st.live = s.live;
    st.checkpoints = s.checkpoints;
    s.stage_images.push_back(std::move(st));
    return s;
}

// A pinned shared-prefix snapshot (pfx-<serial>.bin): only its ids — and their LENGTH, the
// lineage guard's pin_len — matter to the store; a one-layer body keeps the file small. Its
// live ids ARE its only checkpoint: the prefix shape is_prefix_entry() recognizes.
SavedConversation pin_image(const std::vector<int32_t>& ids) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(200 + i);
    s.layer_lo = 0; s.layer_hi = 40;
    s.cvec = false;
    s.live = jckpt(ids, 21);
    s.checkpoints.push_back(jckpt(ids, 21));
    const int64_t cells = jcells(ids.size());
    ConversationKv kv;
    kv.format = 1; kv.cells = cells; kv.heads = JHeads; kv.head_dim = 8;
    kv.page_size = JPage; kv.pooled_rows = 0; kv.idx_dim = JIdxDim;
    kv.k = jfill(size_t(cells) * size_t(JHeads * 8 * 2), 31);
    kv.v = jfill(size_t(cells) * size_t(JHeads * 8 * 2), 32);
    s.kv.push_back(std::move(kv));
    return s;
}

bool same_conv(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.cvec != b.cvec || a.live.ids != b.live.ids ||
        a.live.imgs != b.live.imgs || a.live.gdn != b.live.gdn || a.live.ple != b.live.ple ||
        a.live.tails != b.live.tails || a.live.dead != b.live.dead ||
        a.live.block_pos != b.live.block_pos || a.checkpoints.size() != b.checkpoints.size() ||
        a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (a.checkpoints[i].ids != b.checkpoints[i].ids ||
            a.checkpoints[i].gdn != b.checkpoints[i].gdn || a.checkpoints[i].used != b.checkpoints[i].used)
            return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled))
            return false;
    }
    return true;
}

SavedConversation shifted(size_t n, int32_t base) {
    SavedConversation s = sample(n);
    for (auto& t : s.live.ids) t += base;
    for (auto& k : s.checkpoints)
        for (auto& t : k.ids) t += base;
    return s;
}

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};

    // ── 1. Round trip: put → best → take returns the same image; the entry (and file) STAY ──
    // take() no longer consumes the entry: the (base + journal) pair on disk is what the next
    // park of this conversation appends its delta to, and what holds() sees to skip the
    // re-capture of an unchanged conversation. The file is only deleted by drop_superseded
    // (replacement) or evict_oldest (budget).
    {
        ConversationDiskStore store(dir.string() + "/rt", 1ull << 30, &id);
        check(store.enabled(), "store enabled with non-zero budget");
        SavedConversation original = sample(1000);
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        const uint64_t serial = 42;
        check(store.put(std::move(original), serial, id), "put succeeds");
        check(store.size() == 1, "manifest holds one entry");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 1000; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);
        const auto m = store.best(prompt, imgs, /*cvec=*/false);
        check(m.tokens == 1000, "best() matches the live prefix (1000 tokens)");
        check(m.serial == serial, "best() returns the correct serial");
        auto back = store.take(m, id);
        check(back.has_value(), "take returns the image");
        check(same(sample(1000), *back), "round trip is identical");
        check(store.size() == 1, "take KEEPS the entry in the manifest (the delta base)");
        check(store.bytes() > 0, "take keeps the entry's bytes accounted (its file stays)");
        // El archivo se QUEDA en disco (no se borra en take exitoso).
        check(fs::exists(dir / "rt" / ("conv-" + std::to_string(serial) + ".bin")), "file KEPT after take (preserved)");
        check(store.restores() == 1, "restore counter incremented");
    }

    // ── 2. best() matches prefix without reading the payload (file truncated mid-payload) ────
    // The manifest keeps ids in RAM; a corrupt file must not break best(). take() on a corrupt
    // file returns nullopt (the hash check fails inside session_file_read) and the entry is
    // dropped from the manifest.
    {
        ConversationDiskStore store(dir.string() + "/corrupt", 1ull << 30, &id);
        SavedConversation original = sample(500);
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        check(store.put(std::move(original), 7, id), "put for corrupt test");
        // Truncate the file to half its size.
        const std::string path = dir.string() + "/corrupt/conv-7.bin";
        std::error_code ec;
        const uintmax_t sz = fs::file_size(path, ec);
        check(!ec && sz > 100, "spill file exists with content");
        fs::resize_file(path, sz / 2, ec);
        check(!ec, "truncate succeeded");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 500; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more: the next turn's token (conversation_prefix needs n < prompt.size())
        const auto m = store.best(prompt, imgs, false);
        check(m.tokens == 500, "best() still matches the prefix (manifest in RAM, payload unread)");
        auto back = store.take(m, id);
        check(!back.has_value(), "take rejects corrupt file");
        check(store.size() == 0, "take removed the corrupt entry");
    }

    // ── 3. take() with a wrong SessionFileIdentity returns nullopt and drops the entry ──────
    {
        ConversationDiskStore store(dir.string() + "/id", 1ull << 30, &id);
        SavedConversation original = sample(200);
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        check(store.put(std::move(original), 11, id), "put with correct id");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 200; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more: conversation_prefix needs n < prompt.size()
        const auto m = store.best(prompt, imgs, false);
        check(m.tokens == 200, "best matches");
        const SessionFileIdentity wrong{id.model ^ 1, id.config};
        auto back = store.take(m, wrong);
        check(!back.has_value(), "take with wrong id rejected");
        check(store.size() == 0, "wrong-id entry removed from manifest");
    }

    // ── 4. take() with RAM admit refusal returns nullopt but keeps the file ──────────────────
    // The caller's admit callback says "no RAM now". The file is kept so a later turn can retry.
    {
        ConversationDiskStore store(dir.string() + "/admit", 1ull << 30, &id);
        SavedConversation original = sample(300);
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        check(store.put(std::move(original), 13, id), "put for admit test");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 300; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more: conversation_prefix needs n < prompt.size()
        const auto m = store.best(prompt, imgs, false);
        SessionReadLimits limits;
        limits.admit = [](uint64_t, std::string&) { return false; };   // refuse any allocation
        auto back = store.take(m, id, limits);
        check(!back.has_value(), "take refused by RAM admit");
        check(store.size() == 0, "entry removed from manifest (no retry tracking in Phase 1)");
        // The file may or may not be present: Phase 1's take() removes the file on every
        // failure path. Phase 2 will keep it on memory refusal for retry. We test Phase 1
        // behavior here: file is gone (the manifest no longer references it).
    }

    // ── 5. LRU eviction: fill the budget, put one more, the oldest is dropped ──────────────
    {
        // Budget = 1 MiB; each sample() is ~50 KiB (small K/V). ~20 fit before LRU kicks in.
        ConversationDiskStore store(dir.string() + "/lru", 1ull << 20, &id);
        std::vector<uint64_t> serials;
        for (int i = 0; i < 30; ++i) {
            SavedConversation s = sample(50);
            const uint64_t ser = 100 + i;
            // Force a unique prefix per entry so best() can distinguish them.
            for (size_t k = 0; k < s.live.ids.size(); ++k) s.live.ids[k] = int32_t(2000 + i * 1000 + k);
            if (!store.put(std::move(s), ser, id)) break;
            serials.push_back(ser);
        }
        check(store.size() < 30, "LRU evicted some entries to fit the budget");
        check(store.drops() > 0, "drop counter incremented");
        // The oldest serial (100) should have been evicted; the newest should still be there.
        // sample()'s checkpoint uses imgs = {{3, 0x1234567890abcdef ^ 1}} (the seed is always 1).
        const std::vector<ConversationImageKey> lru_imgs = {{3, 0x1234567890abcdefull ^ 1ull}};
        std::vector<int32_t> prompt_old;
        for (int i = 0; i < 50; ++i) prompt_old.push_back(int32_t(2000 + i));   // matches serial 100
        prompt_old.push_back(9999);   // one more
        const auto m_old = store.best(prompt_old, lru_imgs, false);
        check(m_old.tokens == 0, "oldest entry evicted (no prefix match)");
        std::vector<int32_t> prompt_new;
        for (int i = 0; i < 50; ++i) prompt_new.push_back(int32_t(2000 + 29 * 1000 + i));   // matches serial 129
        prompt_new.push_back(9999);   // one more
        const auto m_new = store.best(prompt_new, lru_imgs, false);
        check(m_new.tokens == 50, "newest entry still present");
    }

    // ── 6. clear() deletes every spill file and empties the manifest (#879 poison case) ─────
    {
        ConversationDiskStore store(dir.string() + "/clear", 1ull << 30, &id);
        for (int i = 0; i < 3; ++i) {
            SavedConversation s = sample(50);
            check(store.put(std::move(s), 200 + i, id), "put for clear test");
        }
        check(store.size() == 3, "3 entries before clear");
        const size_t dropped = store.clear();
        check(dropped == 3, "clear returned the count");
        check(store.size() == 0, "manifest empty after clear");
        check(store.bytes() == 0, "bytes zero after clear");
        // All conv-*.bin files gone.
        int leftover = 0;
        for (const auto& e : fs::directory_iterator(dir / "clear")) {
            if (e.path().filename().string().rfind("conv-", 0) == 0) ++leftover;
        }
        check(leftover == 0, "no spill files left in dir");
    }

    // ── 7. Disk full: put() returns false, store keeps its existing state, no crash ─────────
    // Simulated via SessionWriteOptions.min_free_bytes set absurdly high (more than the test
    // machine's free space).
    {
        ConversationDiskStore store(dir.string() + "/full", 1ull << 30, &id);
        SavedConversation original = sample(100);
        SessionWriteOptions opts;
        opts.min_free_bytes = 1ull << 40;   // 1 TiB headroom: any real disk fails this
        const bool ok = store.put(std::move(original), 99, id, opts);
        check(!ok, "put refuses on simulated full disk");
        check(store.size() == 0, "no entry left from the failed put");
        check(store.writes() == 0, "write counter not incremented on failure");
    }

    // ── 8. purge_on_startup() removes conv-*.bin and leftover .tmp files ────────────────────
    {
        const fs::path pdir = dir / "purge";
        fs::create_directories(pdir);
        // Drop a fake spill file and a fake .tmp leftover from an interrupted write.
        std::ofstream(pdir / "conv-1.bin") << "garbage";
        std::ofstream(pdir / "conv-2.bin") << "garbage";
        std::ofstream(pdir / "abc.tmp") << "interrupted write";
        std::ofstream(pdir / "unrelated.txt") << "kept";
        ConversationDiskStore store(pdir.string(), 1ull << 30, &id);
        store.purge_on_startup();
        check(!fs::exists(pdir / "conv-1.bin"), "purge removed conv-1.bin");
        check(!fs::exists(pdir / "conv-2.bin"), "purge removed conv-2.bin");
        check(!fs::exists(pdir / "abc.tmp"), "purge removed leftover .tmp");
        check(fs::exists(pdir / "unrelated.txt"), "purge kept unrelated files");
    }

    // ── 9. Disabled store (budget = 0): put returns false, best returns no match ───────────
    {
        ConversationDiskStore store(dir.string() + "/disabled", 0, &id);
        check(!store.enabled(), "budget 0 disables store");
        SavedConversation original = sample(100);
        check(!store.put(std::move(original), 1, id), "put on disabled store fails");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more
        // best() on a disabled store: the prompt's image list is irrelevant (entries_ is empty).
        const auto m = store.best(prompt, {}, false);
        check(m.tokens == 0, "best on disabled store returns no match");
    }

    // ── 10. drop_superseded: a fresh park with the same live prefix drops the old spill ─────
    {
        ConversationDiskStore store(dir.string() + "/super", 1ull << 30, &id);
        SavedConversation original = sample(400);
        check(store.put(std::move(original), 50, id), "put for supersede test");
        check(store.size() == 1, "one entry");
        // A new conversation whose live ids + imgs match the spilled one supersedes it.
        SavedConversation next = sample(400);   // same live prefix (1000..1399)
        std::vector<ConversationCheckpoint> chain;
        chain.push_back(next.live);
        const size_t dropped = store.drop_superseded(next.live.ids, next.live.imgs, chain, next.cvec);
        check(dropped == 1, "drop_superseded removed the matching spill");
        check(store.size() == 0, "manifest empty after drop_superseded");
    }

    // ── 11. reload_from_disk: spills written by a previous engine run are picked up ──────
    // Simulates the restart scenario: engine A writes a spill, exits; engine B creates a new
    // ConversationDiskStore on the same dir, calls reload_from_disk(), and the spilled
    // conversation is in the manifest (best() finds it, take() restores it).
    {
        const auto rdir = dir / "reload";
        // Engine A writes one spill.
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.enabled(), "store A enabled");
            SavedConversation original = sample(100);
            check(store_a.put(std::move(original), 100, id), "store A put");
            check(store_a.size() == 1, "store A has one entry");
            check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin on disk");
        }
        // Engine B starts (a new ConversationDiskStore on the same dir). Its manifest starts empty;
        // reload_from_disk rebuilds it from the .bin files written by A. The identity must match
        // (engine B is the same model + config as A — that's what makes the spills readable).
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id);
            check(store_b.enabled(), "store B enabled");
            check(store_b.size() == 0, "store B manifest empty before reload");
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded == 1, "reloaded 1 entry from disk");
            check(store_b.size() == 1, "store B has 1 entry after reload");
            check(store_b.bytes() > 0, "store B has bytes after reload");
            // The manifest's prefix data is what best() matches against — the K/V is gone from RAM.
            // sample() builds the live checkpoint with imgs = {{3, 0x1234567890abcdefull ^ 1}}.
            // The prompt's image list must match for best() to accept the prefix.
            std::vector<int32_t> prompt;
            for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(1000 + i));
            prompt.push_back(9999);   // one more token (the "new message")
            const std::vector<ConversationImageKey> prompt_imgs = {{3, 0x1234567890abcdefull ^ 1}};
            const auto m = store_b.best(prompt, prompt_imgs, false);
            check(m.tokens == 100, "best() found the reloaded entry's 100-token prefix");
            check(m.serial == 100, "best() returned the right serial");
            // take() reads the file and restores the full image (and deletes the file).
            SavedConversation back;
            size_t bytes = 0;
            std::string error;
            SessionStatus status;
            bool ok = session_file_read(rdir.string() + "/conv-100.bin", id, back, bytes, error, {}, &status);
            check(ok, "reloaded file is readable");
            check(back.live.ids.size() == 100, "reloaded image has 100 live ids");
        }
        check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin kept on disk after reload (only take() removes it)");
    }

    // ── 11b. reload_from_disk with multiple spills: all are loaded (no unconditional deletion) ─
    // The fix: when the dir has multiple spills (conversations from multiple sessions),
    // reload_from_disk() validates and loads ALL of them. The disk budget is enforced by
    // evict_oldest() (the real LRU); older spills are NOT unconditionally deleted on startup.
    // Three DIFFERENT conversations with an ample budget: all 3 survive (the same-conversation
    // dedupe keeps exactly one set per conversation — block 23 covers that path).
    {
        const auto rdir = dir / "reload_multi";
        // Engine A writes 3 spills with growing serials.
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(shifted(100, 0), 100, id), "put serial 100");
            check(store_a.put(shifted(200, 4000), 101, id), "put serial 101 (another conversation)");
            check(store_a.put(shifted(300, 8000), 102, id), "put serial 102 (a third conversation)");
            check(store_a.size() == 3, "store A has 3 entries");
            check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin on disk");
            check(fs::exists(rdir / "conv-101.bin"), "conv-101.bin on disk");
            check(fs::exists(rdir / "conv-102.bin"), "conv-102.bin on disk");
        }
        // Engine B reloads: with an ample budget, ALL 3 are kept and loaded.
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded == 3, "reloaded 3 entries (all valid spills loaded)");
            check(store_b.size() == 3, "store B has 3 entries after reload");
            check(store_b.bytes() > 0, "store B has bytes after reload");
            check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin kept on disk");
            check(fs::exists(rdir / "conv-101.bin"), "conv-101.bin kept on disk");
            check(fs::exists(rdir / "conv-102.bin"), "conv-102.bin kept on disk");
            // Iteration order is ASCENDING (oldest first), so entry_serial(0) is the OLDEST
            // serial (100), entry_serial(2) is the NEWEST (102). This is what makes evict_oldest()
            // pop the right one under budget pressure (see 11c).
            check(store_b.entry_serial(0) == 100, "manifest[0] is the oldest serial (100)");
            check(store_b.entry_serial(1) == 101, "manifest[1] is serial 101");
            check(store_b.entry_serial(2) == 102, "manifest[2] is the newest serial (102)");
        }
    }

    // ── 11c. reload_from_disk under budget pressure: oldest evicted, newest kept ──────────
    // The disk budget is enforced by evict_oldest() (the real LRU). With 3 spills and a budget
    // that fits only 2, the OLDEST (lowest serial) is evicted — both from the manifest and from
    // disk. This is the regression test for the ascending-iteration order: if anyone inverts
    // it (newest first), the NEWEST conversation would be at the front of entries_ and
    // evict_oldest() would drop the wrong one.
    {
        const auto rdir = dir / "reload_budget";
        // Engine A writes 3 spills with a huge budget, just to populate the dir.
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(shifted(100, 0), 100, id), "put serial 100");
            check(store_a.put(shifted(200, 4000), 101, id), "put serial 101 (another conversation)");
            check(store_a.put(shifted(300, 8000), 102, id), "put serial 102 (a third conversation)");
            check(store_a.size() == 3, "store A has 3 entries");
            check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin on disk");
            check(fs::exists(rdir / "conv-101.bin"), "conv-101.bin on disk");
            check(fs::exists(rdir / "conv-102.bin"), "conv-102.bin on disk");
            // Each spill file is ~40427 bytes (sample() with 3 KV layers of 4096+ bytes each).
            // 2 spills = ~80854, 3 spills = ~121281. A budget of 100000 fits 2 but not 3: when
            // the 3rd is loaded, evict_oldest() pops the 1st (serial 100). `loaded` counts the
            // files that were read and pushed into the manifest (3); `size()` counts the
            // survivors (2).
            {
                const uint64_t budget_2 = 100000;   // between 80854 (2 spills) and 121281 (3 spills)
                ConversationDiskStore store_b(rdir.string(), budget_2, &id);
                check(store_b.budget() == budget_2, "store B budget is the small one");
                const size_t loaded = store_b.reload_from_disk(id);
                check(loaded == 3, "reloaded 3 entries (all 3 files were read and pushed)");
                check(store_b.size() == 2, "store B has 2 entries after LRU eviction (oldest evicted)");
                check(store_b.entry_serial(0) == 101, "manifest[0] is serial 101 (oldest kept)");
                check(store_b.entry_serial(1) == 102, "manifest[1] is serial 102 (newest kept)");
                check(!fs::exists(rdir / "conv-100.bin"), "oldest conv-100.bin evicted from disk");
                check(fs::exists(rdir / "conv-101.bin"), "conv-101.bin kept on disk");
                check(fs::exists(rdir / "conv-102.bin"), "conv-102.bin kept on disk");
                check(store_b.drops() >= 1, "drops() counted the LRU eviction");
            }
        }
    }

    // ── 12. reload_from_disk deletes corrupt files, keeps valid ones ─────────────────────
    // A corrupt conv-*.bin is deleted (it can never match this engine); a valid one with a
    // lower serial is loaded into the manifest and KEPT on disk. Under the new contract the
    // valid file is not "superseded" by the corrupt one — both are independent entries.
    {
        const auto rdir = dir / "reload_corrupt";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        // Write a valid spill with a low serial.
        SavedConversation original = sample(50);
        check(store.put(std::move(original), 200, id), "put valid spill");
        // Drop a corrupt conv-*.bin file with a HIGHER serial.
        std::ofstream(rdir / "conv-300.bin") << "garbage not a real session file";
        check(store.size() == 1, "store has 1 valid entry");
        const size_t loaded = store.reload_from_disk(id);
        check(loaded == 1, "reloaded 1 entry (the valid conv-200.bin)");
        check(store.size() == 1, "manifest has 1 entry after reload");
        check(fs::exists(rdir / "conv-300.bin"), "corrupt conv-300.bin KEPT on disk (never deleted)");
        check(fs::exists(rdir / "conv-200.bin"), "valid conv-200.bin kept on disk");
    }

    // ── 13. Sidecar .idx: put() writes it ─────────────────────────────────────────────────
    // After a successful put(), conv-<serial>.idx exists next to the .bin and is small.
    {
        const auto rdir = dir / "sidecar_put";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        check(store.put(std::move(sample(100)), 100, id), "put for sidecar test");
        check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin on disk");
        check(fs::exists(rdir / "conv-100.idx"), "conv-100.idx sidecar on disk");
        const uintmax_t bin_size = fs::file_size(rdir / "conv-100.bin");
        const uintmax_t idx_size = fs::file_size(rdir / "conv-100.idx");
        check(idx_size > 0, "idx file is non-empty");
        check(idx_size < bin_size, "idx is smaller than the bin");
        check(idx_size < 4096, "idx is small (few hundred bytes, not KiB)");
    }

    // ── 14. Sidecar .idx: fast path reload without parsing the .bin ────────────────────────
    // Engine A writes a spill, Engine B reloads using the .idx (fast path). The manifest data
    // (size, bytes, serial, ids, imgs) must match what the slow path would produce.
    {
        const auto rdir = dir / "sidecar_fast";
        // Engine A: write + capture the manifest via slow path (so we have a reference).
        std::vector<int32_t> ref_ids;
        std::vector<ConversationImageKey> ref_imgs;
        uint64_t ref_serial = 0;
        size_t ref_size = 0;
        uint64_t ref_bytes = 0;
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(std::move(sample(100)), 100, id), "store A put");
            // The .idx was written by put(). Best() finds the entry and its ids match.
            std::vector<int32_t> prompt;
            for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(1000 + i));
            prompt.push_back(9999);
            const std::vector<ConversationImageKey> prompt_imgs = {{3, 0x1234567890abcdefull ^ 1}};
            const auto m = store_a.best(prompt, prompt_imgs, false);
            check(m.tokens == 100, "store A best() finds 100-token prefix");
            ref_serial = m.serial;
            ref_size = store_a.size();
            ref_bytes = store_a.bytes();
            const std::vector<int32_t>* e_ids = store_a.entry_ids(0);
            const std::vector<ConversationImageKey>* e_imgs = store_a.entry_imgs(0);
            check(e_ids && e_imgs, "store A entry accessors work");
            ref_ids = *e_ids;
            ref_imgs = *e_imgs;
        }
        // Engine B: reload. The .idx is present and valid → fast path, no .bin parse.
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded == 1, "fast-path reloaded 1 entry");
            check(store_b.size() == ref_size, "size matches slow path");
            check(store_b.bytes() == ref_bytes, "bytes match slow path");
            check(store_b.entry_serial(0) == ref_serial, "serial matches slow path");
            const std::vector<int32_t>* e_ids = store_b.entry_ids(0);
            const std::vector<ConversationImageKey>* e_imgs = store_b.entry_imgs(0);
            check(e_ids && *e_ids == ref_ids, "ids match slow path");
            check(e_imgs && *e_imgs == ref_imgs, "imgs match slow path");
            // best() finds the prefix too (proves the manifest data is usable, not just present).
            std::vector<int32_t> prompt;
            for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(1000 + i));
            prompt.push_back(9999);
            const std::vector<ConversationImageKey> prompt_imgs = {{3, 0x1234567890abcdefull ^ 1}};
            const auto m = store_b.best(prompt, prompt_imgs, false);
            check(m.tokens == 100, "fast-path best() finds the 100-token prefix");
            check(fs::exists(rdir / "conv-100.bin"), ".bin kept on disk");
            check(fs::exists(rdir / "conv-100.idx"), ".idx kept on disk");
        }
    }

    // ── 15. Sidecar .idx: fallback when .idx is missing (rewrite on reload) ──────────────
    // Delete the .idx manually, reload → slow path loads the entry AND rewrites the .idx.
    {
        const auto rdir = dir / "sidecar_missing_idx";
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(std::move(sample(80)), 100, id), "put for missing-idx test");
            check(fs::exists(rdir / "conv-100.idx"), ".idx written by put");
        }
        // Delete the .idx — reload must fall back to the slow path.
        fs::remove(rdir / "conv-100.idx");
        check(!fs::exists(rdir / "conv-100.idx"), ".idx deleted before reload");
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded == 1, "slow-path reloaded 1 entry (.idx was missing)");
            check(store_b.size() == 1, "manifest has 1 entry");
            // The slow path MUST rewrite the .idx so the next start is fast.
            check(fs::exists(rdir / "conv-100.idx"), ".idx rewritten by slow path");
        }
    }

    // ── 16. Sidecar .idx: fallback when .idx is corrupt (truncated) ───────────────────────
    // Truncate the .idx to half its size, reload → slow path, .idx rewritten.
    {
        const auto rdir = dir / "sidecar_corrupt_idx";
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(std::move(sample(70)), 100, id), "put for corrupt-idx test");
            check(fs::exists(rdir / "conv-100.idx"), ".idx written");
        }
        // Truncate the .idx to 10 bytes (way too small for the header).
        {
            std::ofstream f(rdir / "conv-100.idx", std::ios::trunc);
            f << "1234567890";
        }
        check(fs::file_size(rdir / "conv-100.idx") == 10, ".idx truncated to 10 bytes");
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded == 1, "slow-path reloaded 1 entry (.idx was corrupt)");
            check(store_b.size() == 1, "manifest has 1 entry");
            check(fs::file_size(rdir / "conv-100.idx") > 10, ".idx rewritten with valid content");
        }
    }

    // ── 17. Sidecar .idx: fallback when identity differs ──────────────────────────────────
    // A .idx with a different config must NOT be used — the slow path validates the .bin
    // against THIS engine's identity and rejects it (the .bin was written by another engine).
    {
        const auto rdir = dir / "sidecar_wrong_identity";
        SessionFileIdentity id_a = id;
        SessionFileIdentity id_b = id;
        id_b.config = 0xDEADBEEFCAFEBABEull;   // different config → different engine instance
        // Engine A writes a spill + .idx.
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id_a);
            check(store_a.put(std::move(sample(60)), 100, id_a), "put with identity A");
            check(fs::exists(rdir / "conv-100.idx"), ".idx written with identity A");
        }
        // Engine B (different identity) reloads. The .idx has id_a.config ≠ id_b.config, so
        // read_idx returns false. The slow path parses the .bin, finds identity mismatch
        // (the .bin's header says id_a), and deletes both .bin and .idx (corruption path).
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id_b);
            const size_t loaded = store_b.reload_from_disk(id_b);
            check(loaded == 0, "wrong-identity .idx → slow path → identity mismatch → 0 loaded");
            check(store_b.size() == 0, "manifest empty (the .bin was from another engine)");
            check(fs::exists(rdir / "conv-100.bin"), "wrong-identity .bin KEPT on disk (never deleted)");
            check(fs::exists(rdir / "conv-100.idx"), "wrong-identity .idx KEPT on disk (never deleted)");
        }
    }

    // ── 18. Sidecar .idx: fallback when the .bin changed (stale .idx) ────────────────────
    // Rewrite the .bin with different content (or touch its mtime), leaving the .idx stale.
    // reload must detect the size/mtime mismatch and fall back to the slow path.
    {
        const auto rdir = dir / "sidecar_stale_idx";
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(std::move(sample(90)), 100, id), "put for stale-idx test");
            check(fs::exists(rdir / "conv-100.idx"), ".idx written");
        }
        // Overwrite the .bin with garbage (different size). The .idx still says the old size.
        {
            std::ofstream f(rdir / "conv-100.bin", std::ios::trunc);
            for (int i = 0; i < 1000; ++i) f << "X";
        }
        // Reload: the .idx's bin_size no longer matches the .bin's actual size → fallback.
        // The slow path parses the .bin, finds it is not a valid session file (garbage), and
        // deletes it (corruption path).
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded == 0, "stale .idx → slow path → garbage .bin → 0 loaded");
            check(store_b.size() == 0, "manifest empty");
            check(fs::exists(rdir / "conv-100.bin"), "garbage .bin KEPT on disk (never deleted)");
            check(fs::exists(rdir / "conv-100.idx"), "stale .idx KEPT on disk (never deleted)");
        }
    }

    // ── 19. Sidecar .idx: coherent deletion (evict, take, clear, drop_superseded) ────────
    // Every site that deletes a .bin must also delete its .idx. This test exercises evict,
    // take, and clear; drop_superseded is covered by the integration test.
    {
        const auto rdir = dir / "sidecar_deletion";
        // evict_oldest: write 3 spills with a budget that fits 1, reload → 2 evicted.
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(std::move(sample(100)), 100, id), "put 100");
            check(store_a.put(std::move(sample(100)), 101, id), "put 101");
            check(store_a.put(std::move(sample(100)), 102, id), "put 102");
            check(fs::exists(rdir / "conv-100.idx"), ".idx for 100");
            check(fs::exists(rdir / "conv-101.idx"), ".idx for 101");
            check(fs::exists(rdir / "conv-102.idx"), ".idx for 102");
        }
        // Reload with a budget that fits only 1 → evict 2 oldest (100, 101). Their .idx must go too.
        {
            const uint64_t budget_1 = 50000;   // ~1 spill is 40427, 2 is 80854
            ConversationDiskStore store_b(rdir.string(), budget_1, &id);
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded >= 1, "at least 1 loaded");
            check(store_b.size() == 1, "only 1 entry fits the budget");
            check(!fs::exists(rdir / "conv-100.bin"), "evicted 100 .bin gone");
            check(!fs::exists(rdir / "conv-100.idx"), "evicted 100 .idx gone");
            check(!fs::exists(rdir / "conv-101.bin"), "evicted 101 .bin gone");
            check(!fs::exists(rdir / "conv-101.idx"), "evicted 101 .idx gone");
            check(fs::exists(rdir / "conv-102.bin"), "kept 102 .bin");
            check(fs::exists(rdir / "conv-102.idx"), "kept 102 .idx");
            // take() the survivor → its .bin and .idx must go.
            std::vector<int32_t> prompt;
            for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(1000 + i));
            prompt.push_back(9999);
            const std::vector<ConversationImageKey> prompt_imgs = {{3, 0x1234567890abcdefull ^ 1}};
            const auto m = store_b.best(prompt, prompt_imgs, false);
            check(m.tokens == 100, "best finds the survivor");
            auto back = store_b.take(m, id);
            check(back.has_value(), "take succeeded");
            check(fs::exists(rdir / "conv-102.bin"), "taken 102 .bin KEPT (preserved on disk)");
            check(fs::exists(rdir / "conv-102.idx"), "taken 102 .idx KEPT (preserved on disk)");
        }
        // clear(): write 2 more, clear, both .bin and .idx gone.
        {
            ConversationDiskStore store_c(rdir.string(), 1ull << 30, &id);
            check(store_c.put(std::move(sample(50)), 200, id), "put 200");
            check(store_c.put(std::move(sample(50)), 201, id), "put 201");
            check(fs::exists(rdir / "conv-200.idx"), ".idx for 200");
            check(fs::exists(rdir / "conv-201.idx"), ".idx for 201");
            store_c.clear();
            check(!fs::exists(rdir / "conv-200.bin"), "clear 200 .bin gone");
            check(!fs::exists(rdir / "conv-200.idx"), "clear 200 .idx gone");
            check(!fs::exists(rdir / "conv-201.bin"), "clear 201 .bin gone");
            check(!fs::exists(rdir / "conv-201.idx"), "clear 201 .idx gone");
        }
    }

    // ── 20. Sidecar .idx: orphan .idx (no .bin) is ignored ───────────────────────────────
    // A .idx without its .bin must not create a phantom entry. read_idx checks the .bin exists.
    {
        const auto rdir = dir / "sidecar_orphan_idx";
        // Drop an orphan .idx (no .bin). reload must ignore it.
        fs::create_directories(rdir);
        {
            std::ofstream f(rdir / "conv-999.idx", std::ios::binary);
            // Write a minimal valid-looking .idx so we test the .bin-exists check, not the parser.
            // (Even garbage would be rejected by the checksum, but we want to test the .bin path.)
            f << "SIDX" << char(1) << char(0) << char(0) << char(0);   // magic + version
            for (int i = 0; i < 64; ++i) f << char(0);   // 8 × 8 bytes of model/config/geometry/...
        }
        check(fs::exists(rdir / "conv-999.idx"), "orphan .idx present");
        check(!fs::exists(rdir / "conv-999.bin"), "no .bin for the orphan .idx");
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        const size_t loaded = store.reload_from_disk(id);
        check(loaded == 0, "orphan .idx → 0 loaded");
        check(store.size() == 0, "manifest empty");
    }

    // ── 21. purge_on_startup cleans .bin, .idx, .idx.tmp; keeps other files ────────────────
    {
        const auto rdir = dir / "sidecar_purge";
        fs::create_directories(rdir);
        // Drop a mix of files: a real .bin + .idx pair, an orphan .idx.tmp, an orphan .bin.tmp,
        // and an unrelated keep.txt that must survive.
        std::ofstream(rdir / "conv-1.bin") << "bin";
        std::ofstream(rdir / "conv-1.idx") << "idx";
        std::ofstream(rdir / "conv-2.idx.tmp") << "idx tmp";
        std::ofstream(rdir / "conv-3.bin.tmp") << "bin tmp";
        std::ofstream(rdir / "keep.txt") << "keep me";
        check(fs::exists(rdir / "conv-1.bin"), "conv-1.bin before purge");
        check(fs::exists(rdir / "conv-1.idx"), "conv-1.idx before purge");
        check(fs::exists(rdir / "conv-2.idx.tmp"), "conv-2.idx.tmp before purge");
        check(fs::exists(rdir / "conv-3.bin.tmp"), "conv-3.bin.tmp before purge");
        check(fs::exists(rdir / "keep.txt"), "keep.txt before purge");
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            store.purge_on_startup();
        }
        check(!fs::exists(rdir / "conv-1.bin"), "conv-1.bin purged");
        check(!fs::exists(rdir / "conv-1.idx"), "conv-1.idx purged");
        check(!fs::exists(rdir / "conv-2.idx.tmp"), "conv-2.idx.tmp purged");
        check(!fs::exists(rdir / "conv-3.bin.tmp"), "conv-3.bin.tmp purged");
        check(fs::exists(rdir / "keep.txt"), "keep.txt kept");
    }

    // ── 22. Same conversation, re-rendered (the client strips the assistant thinking): the old
    // spill's ids are NOT a prefix of the new ones (they end inside the reasoning of a turn the
    // new one re-rendered empty), so the exact/prefix rules never match and one spill per park
    // piled up (the "18 files for 1 conversation" report). The old entry's DEEPEST CHECKPOINT —
    // the last turn boundary of the client's own render — IS contained in the new conversation:
    // the writer's cleanup must drop the old spill (files included). A DIFFERENT conversation
    // sharing only the system root is kept.
    {
        ConversationDiskStore store(dir.string() + "/rerender", 1ull << 30, &id);
        // A: the old turn's spill — live 1000..1999 (its "thinking" included), boundary 1000..1499.
        const auto turn_spill = [&](int tail_base, size_t shared) {
            SavedConversation s = sample(1000);
            s.live.imgs.clear();
            s.checkpoints.clear();
            for (size_t i = 0; i < shared; ++i) s.live.ids.push_back(int32_t(1000 + i));  // the shared render
            for (int i = 0; i < 300; ++i) s.live.ids.push_back(int32_t(tail_base + i));  // the re-rendered turn
            ConversationCheckpoint b;   // the deepest checkpoint: the turn boundary the client keeps
            b.ids.assign(s.live.ids.begin(), s.live.ids.begin() + (std::ptrdiff_t) shared);
            s.checkpoints.push_back(std::move(b));
            return s;
        };
        SavedConversation a = turn_spill(1500, 500);   // live 1000..1799: the render + A's own "thinking"
        check(store.put(std::move(a), 100, id), "22: put A (the old turn's spill)");
        // C: a DIFFERENT conversation (ids 5000..) — must never be dropped by B's cleanup.
        SavedConversation c = sample(1000);
        for (auto& t : c.live.ids) t += 4000;
        for (auto& k : c.checkpoints) for (auto& t : k.ids) t += 4000;
        check(store.put_async(std::move(c), 102, id), "22: put C (a different conversation)");
        store.flush_pending();
        check(store.size() == 2, "22: A and C both on disk");
        // B: the next turn of A's conversation, RE-RENDERED — the prompt keeps A's client-render
        // region [0, 500) verbatim, then diverges inside what A rendered as thinking. B's live is
        // not a superset of A's and shares no clean prefix with it (800 < 1000 tokens).
        SavedConversation b = turn_spill(7000, 500);
        check(store.put_async(std::move(b), 101, id), "22: put B (the re-rendered next turn)");
        store.flush_pending();
        check(store.size() == 2, "22: B replaced A; C survives (manifest = B + C)");
        bool has_a = false, has_b = false, has_c = false;
        for (size_t i = 0; i < store.size(); ++i) {
            has_a = has_a || store.entry_serial(i) == 100;
            has_b = has_b || store.entry_serial(i) == 101;
            has_c = has_c || store.entry_serial(i) == 102;
        }
        check(!has_a, "22: A dropped as superseded (same conversation, re-rendered)");
        check(has_b, "22: B kept (the newest state)");
        check(has_c, "22: C kept (a different conversation is never dropped)");
        check(!fs::exists(dir / "rerender" / "conv-100.bin"), "22: A's .bin deleted");
        check(!fs::exists(dir / "rerender" / "conv-100.idx"), "22: A's .idx deleted");
        check(fs::exists(dir / "rerender" / "conv-102.bin"), "22: C's .bin still there");
    }

    // ── 23. reload_from_disk dedupes the duplicates earlier runs left behind: the FIRST start
    // after this fix collapses the pile (one serial per park, none superseded through the
    // re-render) back to one set per conversation. The newest serial wins; the loser's files
    // are deleted; a different conversation and a not-contained (rolled-back) older spill stay.
    {
        const auto rdir = dir / "dedupe";
        const auto turn_spill = [&](int tail_base) {
            SavedConversation s = sample(1000);
            s.live.imgs.clear();
            s.checkpoints.clear();
            for (int i = 0; i < 500; ++i) s.live.ids.push_back(int32_t(1000 + i));  // the shared render
            for (int i = 0; i < 300; ++i) s.live.ids.push_back(int32_t(tail_base + i));  // re-rendered
            ConversationCheckpoint b;
            b.ids.assign(s.live.ids.begin(), s.live.ids.begin() + 500);
            s.checkpoints.push_back(std::move(b));
            return s;
        };
        {
            ConversationDiskStore writer(rdir.string(), 4ull << 30, &id);
            // Three successive re-rendered turns of ONE conversation + one unrelated spill,
            // all written with sync put() (no cleanup runs there — that is exactly the state
            // the old builds left behind: 4 sets of files in the dir).
            check(writer.put(turn_spill(1500), 200, id), "23: put turn 1 (serial 200, live 1000..1799)");
            check(writer.put(turn_spill(7000), 201, id), "23: put turn 2 re-rendered (serial 201)");
            check(writer.put(turn_spill(8000), 202, id), "23: put turn 3 re-rendered (serial 202)");
            SavedConversation other = sample(1000);
            for (auto& t : other.live.ids) t += 4000;
            for (auto& k : other.checkpoints) for (auto& t : k.ids) t += 4000;
            check(writer.put(std::move(other), 300, id), "23: put an unrelated conversation (serial 300)");
            check(writer.size() == 4, "23: four sets of files on disk before the restart");
        }
        // A fresh engine start over the same dir: the reload loads all four, then the new
        // dedupe keeps only the newest turn of the conversation (202) plus the unrelated one.
        ConversationDiskStore store(rdir.string(), 4ull << 30, &id);
        const size_t loaded = store.reload_from_disk(id);
        check(loaded == 4, "23: all four spills were read and pushed");
        check(store.size() == 2, "23: dedupe collapsed the conversation to its newest spill");
        check(store.entry_serial(0) == 202, "23: the newest turn (202) is the survivor");
        check(store.entry_serial(1) == 300, "23: the unrelated conversation (300) survives");
        check(!fs::exists(rdir / "conv-200.bin"), "23: turn 1's .bin cleaned");
        check(!fs::exists(rdir / "conv-200.idx"), "23: turn 1's .idx cleaned");
        check(!fs::exists(rdir / "conv-201.bin"), "23: turn 2's .bin cleaned");
        check(fs::exists(rdir / "conv-202.bin"), "23: turn 3's .bin kept");
        check(fs::exists(rdir / "conv-300.bin"), "23: the unrelated spill kept");
        // best() still restores the surviving conversation from its turn boundary (the
        // re-render rule works through the .idx fast path: ckpt_ids round-trips in the sidecar).
        std::vector<int32_t> next_prompt;
        for (int i = 0; i < 500; ++i) next_prompt.push_back(int32_t(1000 + i));
        next_prompt.push_back(1);
        const auto m = store.best(next_prompt, {}, /*cvec=*/false);
        check(m.tokens == 500 && m.serial == 202, "23: best() still matches the turn boundary");
    }

    // ── 24. Delta journal ON: a continued turn APPENDS to conv-<serial>.dlt instead of
    // rewriting the .bin; take() folds the records and returns the newest state; the entry
    // stays in the manifest the whole time (take-keep).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal";
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            SavedConversation a0 = conv_image(1000, 1);
            SavedConversation a1 = conv_grow(a0, 24, 2);
            SavedConversation a2 = conv_grow(a1, 24, 3);
            const uint64_t serial = 500;
            check(store.put(std::move(a0), serial, id), "24: the first park writes the base .bin");
            check(fs::exists(rdir / "conv-500.bin"), "24: the base .bin exists");
            check(!fs::exists(rdir / "conv-500.dlt"), "24: no journal yet");
            const uint64_t bin_size = fs::file_size(rdir / "conv-500.bin");
            check(store.put_async(std::move(a1), serial + 1, id), "24: the second park is queued");
            store.flush_pending();
            check(fs::exists(rdir / "conv-500.dlt"), "24: the second park appended a journal record");
            check(fs::file_size(rdir / "conv-500.bin") == bin_size, "24: the base .bin was NOT rewritten");
            check(store.size() == 1, "24: one entry (the base, extended)");
            check(store.put_async(std::move(a2), serial + 2, id), "24: the third park is queued");
            store.flush_pending();
            check(store.size() == 1, "24: still one entry");
            const uint64_t journal_size = fs::file_size(rdir / "conv-500.dlt");
            check(journal_size > 0, "24: the journal holds two records");
            // the delta wrote far less than a full .bin would
            check(journal_size < bin_size / 2, "24: two records cost less than half a .bin");
            // take() folds the journal: the restored image is the THIRD state
            ConversationCache::Match m;
            m.serial = 500; m.tokens = 1048; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value(), "24: take returns the folded image");
            check(same_conv(*back, conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 3)),
                  "24: base + 2 records == the third turn's image");
            check(store.size() == 1, "24: the entry stayed in the manifest after take");
        }
        // restart: the .idx fast path carries the post-apply ids; a fresh take restores it
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store.reload_from_disk(id);
            check(loaded == 1, "24: reload loads the (base + journal) entry");
            ConversationCache::Match m;
            m.serial = 500; m.tokens = 1048; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value() && back->live.ids.size() == 1048,
                  "24: after the restart, the folded state restores (fast path)");
        }
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 25. park → take → park (the per-turn cycle): the park after a take appends to the SAME
    // serial's journal — no new .bin, no second entry (this is the Kaggle churn scenario).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_cycle";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation a0 = conv_image(1000, 1);
        check(store.put(std::move(a0), 600, id), "25: park 0");
        SavedConversation a1 = conv_grow(conv_image(1000, 1), 24, 2);
        check(store.put_async(std::move(a1), 601, id), "25: park 1 (delta)");
        store.flush_pending();
        ConversationCache::Match m;
        m.serial = 600; m.tokens = 1024; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value(), "25: take 1");
        check(store.size() == 1, "25: take kept the entry");
        // the conversation grew again after the restore: the park must APPEND, not rewrite
        SavedConversation a2 = conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 4);
        check(store.put_async(std::move(a2), 602, id), "25: park 2 after the take");
        store.flush_pending();
        check(store.size() == 1, "25: still ONE entry (the same base)");
        check(fs::exists(rdir / "conv-600.dlt"), "25: the journal exists");
        const uint64_t bin_size = fs::file_size(rdir / "conv-600.bin");
        check(bin_size == fs::file_size(rdir / "conv-600.bin"), "25: the .bin never moved");
        // and an UNCHANGED re-park is skipped by holds() exactly as the engine does
        std::vector<int32_t> live_now = conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 4).live.ids;
        check(store.holds(live_now, {{7, 0xabcd0001ull}}, false),
              "25: holds() sees the extended conversation (the retry skip)");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 26. protect_serial regression (the Rev. 1 bug): the cleanup that follows EVERY
    // write runs with the newest ids of the conversation it just absorbed — after an append
    // the base entry's ids EQUAL them (the exact-match branch), after the next append they
    // are a PREFIX (the prefix branch), and a re-rendered turn's cleanup reaches it through
    // its checkpoint. Three sequential appends + their cleanups must all leave the base
    // (and its journal) alive; only an OLDER spill of the same conversation goes.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_protect";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        // an OLD spill of the same conversation (a strict prefix, written before the base)
        check(store.put(conv_image(1000, 1), 700, id), "26: the old turn's spill (serial 700)");
        check(store.put(conv_image(1000, 1), 701, id), "26: the base (serial 701)");
        check(store.size() == 2, "26: two entries before any append");
        // append 1: the cleanup runs with a1's ids (exact-match against the extended base,
        // prefix against the old turn 700 — the old turn goes, the base stays)
        SavedConversation a1 = conv_grow(conv_image(1000, 1), 24, 2);
        check(store.put_async(std::move(a1), 702, id), "26: append 1 queued");
        store.flush_pending();
        check(store.size() == 1, "26: the old turn went, the extended base stayed (protect_serial)");
        check(fs::exists(rdir / "conv-701.bin") && fs::exists(rdir / "conv-701.dlt"),
              "26: the base's .bin and .dlt survived their own append's cleanup");
        check(!fs::exists(rdir / "conv-700.bin"), "26: the OLD turn's .bin was dropped by the cleanup");
        // append 2 (prefix branch: the entry's ids are now a strict prefix of a2's)
        SavedConversation a2 = conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 5);
        check(store.put_async(std::move(a2), 703, id), "26: append 2 queued");
        store.flush_pending();
        check(store.size() == 1 && fs::exists(rdir / "conv-701.dlt"),
              "26: the prefix-branch cleanup kept the base too");
        // append 3 (a re-rendered turn: the entry's ids diverge, its checkpoint is contained)
        SavedConversation a3 = conv_rerender(conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 5), 100, 6);
        check(a3.live.ids.size() == 524 + 100, "26: the re-render is anchored at the checkpoint");
        check(store.put_async(std::move(a3), 704, id), "26: append 3 (re-render) queued");
        store.flush_pending();
        check(store.size() == 1 && fs::exists(rdir / "conv-701.bin"),
              "26: the re-render-branch cleanup kept the base as well");
        // (nothing ever folds the journal — the take below proves the three-record chain
        // restores correctly, record by record)
        // the whole chain restores
        ConversationCache::Match m;
        m.serial = 701; m.tokens = 1000; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && back->live.ids.size() == 624,
              "26: the three-record chain folds back to the re-rendered state");
        check(same_conv(*back, conv_rerender(conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 5), 100, 6)),
              "26: the folded state is the re-rendered image, byte-exactly");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 27. Rewind: a much SHORTER conversation that diverges before the base's checkpoint is
    // a different branch — a full write, and the longer base survives (the rollback case).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_rewind";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation a0 = conv_image(1000, 1);
        check(store.put(std::move(a0), 800, id), "27: the base");
        SavedConversation b = conv_image(600, 9);   // diverges at token 0 (ids 3000.. vs 3000.. — same
        for (auto& t : b.live.ids) t += 7777;       // start, so shift it: a genuinely different history
        for (auto& c : b.checkpoints)
            for (auto& t : c.ids) t += 7777;
        check(store.put(std::move(b), 801, id), "27: a different (rewound) conversation");
        store.flush_pending();
        check(store.size() == 2, "27: both entries stay (a rollback branch is kept)");
        check(fs::exists(rdir / "conv-800.bin") && fs::exists(rdir / "conv-801.bin"),
              "27: two .bin files (no delta was possible)");
        check(!fs::exists(rdir / "conv-800.dlt") && !fs::exists(rdir / "conv-801.dlt"),
              "27: no journals were created");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 28. Reload without the .idx (the slow path): the .bin is parsed, the journal folded
    // into the entry, and the .idx is rewritten — the next start takes the fast path again.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_reload";
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            SavedConversation a0 = conv_image(1000, 1);
            SavedConversation a1 = conv_grow(a0, 24, 2);
            check(store.put(std::move(a0), 900, id), "28: the base");
            check(store.put_async(std::move(a1), 901, id), "28: one delta");
            store.flush_pending();
        }
        fs::remove(rdir / "conv-900.idx");   // force the slow path
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store.reload_from_disk(id);
            check(loaded == 1, "28: the slow path loaded the entry");
            ConversationCache::Match m;
            m.serial = 900; m.tokens = 1024; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value() && back->live.ids.size() == 1024,
                  "28: the slow path folded the journal into the entry");
            check(fs::exists(rdir / "conv-900.idx"), "28: the .idx was rewritten");
        }
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 29. Opt-out (STRATA_SPILL_JOURNAL=0): no journal is ever created; the old
    // rewrite-in-place behaviour. (The default is ON since v2 — the rest of the suite runs
    // with it — so the escape must be the explicit zero, not the unset env.)
    {
        jsetenv("STRATA_SPILL_JOURNAL", "0");
        const auto rdir = dir / "journal_off";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation a0 = conv_image(1000, 1);
        SavedConversation a1 = conv_grow(a0, 40, 2);
        check(store.put(std::move(a0), 950, id), "29: the first park");
        check(store.put_async(std::move(a1), 951, id), "29: the second park");
        store.flush_pending();
        check(!fs::exists(rdir / "conv-950.dlt"), "29: no journal with the flag off");
        check(store.size() == 1, "29: the superseded cleanup replaced the old spill (today's shape)");
        ConversationCache::Match m;
        m.serial = 951; m.tokens = 1040; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && back->live.ids.size() == 1040, "29: the newest state restores");
    }

    // ── 30. purge_on_startup and the reload's orphan sweep both take a stray .dlt.
    {
        const auto rdir = dir / "journal_orphan";
        fs::create_directories(rdir);
        std::ofstream(rdir / "conv-9.bin") << "bin";
        std::ofstream(rdir / "conv-9.dlt") << "dlt";
        std::ofstream(rdir / "conv-10.dlt") << "orphan journal";   // no .bin
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            store.purge_on_startup();
        }
        check(!fs::exists(rdir / "conv-9.bin"), "30: purge removed the .bin");
        check(!fs::exists(rdir / "conv-9.dlt"), "30: purge removed its .dlt");
        check(!fs::exists(rdir / "conv-10.dlt"), "30: purge removed the orphan .dlt");
        // and the reload's own sweep (a .dlt whose main is missing)
        std::ofstream(rdir / "conv-11.bin") << "junk";   // fails validation, stays on disk
        std::ofstream(rdir / "conv-12.dlt") << "orphan";
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            (void) store.reload_from_disk(id);
        }
        check(!fs::exists(rdir / "conv-12.dlt"), "30: the reload swept the orphan .dlt");
    }

    // ── 31. Append-only FOREVER: turns past every old compaction bound (16 records / 64 MiB /
    // 20% of the .bin) keep APPENDING — the .dlt survives with every record, the parent .bin
    // is never rewritten (its size never moves), and take() folds the whole chain back to the
    // newest state. This is the parent/children contract: one .bin written once + one record
    // per continued turn, for the conversation's whole life.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_append_only";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation cur = conv_image(1000, 1);
        check(store.put(std::move(cur), 1000, id), "31: the base");
        cur = conv_image(1000, 1);
        const uint64_t bin_size = fs::file_size(rdir / "conv-1000.bin");
        // grow a page at a time: each record re-sends one partial page (~17 KiB); 24 turns is
        // past every old bound — the record count (16) for sure, and the byte/ratio bounds too
        for (int turn = 1; turn <= 24; ++turn) {
            SavedConversation next = conv_grow(cur, 30, uint8_t(turn));
            SavedConversation queued = next;   // the queue takes a copy; the chain keeps the original
            cur = std::move(next);
            check(store.put_async(std::move(queued), uint64_t(1000 + turn), id), "31: a delta turn");
            store.flush_pending();
            check(fs::file_size(rdir / "conv-1000.bin") == bin_size,
                  "31: the parent .bin never moved (byte-for-byte the first park)");
        }
        check(fs::exists(rdir / "conv-1000.dlt"), "31: the journal survived — nothing folds it");
        check(store.size() == 1, "31: one entry (the parent + its children)");
        ConversationCache::Match m;
        m.serial = 1000; m.tokens = (int64_t) cur.live.ids.size(); m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && same_conv(*back, cur),
              "31: the parent + every record restores the newest state");
        check(fs::file_size(rdir / "conv-1000.bin") == bin_size,
              "31: the .bin is still the first park's bytes after the take");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 32. The budget sees the journal: once the .dlt grows past the budget, the LRU evicts
    // the OLDEST conversation (never the base its own record extends), and the eviction
    // removes .bin + .dlt + .idx together.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_budget";
        const uint64_t bin_sz = [&] {
            ConversationDiskStore probe(rdir.string() + "-probe", 1ull << 30, &id);
            probe.put(conv_image(1000, 1), 1, id);
            return fs::file_size(rdir.string() + "-probe/conv-1.bin");
        }();
        // room for two .bin files plus a small journal, but not for two .bin files plus a
        // growing one
        ConversationDiskStore store(rdir.string(), 2 * bin_sz + bin_sz / 8, &id);
        // an OLDER, different conversation: the eviction's intended victim
        SavedConversation other = conv_image(1000, 1);
        for (auto& t : other.live.ids) t += 5555;
        for (auto& c : other.checkpoints)
            for (auto& t : c.ids) t += 5555;
        check(store.put(std::move(other), 1200, id), "32: the older conversation fits");
        check(store.put(conv_image(1000, 1), 1100, id), "32: the base fits beside it");
        check(store.size() == 2, "32: two entries");
        // grow the base until the pair crosses the budget: the older conversation goes, the
        // base never pays for its own records
        SavedConversation cur = conv_image(1000, 1);
        for (int turn = 1; turn <= 6 && store.size() == 2; ++turn) {
            SavedConversation next = conv_grow(cur, 40, uint8_t(turn));
            SavedConversation queued = next;
            cur = std::move(next);
            check(store.put_async(std::move(queued), uint64_t(1100 + turn), id), "32: a delta turn");
            store.flush_pending();
        }
        check(store.size() == 1, "32: the older conversation was evicted, the base stayed");
        check(!fs::exists(rdir / "conv-1200.bin"), "32: the victim's .bin went");
        check(fs::exists(rdir / "conv-1100.bin") && fs::exists(rdir / "conv-1100.dlt"),
              "32: the base and its journal survived their own growth");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 33. The 2-GPU shape (a layer split): a continued turn appends ONE record that covers
    // the main AND the stage — neither file is rewritten; take() folds both; the restart
    // restores both (this is the Kaggle 2×T4 churn scenario the journal exists for).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_stages";
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            SavedConversation a0 = conv_stage_of(1000, 1);
            SavedConversation a1 = conv_stage_grow(a0, 24, 2);
            SavedConversation a2 = conv_stage_grow(a1, 24, 3);
            const uint64_t serial = 1300;
            check(store.put(std::move(a0), serial, id), "33: the first park writes .bin + .stage0.bin");
            check(fs::exists(rdir / "conv-1300.bin") && fs::exists(rdir / "conv-1300.stage0.bin"),
                  "33: both files exist");
            check(!fs::exists(rdir / "conv-1300.dlt"), "33: no journal yet");
            const uint64_t bin_size = fs::file_size(rdir / "conv-1300.bin");
            const uint64_t stage_size = fs::file_size(rdir / "conv-1300.stage0.bin");
            check(store.put_async(std::move(a1), serial + 1, id), "33: the second park is queued");
            store.flush_pending();
            check(fs::exists(rdir / "conv-1300.dlt"), "33: the second park appended a journal record");
            check(fs::file_size(rdir / "conv-1300.bin") == bin_size, "33: the main .bin was NOT rewritten");
            check(fs::file_size(rdir / "conv-1300.stage0.bin") == stage_size,
                  "33: the stage file was NOT rewritten either");
            check(store.size() == 1, "33: one entry (the base, extended)");
            check(store.put_async(std::move(a2), serial + 2, id), "33: the third park is queued");
            store.flush_pending();
            check(store.size() == 1, "33: still one entry");
            const uint64_t journal_size = fs::file_size(rdir / "conv-1300.dlt");
            check(journal_size > 0, "33: the journal holds two records");
            check(journal_size < (bin_size + stage_size) / 2,
                  "33: two records cost less than half a full 2-GPU spill");
            // take() folds the journal over BOTH images
            ConversationCache::Match m;
            m.serial = 1300; m.tokens = 1048; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value(), "33: take returns the folded image");
            SavedConversation want = conv_stage_grow(conv_stage_grow(conv_stage_of(1000, 1), 24, 2), 24, 3);
            check(same_conv(*back, want), "33: the main folds to the third turn's state");
            check(back->stage_images.size() == 1 && same_conv(back->stage_images[0], want.stage_images[0]) &&
                  back->stage_images[0].layer_lo == 40 && back->stage_images[0].layer_hi == 48,
                  "33: the stage folds to the third turn's state too");
            check(store.size() == 1, "33: the entry stayed in the manifest after take");
        }
        // restart: the .idx fast path carries the pair; a fresh take restores both images
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            check(store.reload_from_disk(id) == 1, "33: reload loads the (base + journal) entry");
            ConversationCache::Match m;
            m.serial = 1300; m.tokens = 1048; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value() && back->live.ids.size() == 1048 && back->stage_images.size() == 1 &&
                      back->stage_images[0].live.ids.size() == 1048,
                  "33: after the restart, both images restore folded");
        }
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 34. A different layer split never appends: same stage count but a different carve is
    // not the same base — the full write replaces the entry (a journal built for one carve
    // cannot chain onto another).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_stages_mismatch";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        check(store.put(conv_stage_of(1000, 1), 1400, id), "34: the 2-GPU base");
        // a capture from ANOTHER split: the stage's carve moved (44.. instead of 40..)
        SavedConversation other = conv_stage_grow(conv_stage_of(1000, 1), 24, 2);
        other.stage_images[0].layer_lo = 44;
        check(store.put_async(std::move(other), 1401, id), "34: the other split's park is queued");
        store.flush_pending();
        check(store.size() == 1, "34: one entry — the other split's spill superseded the base");
        check(!fs::exists(rdir / "conv-1400.bin") && !fs::exists(rdir / "conv-1400.stage0.bin"),
              "34: the old split's spill went (full write + cleanup)");
        check(fs::exists(rdir / "conv-1401.bin") && fs::exists(rdir / "conv-1401.stage0.bin"),
              "34: the other split wrote its own spill");
        check(!fs::exists(rdir / "conv-1400.dlt") && !fs::exists(rdir / "conv-1401.dlt"),
              "34: no journal was ever created");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 35. Append-only with stages: turns past every old bound keep appending ONE record
    // that covers the main AND the stage — neither parent file is ever rewritten, and take()
    // folds the whole chain over both images.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_stages_append_only";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation cur = conv_stage_of(1000, 1);
        check(store.put(std::move(cur), 1500, id), "35: the 2-GPU base");
        cur = conv_stage_of(1000, 1);
        const uint64_t bin_size = fs::file_size(rdir / "conv-1500.bin");
        const uint64_t stage_size = fs::file_size(rdir / "conv-1500.stage0.bin");
        for (int turn = 1; turn <= 24; ++turn) {
            SavedConversation next = conv_stage_grow(cur, 30, uint8_t(turn));
            SavedConversation queued = next;   // the queue takes a copy; the chain keeps the original
            cur = std::move(next);
            check(store.put_async(std::move(queued), uint64_t(1500 + turn), id), "35: a delta turn");
            store.flush_pending();
        }
        check(fs::exists(rdir / "conv-1500.dlt"), "35: the journal survived — nothing folds it");
        check(fs::file_size(rdir / "conv-1500.bin") == bin_size &&
              fs::file_size(rdir / "conv-1500.stage0.bin") == stage_size,
              "35: neither the parent .bin nor the stage file was ever rewritten");
        check(store.size() == 1, "35: one entry after every turn");
        ConversationCache::Match m;
        m.serial = 1500; m.tokens = (int64_t) cur.live.ids.size(); m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && same_conv(*back, cur),
              "35: the parent + every record restores the newest state");
        check(back->stage_images.size() == 1 && same_conv(back->stage_images[0], cur.stage_images[0]),
              "35: the stage folds to the newest state too");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 36. The report's own shape (the case this whole block exists for): the client
    // re-renders between parks and the divergence lands BELOW the old checkpoint (measured:
    // 29 ids below it), so neither exact cut can see it and no .dlt is ever created — one full
    // .bin per park. With a pinned shared root on disk (the run had pfx-*.bin at 22259), the
    // common-prefix cut at the old render's frontier appends ONE record: no new .bin, and
    // take() restores the re-rendered state byte-exactly.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_lcp_real";
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            // the run's park 5814: 23143 live ids, deepest checkpoint at 22728
            SavedConversation a = conv_at(23143, 22728, 1);
            // the pinned shared root: 22259 ids (pfx-*.bin, on disk since the run's first request)
            const std::vector<int32_t> pin_ids(a.live.ids.begin(), a.live.ids.begin() + 22259);
            check(store.put(pin_image(pin_ids), 300, id, {}, true), "36: the pinned root lands");
            check(store.put(std::move(a), 5814, id), "36: park 5814 (the base)");
            const uint64_t bin_size = fs::file_size(rdir / "conv-5814.bin");
            // park 5815: the same conversation re-rendered — 22699 common ids (29 below the old
            // checkpoint), then a fresh 999-id turn: 23698 ids
            SavedConversation b = conv_lcp_turn(conv_at(23143, 22728, 1), 22699, 999, 2);
            check(b.live.ids.size() == 23698, "36: the re-rendered park is 23698 ids");
            check(store.put_async(std::move(b), 5815, id), "36: park 5815 is queued");
            store.flush_pending();
            check(fs::exists(rdir / "conv-5814.dlt"), "36: the .dlt EXISTS (the fix)");
            check(!fs::exists(rdir / "conv-5815.bin"), "36: no new .bin for the re-rendered park");
            check(fs::file_size(rdir / "conv-5814.bin") == bin_size,
                  "36: the base .bin was NOT rewritten");
            check(store.size() == 2, "36: two entries (the pin + the extended conversation)");
            const uint64_t journal_size = fs::file_size(rdir / "conv-5814.dlt");
            check(journal_size > 0 && journal_size < bin_size / 2,
                  "36: one record costs far less than the .bin it saved");
            // the folded state IS the re-rendered park, byte-exactly
            ConversationCache::Match m;
            m.serial = 5814; m.tokens = 23698; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value(), "36: take returns the folded image");
            check(same_conv(*back, conv_lcp_turn(conv_at(23143, 22728, 1), 22699, 999, 2)),
                  "36: base + record == the re-rendered park, byte-exactly");
        }
        // restart: the .idx fast path carries the post-apply ids; the folded state restores
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            check(store.reload_from_disk(id) == 2, "36: reload loads the pin and the entry");
            ConversationCache::Match m;
            m.serial = 5814; m.tokens = 23698; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value() && back->live.ids.size() == 23698,
                  "36: after the restart, the folded state restores");
        }
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 37. A divergence INSIDE the pinned root is not a conversation: the cut falls at or
    // below pin_len — the region every conversation of this client shares — so no delta, a
    // full write, and every entry coexists (the keep > pin_len guard).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_lcp_in_pin";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation a = conv_at(1000, 850, 1);
        const std::vector<int32_t> pin_ids(a.live.ids.begin(), a.live.ids.begin() + 600);
        check(store.put(pin_image(pin_ids), 301, id, {}, true), "37: the pinned root (600 ids)");
        check(store.put(std::move(a), 400, id), "37: the base");
        // the next park diverges at 550, inside the pin (the half rule alone would take it:
        // 550*2 >= 750) — the shared-root guard must refuse
        SavedConversation b = conv_lcp_turn(conv_at(1000, 850, 1), 550, 200, 2);
        check(store.put_async(std::move(b), 401, id), "37: the divergent park is queued");
        store.flush_pending();
        check(!fs::exists(rdir / "conv-400.dlt"), "37: no journal (the cut is inside the root)");
        check(fs::exists(rdir / "conv-401.bin"), "37: the park full-wrote its own .bin");
        check(store.size() == 3, "37: pin + base + the new park all coexist");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 38. The half rule on the common-prefix cut: a base carrying less than half the new
    // image does not save the rewrite — the codec would refuse the plan, and the selector must
    // not even offer it (1000 common ids under a 3000-id park).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_lcp_half";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation a = conv_at(1200, 1050, 1);
        const std::vector<int32_t> pin_ids(a.live.ids.begin(), a.live.ids.begin() + 400);
        check(store.put(pin_image(pin_ids), 302, id, {}, true), "38: the pinned root (400 ids)");
        check(store.put(std::move(a), 500, id), "38: the base");
        // 1000 common ids — past the pin (400) and at the frontier (1050): both identity guards
        // PASS; the half rule alone must refuse (1000*2 < 3000)
        SavedConversation b = conv_lcp_turn(conv_at(1200, 1050, 1), 1000, 2000, 2);
        check(b.live.ids.size() == 3000, "38: the new park is 3000 ids");
        check(store.put_async(std::move(b), 501, id), "38: the park is queued");
        store.flush_pending();
        check(!fs::exists(rdir / "conv-500.dlt"), "38: no journal (the base is under half)");
        check(fs::exists(rdir / "conv-501.bin"), "38: the park full-wrote");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 39. Another conversation of the same client: it shares the pinned root and even the
    // first message past it, but its divergence sits FAR below the old render's frontier —
    // not this conversation's re-render. No delta onto the old entry; the old conversation
    // stays restorable by its own prefix (the lineage guard's whole purpose).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_lcp_stranger";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation a = conv_at(1200, 1050, 1);
        const std::vector<int32_t> pin_ids(a.live.ids.begin(), a.live.ids.begin() + 400);
        check(store.put(pin_image(pin_ids), 303, id, {}, true), "39: the pinned root (400 ids)");
        check(store.put(std::move(a), 600, id), "39: conversation X parks");
        // conversation Y: shares root + 200 ids of the first message (the cut at 600 is past
        // the pin and past the half rule: 600*2 >= 1150) but far below X's frontier (1050)
        SavedConversation y = conv_lcp_turn(conv_at(1200, 1050, 1), 600, 550, 3);
        check(y.live.ids.size() == 1150, "39: Y is 1150 ids");
        check(store.put_async(std::move(y), 601, id), "39: conversation Y parks");
        store.flush_pending();
        check(!fs::exists(rdir / "conv-600.dlt"), "39: no journal onto X's entry");
        check(fs::exists(rdir / "conv-601.bin"), "39: Y wrote its own spill");
        check(store.size() == 3, "39: both conversations coexist");
        // X is still restorable by its own prefix — Y's park did not consume it
        ConversationCache::Match m;
        m.serial = 600; m.tokens = 1200; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && same_conv(*back, conv_at(1200, 1050, 1)),
              "39: X restores whole after Y's park");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 40. Without a pinned root on disk (pin_len == 0) the common-prefix path stays OFF:
    // only today's exact cuts. A partial-prefix re-render full-writes; the checkpoint-anchored
    // re-render (today's rule) still appends.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_lcp_no_pin";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        check(store.put(conv_at(1000, 500, 1), 700, id), "40: the base (no pin on disk)");
        // a re-render that diverges BELOW the checkpoint (at 400): a partial prefix no exact
        // cut can see — with no pin to anchor identity, the conservative full write
        SavedConversation b = conv_lcp_turn(conv_at(1000, 500, 1), 400, 600, 2);
        check(store.put_async(std::move(b), 701, id), "40: the partial-prefix park is queued");
        store.flush_pending();
        check(!fs::exists(rdir / "conv-700.dlt") && !fs::exists(rdir / "conv-701.dlt"),
              "40: no journal from a partial prefix without a pin");
        check(store.size() == 2, "40: both spills on disk (today's shape)");
        // today's rule still works: a re-render anchored exactly at the checkpoint appends —
        // and nothing ever folds the record away, so the .dlt itself is the proof
        SavedConversation c = conv_rerender(conv_at(1000, 500, 1), 300, 3);
        check(store.put_async(std::move(c), 702, id), "40: the checkpoint-anchored park is queued");
        store.flush_pending();
        check(!fs::exists(rdir / "conv-702.bin"),
              "40: no new .bin — the park appended to the base entry");
        check(fs::exists(rdir / "conv-700.dlt"), "40: the record landed in the journal");
        ConversationCache::Match m;
        m.serial = 700; m.tokens = 800; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && same_conv(*back, conv_rerender(conv_at(1000, 500, 1), 300, 3)),
              "40: the exact ckpt cut still chains (today's rule)");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 41. The chain: park after park after park, each re-rendered, each diverging below the
    // previous checkpoint — every one of them appends to the SAME entry (the reporting run's
    // churn: 4 parks, 0 journals, 4.9 GiB; with the fix: 4 parks, 3 records, one .bin). The
    // take() below proves the two-record FOLD, base → record 1 → record 2.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_lcp_chain";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation a0 = conv_at(4000, 3900, 1);
        const std::vector<int32_t> pin_ids(a0.live.ids.begin(), a0.live.ids.begin() + 400);
        check(store.put(pin_image(pin_ids), 304, id, {}, true), "41: the pinned root");
        SavedConversation parked = a0;
        check(store.put(std::move(parked), 800, id), "41: park 1 (the base)");
        const uint64_t bin_size = fs::file_size(rdir / "conv-800.bin");
        // park 2: diverges 29 ids below the old checkpoint (3871 under 3900); its own deepest
        // checkpoint (the last turn boundary) at 3880
        SavedConversation b = conv_lcp_turn(a0, 3871, 30, 2, 3880);
        SavedConversation queued = b;
        check(store.put_async(std::move(queued), 801, id), "41: park 2 queued");
        store.flush_pending();
        check(fs::exists(rdir / "conv-800.dlt"), "41: record 1 appended");
        const uint64_t journal_1 = fs::file_size(rdir / "conv-800.dlt");
        // park 3: re-rendered again — the entry now holds park 2's state, whose checkpoint
        // sits at 3880; the new cut is ITS frontier minus 29 (3851)
        SavedConversation c = conv_lcp_turn(b, 3851, 40, 3);
        queued = c;
        check(store.put_async(std::move(queued), 802, id), "41: park 3 queued");
        store.flush_pending();
        check(fs::file_size(rdir / "conv-800.dlt") > journal_1, "41: record 2 appended (the chain)");
        check(fs::file_size(rdir / "conv-800.bin") == bin_size, "41: the .bin never moved");
        check(store.size() == 2, "41: pin + ONE conversation entry after three parks");
        check(!fs::exists(rdir / "conv-801.bin") && !fs::exists(rdir / "conv-802.bin"),
              "41: no per-park .bin files");
        // the whole chain folds back to park 3's state
        ConversationCache::Match m;
        m.serial = 800; m.tokens = 3891; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value(), "41: take returns the folded image");
        check(same_conv(*back, c), "41: base + 2 records == park 3, byte-exactly");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 42. The 2-GPU shape with an intermediate cut: the re-rendered park appends ONE record
    // covering the main AND the stage (neither file rewritten); take() folds both.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_lcp_stages";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation a = conv_stage_of(4000, 1);
        {   // move the checkpoint to a real park's shape: the last turn boundary (3900)
            std::vector<int32_t> ck(a.live.ids.begin(), a.live.ids.begin() + 3900);
            a.checkpoints = {jckpt(ck, 51)};
            a.stage_images[0].checkpoints = a.checkpoints;
        }
        const std::vector<int32_t> pin_ids(a.live.ids.begin(), a.live.ids.begin() + 400);
        check(store.put(pin_image(pin_ids), 305, id, {}, true), "42: the pinned root");
        SavedConversation parked = a;
        check(store.put(std::move(parked), 900, id), "42: the 2-GPU base");
        const uint64_t bin_size = fs::file_size(rdir / "conv-900.bin");
        const uint64_t stage_size = fs::file_size(rdir / "conv-900.stage0.bin");
        // the re-rendered park: 3871 common ids (29 below the 3900 checkpoint), a fresh 30
        SavedConversation b = conv_stage_lcp(a, 3871, 30, 2);
        check(b.live.ids.size() == 3901, "42: the re-rendered park is 3901 ids");
        SavedConversation queued = b;
        check(store.put_async(std::move(queued), 901, id), "42: the re-rendered park is queued");
        store.flush_pending();
        check(fs::exists(rdir / "conv-900.dlt"), "42: the .dlt covers both images");
        check(fs::file_size(rdir / "conv-900.bin") == bin_size &&
              fs::file_size(rdir / "conv-900.stage0.bin") == stage_size,
              "42: neither file was rewritten");
        check(store.size() == 2, "42: pin + ONE entry");
        ConversationCache::Match m;
        m.serial = 900; m.tokens = 3901; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value(), "42: take returns the folded image");
        check(same_conv(*back, b), "42: the main folds to the re-rendered state");
        check(back->stage_images.size() == 1 && same_conv(back->stage_images[0], b.stage_images[0]),
              "42: the stage folds to the re-rendered state too");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 43. drop_superseded stays untouched: after a common-prefix delta no NEW .bin appears
    // (the park created no entry), the delta base survives its own append's cleanup
    // (protect_serial), and a DIFFERENT conversation's files are never collateral damage.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_lcp_cleanup";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation x = conv_at(4000, 3900, 1);
        const std::vector<int32_t> pin_ids(x.live.ids.begin(), x.live.ids.begin() + 400);
        check(store.put(pin_image(pin_ids), 306, id, {}, true), "43: the pinned root");
        SavedConversation parked = x;
        check(store.put(std::move(parked), 1000, id), "43: conversation X parks");
        SavedConversation z = conv_image(800, 7);   // a different conversation of the client
        for (auto& t : z.live.ids) t += 7777;
        for (auto& c : z.checkpoints)
            for (auto& t : c.ids) t += 7777;
        check(store.put(std::move(z), 1001, id), "43: conversation Z parks");
        auto count_bins = [&rdir] {
            size_t n = 0;
            for (const auto& f : fs::directory_iterator(rdir)) {
                const std::string name = f.path().filename().string();
                if (name.rfind("conv-", 0) == 0 && name.find(".stage") == std::string::npos &&
                    name.size() > 4 && name.compare(name.size() - 4, 4, ".bin") == 0)
                    ++n;
            }
            return n;
        };
        check(count_bins() == 2, "43: two .bin files before the re-rendered park");
        // X re-rendered: 3871 common ids (29 below X's 3900 checkpoint), a fresh 40
        SavedConversation x2 = conv_lcp_turn(x, 3871, 40, 2);
        check(store.put_async(std::move(x2), 1002, id), "43: X's re-rendered park is queued");
        store.flush_pending();
        check(count_bins() == 2, "43: still two .bin files (the park created no entry)");
        check(fs::exists(rdir / "conv-1000.dlt"), "43: X's park appended to X's entry");
        check(fs::exists(rdir / "conv-1001.bin"), "43: Z's .bin survived X's park");
        check(store.size() == 3, "43: pin + X (extended) + Z");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 44. The v3 sidecar end-to-end: every append publishes conv-<serial>.rst, take() folds
    // through it, and the budget's accounting stays EXACTLY the sum of the files on disk
    // (main + stages + journal + sidecar) across appends, take, the fast (.idx) reload, the
    // slow (no .idx) reload and an eviction.
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_v3";
        // what the store counts: every conv-*/pfx-* file except the .idx metadata
        auto disk_bytes = [] (const fs::path& d) {
            uint64_t n = 0;
            std::error_code ec;
            for (const auto& f : fs::directory_iterator(d, ec)) {
                if (ec) break;
                const std::string name = f.path().filename().string();
                if (name.rfind("conv-", 0) != 0 && name.rfind("pfx-", 0) != 0) continue;
                if (name.size() > 4 && name.compare(name.size() - 4, 4, ".idx") == 0) continue;
                if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) continue;
                n += (uint64_t) fs::file_size(f, ec);
            }
            return n;
        };
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            SavedConversation a0 = conv_image(1000, 1);
            SavedConversation a1 = conv_grow(a0, 24, 2);
            SavedConversation a2 = conv_grow(a1, 24, 3);
            SavedConversation a3 = conv_grow(a2, 24, 4);
            check(store.put(std::move(a0), 800, id), "44: the base park");
            check(!fs::exists(rdir / "conv-800.rst"), "44: no sidecar before any append");
            check(store.put_async(std::move(a1), 801, id), "44: append 1 queued");
            store.flush_pending();
            check(fs::exists(rdir / "conv-800.rst"), "44: append 1 published the sidecar");
            check(store.put_async(std::move(a2), 802, id), "44: append 2 queued");
            store.flush_pending();
            check(store.put_async(std::move(a3), 803, id), "44: append 3 queued");
            store.flush_pending();
            check(store.bytes() == disk_bytes(rdir), "44: accounting == disk after 3 appends");
            ConversationCache::Match m;
            m.serial = 800; m.tokens = 1072; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value(), "44: take folds base + 3 v3 records + sidecar");
            check(same_conv(*back, conv_grow(conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 3), 24, 4)),
                  "44: the folded image is turn 3's, byte-exactly");
            check(store.bytes() == disk_bytes(rdir), "44: accounting == disk after take");
        }
        // fast path: the .idx carries it (the .rst's bytes ride inside file_bytes)
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            check(store.reload_from_disk(id) == 1, "44: fast-path reload loads the entry");
            check(store.bytes() == disk_bytes(rdir), "44: accounting == disk after the fast reload");
            ConversationCache::Match m;
            m.serial = 800; m.tokens = 1072; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value() && back->live.ids.size() == 1072, "44: the fast path restores");
        }
        // slow path: no .idx — the fold rebuilds the manifest from .bin + .dlt + .rst
        {
            fs::remove(rdir / "conv-800.idx");
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            check(store.reload_from_disk(id) == 1, "44: slow-path reload loads the entry");
            check(store.bytes() == disk_bytes(rdir), "44: accounting == disk after the slow reload");
            ConversationCache::Match m;
            m.serial = 800; m.tokens = 1072; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value() && same_conv(*back, conv_grow(conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 3), 24, 4)),
                  "44: the slow path restores turn 3 byte-exactly");
        }
        // eviction: the whole spill goes, sidecar included
        {
            ConversationDiskStore store(rdir.string(), 1, &id);   // a 1-byte budget evicts on load
            store.reload_from_disk(id);
            check(store.size() == 0, "44: the tight budget evicted the entry");
            check(!fs::exists(rdir / "conv-800.bin") && !fs::exists(rdir / "conv-800.dlt") &&
                  !fs::exists(rdir / "conv-800.rst") && !fs::exists(rdir / "conv-800.idx"),
                  "44: the eviction removed .bin, .dlt, .rst and .idx");
            check(store.bytes() == disk_bytes(rdir), "44: accounting == disk after the eviction");
        }
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 45. STRATA_SPILL_JOURNAL_V3=0 (the escape hatch): a reloaded store with v3 OFF
    // appends v2 records after the v3 ones; the sidecar STAYS (the earlier records chain
    // through it) and the fold lands the newest state ─────────────────────────────────────
    {
        const auto rdir = dir / "journal_v3_off";
        {
            jsetenv("STRATA_SPILL_JOURNAL", "1");
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            check(store.put(conv_image(1000, 1), 810, id), "45: the base park");
            check(store.put_async(conv_grow(conv_image(1000, 1), 24, 2), 811, id), "45: v3 append 1");
            store.flush_pending();
            check(store.put_async(conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 3), 812, id),
                  "45: v3 append 2");
            store.flush_pending();
            check(fs::exists(rdir / "conv-810.rst"), "45: the sidecar is live");
            junsetenv("STRATA_SPILL_JOURNAL");
        }
        {
            jsetenv("STRATA_SPILL_JOURNAL", "1");
            jsetenv("STRATA_SPILL_JOURNAL_V3", "0");
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            check(store.reload_from_disk(id) == 1, "45: reload with v3 off");
            // a v2 record lands after the v3 pair; the sidecar survives it
            check(store.put_async(conv_grow(conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 3), 24, 4), 813, id),
                  "45: v2 append 3 queued");
            store.flush_pending();
            check(fs::exists(rdir / "conv-810.rst"),
                  "45: the v2 append KEPT the sidecar (the v3 records still chain through it)");
            ConversationCache::Match m;
            m.serial = 810; m.tokens = 1072; m.live = true;
            auto back = store.take(m, id);
            check(back.has_value() && back->live.ids.size() == 1072,
                  "45: the mixed v3,v3,v2 journal folds to turn 3");
            junsetenv("STRATA_SPILL_JOURNAL_V3");
            junsetenv("STRATA_SPILL_JOURNAL");
        }
    }

    // ── 46. A failed sidecar write rolls the record back: a directory squatting on
    // conv-<serial>.rst.tmp makes state_write fail; the journal returns to the size the
    // entry remembers (here: never existed), and the park falls back to the full write ─────
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_rst_fail";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        check(store.put(conv_image(1000, 1), 820, id), "46: the base park");
        // the blocker: the .rst temp name is taken by a directory
        fs::create_directories(rdir / "conv-820.rst.tmp");
        check(store.put_async(conv_grow(conv_image(1000, 1), 24, 2), 821, id),
              "46: the park is queued");
        store.flush_pending();
        check(!fs::exists(rdir / "conv-820.dlt"),
              "46: the rolled-back journal is gone (it never existed before the append)");
        check(!fs::exists(rdir / "conv-820.rst"), "46: no sidecar was published");
        fs::remove_all(rdir / "conv-820.rst.tmp");
        // the fallback full write replaced the entry: the new serial's .bin carries turn 1
        check(fs::exists(rdir / "conv-821.bin"), "46: the full write landed under the park's serial");
        check(!fs::exists(rdir / "conv-820.bin"),
              "46: the old base went with the entry the full write replaced");
        ConversationCache::Match m;
        m.serial = 821; m.tokens = 1024; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && back->live.ids.size() == 1024,
              "46: the full write's image restores (the fallback)");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 47. purge_on_startup sweeps the v3 sidecar and its temp too ──────────────────────
    {
        const auto rdir = dir / "purge_rst";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        check(store.put(conv_image(1000, 1), 830, id), "47: the base park");
        check(store.put_async(conv_grow(conv_image(1000, 1), 24, 2), 831, id), "47: an append");
        store.flush_pending();
        junsetenv("STRATA_SPILL_JOURNAL");
        check(fs::exists(rdir / "conv-830.rst"), "47: the sidecar exists");
        { std::FILE* f = std::fopen((rdir / "conv-830.rst.tmp").c_str(), "wb"); std::fputs("x", f); std::fclose(f); }
        { std::FILE* f = std::fopen((rdir / "keep-rst.txt").c_str(), "wb"); std::fputs("keep", f); std::fclose(f); }
        store.purge_on_startup();
        check(!fs::exists(rdir / "conv-830.bin") && !fs::exists(rdir / "conv-830.dlt") &&
              !fs::exists(rdir / "conv-830.rst") && !fs::exists(rdir / "conv-830.rst.tmp"),
              "47: the purge removed .bin, .dlt, .rst and .rst.tmp");
        check(fs::exists(rdir / "keep-rst.txt"), "47: unrelated files stay");
    }

    // ── 48. Serial collision + layer split (the use_serial fix): a DIFFERENT conversation
    // colliding with a held serial writes under a fresh serial — its stage files land beside
    // THEIR main (conv-<fresh>.stage0.bin), and take() finds them through the entry's serial
    {
        const auto rdir = dir / "collision_stages";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        check(store.put(conv_image(1000, 1), 840, id), "48: conversation A parks at serial 840");
        // conversation B: same serial (the cache counter restarted), a layer split, and
        // different ids (NOT a branch of A) — the collision guard must move it to a fresh serial
        SavedConversation b = conv_stage_of(1000, 5);
        const int32_t shift = 500000;
        for (auto& t : b.live.ids) t += shift;
        for (auto& c : b.checkpoints) for (auto& t : c.ids) t += shift;
        for (auto& st : b.stage_images) {
            for (auto& t : st.live.ids) t += shift;
            for (auto& c : st.checkpoints) for (auto& t : c.ids) t += shift;
        }
        check(store.put(std::move(b), 840, id), "48: conversation B parks under the colliding serial");
        check(fs::exists(rdir / "conv-841.bin"), "48: B's main landed under the fresh serial 841");
        check(fs::exists(rdir / "conv-841.stage0.bin"),
              "48: B's stage landed under the FRESH serial (not beside A's)");
        check(store.size() == 2, "48: both conversations are held");
        // B restores WITH its stage
        ConversationCache::Match mb;
        mb.serial = 841; mb.tokens = 1000; mb.live = true;
        auto back_b = store.take(mb, id);
        check(back_b.has_value(), "48: take() finds B through the fresh serial");
        check(back_b->stage_images.size() == 1 &&
                  back_b->stage_images[0].layer_lo == 40 && back_b->stage_images[0].layer_hi == 48,
              "48: B's stage image restored (the use_serial fix)");
        // and A is untouched
        ConversationCache::Match ma;
        ma.serial = 840; ma.tokens = 1000; ma.live = true;
        auto back_a = store.take(ma, id);
        check(back_a.has_value() && back_a->stage_images.empty(),
              "48: A restores alone, as parked");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 49. Mid-journal corruption with a sidecar: take() fails (the fold is failed), the
    // entry leaves the manifest and EVERY file stays on disk; a reload without the .idx
    // skips the entry the same way, keeping the files ─────────────────────────────────────
    {
        const auto rdir = dir / "journal_midcorrupt";
        {
            jsetenv("STRATA_SPILL_JOURNAL", "1");
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            check(store.put(conv_image(1000, 1), 850, id), "49: the base park");
            check(store.put_async(conv_grow(conv_image(1000, 1), 24, 2), 851, id), "49: append 1");
            store.flush_pending();
            check(store.put_async(conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 3), 852, id),
                  "49: append 2");
            store.flush_pending();
            check(store.put_async(conv_grow(conv_grow(conv_grow(conv_image(1000, 1), 24, 2), 24, 3), 24, 4), 853, id),
                  "49: append 3");
            store.flush_pending();
            // corrupt the LAST record's payload (80 bytes before the trailer, inside it)
            {
                const uint64_t size = fs::file_size(rdir / "conv-850.dlt");
                std::FILE* f = std::fopen((rdir / "conv-850.dlt").c_str(), "r+b");
                std::fseek(f, (long) (size - 100), SEEK_SET);
                const int c = std::fgetc(f);
                std::fseek(f, (long) (size - 100), SEEK_SET);
                std::fputc(c ^ 0xff, f);
                std::fclose(f);
            }
            ConversationCache::Match m;
            m.serial = 850; m.tokens = 1072; m.live = true;
            auto back = store.take(m, id);
            check(!back.has_value(), "49: the take fails (the fold is failed)");
            check(store.size() == 0, "49: the entry left the manifest");
            check(fs::exists(rdir / "conv-850.bin") && fs::exists(rdir / "conv-850.dlt") &&
                  fs::exists(rdir / "conv-850.rst"),
                  "49: every file KEPT on disk (suspect, not garbage)");
            junsetenv("STRATA_SPILL_JOURNAL");
        }
        // a reload without the .idx: the slow path skips the entry, files still kept (H6)
        {
            fs::remove(rdir / "conv-850.idx");
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store.reload_from_disk(id);
            check(loaded == 0 && store.size() == 0, "49: the slow-path reload skipped the entry");
            check(fs::exists(rdir / "conv-850.bin") && fs::exists(rdir / "conv-850.dlt") &&
                  fs::exists(rdir / "conv-850.rst"),
                  "49: the reload kept every file (H6: no manifest data from a failed fold)");
        }
    }

    // ── 50. Orphan v3 files: a .rst (and .rst.tmp) whose main .bin is gone is swept by the
    // reload scan ─────────────────────────────────────────────────────────────────────────
    {
        const auto rdir = dir / "orphan_rst";
        fs::create_directories(rdir);
        { std::FILE* f = std::fopen((rdir / "conv-860.rst").c_str(), "wb"); std::fputs("SRST", f); std::fclose(f); }
        { std::FILE* f = std::fopen((rdir / "conv-860.rst.tmp").c_str(), "wb"); std::fputs("x", f); std::fclose(f); }
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        store.reload_from_disk(id);
        check(!fs::exists(rdir / "conv-860.rst") && !fs::exists(rdir / "conv-860.rst.tmp"),
              "50: the orphan sidecar and its temp were swept");
    }

    // ── 51. The manifest's checkpoint mirrors the FILE's selection: spill_resume_checkpoint's
    // — the chain's DEEPEST checkpoint, ignoring a pinned root (the pinned system prompt is
    // persisted apart as pfx-*.bin; the turn boundary is the resume point a re-rendering client
    // needs). take() must return exactly that one, and the manifest must promise the same ───
    {
        const auto rdir = dir / "ckpt_selection";
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation img = conv_image(1024, 1);
        const std::vector<int32_t> pinned_ids(img.live.ids.begin(), img.live.ids.begin() + 200);
        ConversationCheckpoint pinned = jckpt(pinned_ids, 61);
        pinned.pinned = true;
        ConversationCheckpoint deep = jckpt(
            std::vector<int32_t>(img.live.ids.begin(), img.live.ids.begin() + 900), 62);
        img.checkpoints.push_back(pinned);
        img.checkpoints.push_back(deep);
        check(store.put(std::move(img), 870, id), "51: the park lands");
        // an append extends the journal; its record/sidecar carry the deepest selection too
        SavedConversation next = conv_grow(conv_image(1024, 1), 24, 2);
        ConversationCheckpoint np = jckpt(pinned_ids, 61);
        np.pinned = true;
        next.checkpoints.push_back(np);
        next.checkpoints.push_back(jckpt(
            std::vector<int32_t>(next.live.ids.begin(), next.live.ids.begin() + 900), 63));
        // the probe prompt (901 tokens: the turn boundary 900 + one more, so the strict-prefix
        // match accepts it) is captured BEFORE the put_async moves `next` away
        const std::vector<int32_t> probe(next.live.ids.begin(), next.live.ids.begin() + 901);
        check(store.put_async(std::move(next), 871, id), "51: the append queues");
        store.flush_pending();
        ConversationCache::Match m;
        m.serial = 870; m.tokens = 1048; m.live = true;
        // best() must promise the turn boundary (900), never the pinned root (200)
        const auto match = store.best(probe, {}, false);
        check(match.tokens == 900 && match.serial == 870,
              "51: best() promises the deepest checkpoint (the turn boundary)");
        auto back = store.take(m, id);
        check(back.has_value(), "51: take returns the folded image");
        check(back->checkpoints.size() == 1 && back->checkpoints[0].ids.size() == 900,
              "51: the folded chain is the DEEPEST checkpoint (the file's selection)");
        // the manifest's correction kept the same selection (a second take agrees)
        auto back2 = store.take(m, id);
        check(back2.has_value() && back2->checkpoints.size() == 1 &&
                  back2->checkpoints[0].ids.size() == 900,
              "51: the manifest's ckpt and the file's agree across takes");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 52. STRATA_SPILL_DIAG=1: the [delta/diag] breakdown line goes to stderr and the
    // append itself still lands (Phase 0's instrument is inert without the flag) ──────────
    {
        const auto rdir = dir / "diag";
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        jsetenv("STRATA_SPILL_DIAG", "1");
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        check(store.put(conv_image(1000, 1), 880, id), "52: the base park");
        check(store.put_async(conv_grow(conv_image(1000, 1), 24, 2), 881, id), "52: the append queues");
        store.flush_pending();
        check(fs::exists(rdir / "conv-880.dlt") && fs::exists(rdir / "conv-880.rst"),
              "52: with diag ON the append (record + sidecar) still lands");
        ConversationCache::Match m;
        m.serial = 880; m.tokens = 1024; m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && back->live.ids.size() == 1024, "52: the fold still restores");
        junsetenv("STRATA_SPILL_DIAG");
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 53. The zoocode/roocode resume regression: a pinned system prompt + turns re-rendered
    // WITHOUT the assistant's thinking tokens + a restart. The delta record (and the .rst
    // sidecar, and the manifest) must carry the chain's DEEPEST checkpoint — the turn boundary
    // the next prompt still contains — or every delta-restored resume falls back to the pinned
    // root and the engine recomputes the whole history past the system prompt. The engine's
    // post-restore derivation is replicated here: the live if a strict prefix of the request's
    // prompt, else the LONGEST restored checkpoint that is (generate.cpp does exactly this).
    {
        const auto rdir = dir / "resume_regression";
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        // capture-consistent images with EXPLICIT ids (the re-render model needs its own tokens)
        auto ids_image = [&](const std::vector<int32_t>& ids, uint8_t seed) {
            SavedConversation s;
            for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(200 + i);
            s.layer_lo = 0; s.layer_hi = 40;
            s.cvec = false;
            s.live = jckpt(ids, seed);
            const size_t tokens = ids.size();
            const int64_t cells = jcells(tokens);
            for (int layer = 0; layer < 3; ++layer) {
                ConversationKv kv;
                kv.format = 1; kv.cells = cells; kv.heads = JHeads; kv.head_dim = JHeadDim;
                kv.page_size = JPage;
                kv.pooled_rows = layer < 2 && tokens > 0 ? (int64_t) tokens / JIdxBlock + 1 : 0;
                kv.idx_dim = JIdxDim;
                kv.k = jfill(size_t(cells) * size_t(JHeads * JHeadDim * 2), uint8_t(layer * 10 + seed));
                kv.v = jfill(size_t(cells) * size_t(JHeads * JHeadDim * 2), uint8_t(layer * 10 + seed + 1));
                kv.k_scale = layer < 2 ? jfill(size_t(cells) * size_t(JHeads * 2), uint8_t(layer * 10 + seed + 2))
                                       : ConversationBuffer{};
                kv.v_scale = layer < 2 ? jfill(size_t(cells) * size_t(JHeads * 2), uint8_t(layer * 10 + seed + 3))
                                       : ConversationBuffer{};
                kv.pooled = layer < 2 && tokens > 0
                                ? jfill(size_t(kv.pooled_rows) * size_t(JIdxDim * 4), uint8_t(layer * 10 + seed + 4))
                                : ConversationBuffer{};
                s.kv.push_back(std::move(kv));
            }
            return s;
        };
        // the engine's resume derivation after a disk restore: the live if a strict prefix of
        // the prompt, else the longest restored checkpoint that is (generate.cpp's rule — the
        // turn boundary survives the client stripping the thinking tokens; the live does not)
        auto engine_resume = [](const SavedConversation& img, const std::vector<int32_t>& prompt) {
            int64_t resume = 0;
            if (conversation_prefix(img.live, prompt, {}) > 0) resume = (int64_t) img.live.ids.size();
            for (const auto& c : img.checkpoints)
                resume = std::max(resume, conversation_prefix(c, prompt, {}));
            return resume;
        };
        // the conversation: a 400-token pinned system prompt; turn k's prompt P_k, the
        // assistant's private part a_k (thinking + answer — what the client strips), and
        // P_{k+1} = P_k + the rendered answer + the next user turn
        std::vector<int32_t> sys;
        for (size_t i = 0; i < 400; ++i) sys.push_back(int32_t(3000 + i));
        std::vector<int32_t> P1 = sys;
        P1.push_back(3000 + 400); P1.push_back(3000 + 401);                       // +u1 → 402
        std::vector<int32_t> a1;
        for (size_t i = 0; i < 300; ++i) a1.push_back(int32_t(9000 + i));         // thinking + answer
        std::vector<int32_t> P2 = P1;
        for (size_t i = 0; i < 80; ++i) P2.push_back(int32_t(3100 + i));          // rendered answer
        for (size_t i = 0; i < 60; ++i) P2.push_back(int32_t(3200 + i));          // +u2 → 542
        std::vector<int32_t> a2;
        for (size_t i = 0; i < 200; ++i) a2.push_back(int32_t(9100 + i));
        std::vector<int32_t> P3 = P2;
        for (size_t i = 0; i < 60; ++i) P3.push_back(int32_t(3300 + i));          // rendered answer
        for (size_t i = 0; i < 70; ++i) P3.push_back(int32_t(3400 + i));          // +u3 → 672
        check(P1.size() == 402 && P2.size() == 542 && P3.size() == 672, "53: the turn model's lengths");
        auto with_private = [](const std::vector<int32_t>& p, const std::vector<int32_t>& a) {
            std::vector<int32_t> live = p;
            live.insert(live.end(), a.begin(), a.end());
            return live;
        };
        // park k's capture: live = P_k + a_k, chain = [the pinned root, T_{k-1}?, T_k]. The
        // kinds are what the engine sets at the cut sites (turn_at → turn, the pin → pin) — the
        // spill's resume-set selection reads them.
        ConversationCheckpoint pin = jckpt(sys, 61);
        pin.pinned = true;
        pin.kind = CheckpointKind::pin;
        ConversationCheckpoint T1 = jckpt(P1, 62);
        T1.kind = CheckpointKind::turn;
        ConversationCheckpoint T2 = jckpt(P2, 63);
        T2.kind = CheckpointKind::turn;
        SavedConversation park1 = ids_image(with_private(P1, a1), 1);
        park1.checkpoints = {pin, T1};
        SavedConversation park2 = ids_image(with_private(P2, a2), 2);
        park2.checkpoints = {pin, T1, T2};   // T1 is still a prefix of the live: the chain keeps it

        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            // the pinned shared root on disk (what find_delta_base's pin_len and the sibling
            // matches see) — its live IS its only checkpoint (the prefix shape)
            SavedConversation pfx = ids_image(sys, 9);
            pfx.checkpoints = {pin};
            check(store.put(std::move(pfx), 1, id, {}, true), "53: the pfx entry lands");
            // park 1 → the base .bin (a FULL write: nothing to extend yet)
            check(store.put(std::move(park1), 2, id), "53: park 1 lands (full write)");
            // request 2 restores: the base write carries the whole chain, so the engine finds
            // T1 — and with the fix best() PROMISES it too (the manifest's ckpt is the deepest)
            {
                const auto m = store.best(P2, {}, false);
                check(m.tokens == 402 && m.serial == 2, "53: request 2: best() promises T1 (402)");
                auto back = store.take(m, id);
                check(back.has_value(), "53: request 2: take restores");
                check(engine_resume(*back, P2) == 402,
                      "53: request 2: the engine resumes at T1, not the pinned root");
            }
            // park 2 → ONE delta record appends onto park 1's base
            check(store.put_async(std::move(park2), 3, id), "53: park 2 queues");
            store.flush_pending();
            check(fs::exists(rdir / "conv-2.dlt") && fs::file_size(rdir / "conv-2.dlt") > 0,
                  "53: park 2 appended a journal record");
            // request 3 restores through the fold: the .rst's resume set carries T2 (the
            // deepest turn) AND T1 (the previous boundary) — the engine resumes at T2, and a
            // rewind past it still finds T1 below
            {
                const auto m = store.best(P3, {}, false);
                check(m.tokens == 542 && m.serial == 2, "53: request 3: best() promises T2 (542)");
                auto back = store.take(m, id);
                check(back.has_value(), "53: request 3: take restores");
                check(back->checkpoints.size() == 2 &&
                          back->checkpoints[0].ids == T1.ids && back->checkpoints[1].ids == T2.ids,
                      "53: the folded chain carries the resume set {T1, T2}, never the pinned root");
                check(back->checkpoints[0].kind == CheckpointKind::turn &&
                          back->checkpoints[1].kind == CheckpointKind::turn,
                      "53: the folded set keeps the kinds (the .rst v2 kind byte)");
                check(engine_resume(*back, P3) == 542,
                      "53: request 3: the engine resumes at T2, not the pinned root");
            }
            // rollback (an EDITED turn): a prompt that rewinds PAST the last turn boundary.
            // The resume set carries T1 too (the previous boundary, still a prefix of the
            // edited render), so the restore lands on the PREVIOUS TURN instead of the pinned
            // root — the old deepest-only record collapsed here to the 400-token pin.
            std::vector<int32_t> rollback = P1;
            for (size_t i = 0; i < 80; ++i) rollback.push_back(int32_t(3500 + i));   // an EDITED turn
            for (size_t i = 0; i < 60; ++i) rollback.push_back(int32_t(3600 + i));
            {
                const auto m = store.best(rollback, {}, false);
                check(m.tokens == 402, "53: rollback: best() promises T1 (the previous boundary)");
                auto back = store.take(m, id);
                check(back.has_value() && engine_resume(*back, rollback) == 402,
                      "53: rollback: resume at T1 — one message re-read, not the whole history");
            }
        }
        // restart: a fresh store reloads the manifest from disk; the folded entry still
        // restores T2 for request 3
        {
            ConversationDiskStore fresh(rdir.string(), 1ull << 30, &id);
            check(fresh.reload_from_disk(id) >= 1, "53: the restart reloads the spill files");
            const auto m = fresh.best(P3, {}, false);
            check(m.tokens == 542, "53: restart: best() still promises T2");
            auto back = fresh.take(m, id);
            check(back.has_value() && engine_resume(*back, P3) == 542,
                  "53: restart: the engine still resumes at T2");
            // ── Phase 4, the heal: journals the AFFECTED builds (b4a5e5ef..b132fbcf) wrote
            // carry only the pinned root in their record/sidecar. Simulate turn 3's park as
            // one of those builds would have written it (the capture's real chain is
            // [pin, T3], the record keeps the pin alone), and check the two halves of the
            // recovery: the manifest is HONEST about the loss (the resume floors at the pin
            // for one request — a recompute, never a mismatch), and the conversation's NEXT
            // park appends a record carrying the real turn boundary again — self-healing.
            std::vector<int32_t> a3;
            for (size_t i = 0; i < 150; ++i) a3.push_back(int32_t(9200 + i));
            const std::vector<int32_t> live3 = with_private(P3, a3);
            SavedConversation park3_buggy = ids_image(live3, 3);
            park3_buggy.checkpoints = {pin};   // what the affected codec serialized
            check(fresh.put_async(std::move(park3_buggy), 4, id), "53: the affected-build park queues");
            fresh.flush_pending();
            std::vector<int32_t> P4 = P3;
            for (size_t i = 0; i < 50; ++i) P4.push_back(int32_t(3700 + i));   // rendered answer
            for (size_t i = 0; i < 60; ++i) P4.push_back(int32_t(3800 + i));    // +u4 → 782
            {
                const auto m4 = fresh.best(P4, {}, false);
                check(m4.tokens == 400,
                      "53: heal: the affected journal's floor is honest (the pinned root)");
                auto b4 = fresh.take(m4, id);
                check(b4.has_value() && engine_resume(*b4, P4) == 400,
                      "53: heal: ONE recompute from the pin — honest, never a mismatch");
            }
            // the fixed build's next park of this conversation: the append re-learns the
            // turn boundary (T4 = the whole of P4), and request 5 restores at it
            std::vector<int32_t> a4;
            for (size_t i = 0; i < 120; ++i) a4.push_back(int32_t(9300 + i));
            SavedConversation park4 = ids_image(with_private(P4, a4), 4);
            ConversationCheckpoint T4 = jckpt(P4, 64);
            T4.kind = CheckpointKind::turn;
            park4.checkpoints = {pin, T4};
            check(fresh.put_async(std::move(park4), 5, id), "53: the fixed build's park queues");
            fresh.flush_pending();
            std::vector<int32_t> P5 = P4;
            for (size_t i = 0; i < 40; ++i) P5.push_back(int32_t(3900 + i));    // rendered answer
            for (size_t i = 0; i < 50; ++i) P5.push_back(int32_t(4000 + i));    // +u5 → 872
            {
                const auto m5 = fresh.best(P5, {}, false);
                check(m5.tokens == 782, "53: heal: the next park re-learned T4 (782)");
                auto b5 = fresh.take(m5, id);
                check(b5.has_value() && engine_resume(*b5, P5) == 782,
                      "53: heal: request 5 resumes at the re-learned turn boundary");
            }
        }
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 54. The 2026-11-02 Kaggle restart regression, EXACTLY as the log shows it: a
    // --tail-role-token client (zoocode) whose prompt ends with the server's reasoning-effort
    // system turn, parked through the delta journal (the .bin parent + one .dlt record per
    // turn), the engine restarted, and the client's retry re-rendering the LAST user message
    // with a refreshed environment_details timestamp — the divergence lands INSIDE that
    // message, PAST the last request's turn boundary (T2) but BEFORE the previous one (T1).
    // The chain at the last park is {periodic, pin, T1, E1, T2, E2}; only T1 and below survive
    // the retry's rewrite. The old manifest promised only the deepest (E2, which carries the
    // effort tail the next prompt NEVER re-sends) and the journal folded the chain down to
    // that one checkpoint — so best() missed the conversation's own entry and the restore
    // collapsed to the pinned root (the log: disk_best=22259 of a 43552-token conversation,
    // 21023 tokens re-read at 197 tok/s, 106 s). The fix: the .rst carries the resume SET
    // {periodic, T1, T2, E2} and the manifest's ckpt_lens lets best() match any of them.
    {
        const auto rdir = dir / "effort_tail_restart";
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        auto engine_resume = [](const SavedConversation& img, const std::vector<int32_t>& prompt) {
            int64_t resume = 0;
            if (conversation_prefix(img.live, prompt, {}) > 0) resume = (int64_t) img.live.ids.size();
            for (const auto& c : img.checkpoints)
                resume = std::max(resume, conversation_prefix(c, prompt, {}));
            return resume;
        };
        auto rng = [](int32_t base, size_t n) {
            std::vector<int32_t> v;
            for (size_t i = 0; i < n; ++i) v.push_back(base + int32_t(i));
            return v;
        };
        auto cat = [](std::vector<int32_t> a, const std::vector<int32_t>& b) {
            a.insert(a.end(), b.begin(), b.end());
            return a;
        };
        // ── the turn model (a 1:50 scale of the log's 43k-token conversation) ──
        // sys(400) u1(60) effort1(10) header1(2) = P1 (472); T1 = 460 (before the effort turn),
        // E1 = 471 (n-1: the end-of-prompt checkpoint, carrying effort1 + header1)
        const std::vector<int32_t> sys = rng(3000, 400);
        const std::vector<int32_t> u1 = rng(3400, 60);
        const std::vector<int32_t> effort1 = rng(5000, 10);
        const std::vector<int32_t> header1 = rng(6000, 2);
        const std::vector<int32_t> P1 = cat(cat(cat(sys, u1), effort1), header1);
        const std::vector<int32_t> a1 = rng(9000, 80);        // the private reply (thinking + answer)
        const std::vector<int32_t> live1 = cat(P1, a1);       // 552
        // P2 = sys u1 a1r(40) u2(50) effort2(10) header2(2) = 562; T2 = 550, E2 = 561. The
        // client strips the thinking and the effort turn; u2's environment_details timestamp
        // sits at P2[520].
        const std::vector<int32_t> a1r = rng(3100, 40);
        const std::vector<int32_t> u2 = rng(3200, 50);
        const std::vector<int32_t> effort2 = rng(5010, 10);
        const std::vector<int32_t> header2 = rng(6002, 2);
        const std::vector<int32_t> P2 = cat(cat(cat(cat(cat(sys, u1), a1r), u2), effort2), header2);
        const std::vector<int32_t> a2 = rng(9100, 60);
        const std::vector<int32_t> live2 = cat(P2, a2);       // 622
        // the retry after the restart: u2 RE-RENDERED with the timestamp refreshed (P3[520]
        // changes) — the divergence sits inside the last user message, past T2 (550) but before
        // T1 (460): only T1 and below are still prefixes. The clean restart keeps u2 verbatim
        // and diverges at the effort turn (550): T2 is the resume.
        std::vector<int32_t> u2r = u2; u2r[20] = 4999;
        const std::vector<int32_t> a2r = rng(3300, 35);
        const std::vector<int32_t> u3 = rng(3500, 40);
        const std::vector<int32_t> effort3 = rng(5020, 10);
        const std::vector<int32_t> header3 = rng(6004, 2);
        const std::vector<int32_t> P3_retry =
            cat(cat(cat(cat(cat(cat(cat(sys, u1), a1r), u2r), a2r), u3), effort3), header3);   // 637
        std::vector<int32_t> P3_clean = P3_retry; P3_clean[520] = u2[20];
        check(P1.size() == 472 && P2.size() == 562 && P3_retry.size() == 637, "54: the model's lengths");
        {
            auto diverge_at = [](const std::vector<int32_t>& x, const std::vector<int32_t>& y) {
                const size_t m = std::min(x.size(), y.size());
                size_t d = 0;
                while (d < m && x[d] == y[d]) ++d;
                return d;
            };
            check(diverge_at(live2, P3_retry) == 520, "54: the retry diverges at 520 (the timestamp)");
            check(diverge_at(live2, P3_clean) == 550, "54: the clean restart diverges at 550 (the effort turn)");
        }
        // the engine's chain at park 2, kinds as the cut sites set them
        ConversationCheckpoint periodic = jckpt(std::vector<int32_t>(live2.begin(), live2.begin() + 256), 70);
        ConversationCheckpoint pin = jckpt(sys, 61);
        pin.pinned = true;
        pin.kind = CheckpointKind::pin;
        ConversationCheckpoint T1 = jckpt(std::vector<int32_t>(P1.begin(), P1.begin() + 460), 62);
        T1.kind = CheckpointKind::turn;
        ConversationCheckpoint E1 = jckpt(std::vector<int32_t>(P1.begin(), P1.begin() + 471), 63);   // n-1
        E1.kind = CheckpointKind::end_of_prompt;
        ConversationCheckpoint T2 = jckpt(std::vector<int32_t>(P2.begin(), P2.begin() + 550), 64);
        T2.kind = CheckpointKind::turn;
        ConversationCheckpoint E2 = jckpt(std::vector<int32_t>(P2.begin(), P2.begin() + 561), 65);   // n-1
        E2.kind = CheckpointKind::end_of_prompt;
        SavedConversation park1 = ids_conv_image(live1, 1);
        park1.checkpoints = {periodic, pin, T1, E1};
        SavedConversation park2 = ids_conv_image(live2, 2);
        park2.checkpoints = {periodic, pin, T1, E1, T2, E2};

        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            SavedConversation pfx = ids_conv_image(sys, 9);
            pfx.checkpoints = {pin};
            check(store.put(std::move(pfx), 1, id, {}, true), "54: the pfx entry lands");
            check(store.put(std::move(park1), 2, id), "54: park 1 lands (the full write)");
            // RC1, the manifest: a full write's .bin stores the whole chain, so best() must be
            // able to promise T1 (460) — not just the deepest (E1, 471, which carries the
            // effort tail P2 never re-sends). Before the fix this fell to the pfx (400).
            {
                const auto m = store.best(P2, {}, false);
                check(m.tokens == 460 && m.serial == 2, "54: request 2: best() promises T1 (460)");
            }
            check(store.put_async(std::move(park2), 3, id), "54: park 2 queues");
            store.flush_pending();
            check(fs::exists(rdir / "conv-2.dlt"), "54: park 2 appended a journal record");
            // RC2+RC2b, the journal: the fold delivers the resume SET, and the retry's
            // divergence inside the last user message falls back to T1 (the PREVIOUS turn)
            {
                const auto mc = store.best(P3_clean, {}, false);
                check(mc.tokens == 550 && mc.serial == 2, "54: clean restart: best() promises T2 (550)");
                const auto mr = store.best(P3_retry, {}, false);
                check(mr.tokens == 460 && mr.serial == 2,
                      "54: retry with a refreshed timestamp: best() promises T1 (460)");
                auto back = store.take(mr, id);
                check(back.has_value(), "54: the retry's take restores");
                check(back->checkpoints.size() == 4, "54: the folded chain is the resume set (4)");
                check(back->checkpoints[0].ids.size() == 256 &&
                          back->checkpoints[0].kind == CheckpointKind::periodic &&
                          back->checkpoints[1].ids.size() == 460 &&
                          back->checkpoints[1].kind == CheckpointKind::turn &&
                          back->checkpoints[2].ids.size() == 550 &&
                          back->checkpoints[2].kind == CheckpointKind::turn &&
                          back->checkpoints[3].ids.size() == 561 &&
                          back->checkpoints[3].kind == CheckpointKind::end_of_prompt,
                      "54: the set is {periodic 256, T1 460, T2 550, E2 561} with kinds intact");
                check(engine_resume(*back, P3_retry) == 460,
                      "54: the engine resumes at T1 — 177 tokens re-read, not 237 from the pin");
                check(engine_resume(*back, P3_clean) == 550,
                      "54: the same restored chain serves the clean restart at T2");
            }
        }
        // the restart itself: a fresh store reloads (the .idx v6 fast path), and the same
        // promises hold — the exact E-of-the-log scenario
        {
            ConversationDiskStore fresh(rdir.string(), 1ull << 30, &id);
            check(fresh.reload_from_disk(id) >= 1, "54: the restart reloads the spill files");
            const auto mr = fresh.best(P3_retry, {}, false);
            check(mr.tokens == 460, "54: restart: best() promises T1 (460) for the timestamp retry");
            auto back = fresh.take(mr, id);
            check(back.has_value() && engine_resume(*back, P3_retry) == 460,
                  "54: restart: the engine resumes at T1");
            const auto mc = fresh.best(P3_clean, {}, false);
            check(mc.tokens == 550, "54: restart: best() promises T2 (550) for the clean next turn");
        }
        // the escape hatch: STRATA_SPILL_RESUME_SET=0 degenerates the set to today's ONE
        // deepest checkpoint — honest (best() falls to the pfx for the retry), never a mismatch
        {
            const auto edir = dir / "effort_tail_escape";
            ConversationDiskStore store(edir.string(), 1ull << 30, &id);
            SavedConversation pfx = ids_conv_image(sys, 9);
            pfx.checkpoints = {pin};
            check(store.put(std::move(pfx), 1, id, {}, true), "54: escape: the pfx entry lands");
            SavedConversation park2b = ids_conv_image(live2, 5);
            park2b.checkpoints = {periodic, pin, T1, E1, T2, E2};
            check(store.put(std::move(park2b), 2, id), "54: escape hatch: the full write lands");
            const std::vector<int32_t> a2b = rng(9400, 60);
            SavedConversation park3b = ids_conv_image(cat(live2, a2b), 6);
            ConversationCheckpoint T3 = jckpt(live2, 66);       // the deepest of the new chain
            T3.kind = CheckpointKind::turn;
            park3b.checkpoints = {periodic, pin, T1, E1, T2, E2, T3};
            jsetenv("STRATA_SPILL_RESUME_SET", "0");
            check(store.put_async(std::move(park3b), 3, id), "54: escape hatch: the append queues");
            store.flush_pending();
            junsetenv("STRATA_SPILL_RESUME_SET");
            const auto m = store.best(P3_retry, {}, false);
            check(m.tokens == 400,
                  "54: escape hatch: the retry floors at the pinned root (today's behavior)");
            auto back = store.take(m, id);
            check(back.has_value() && engine_resume(*back, P3_retry) == 400,
                  "54: escape hatch: ONE recompute from the pin — honest, never a mismatch");
        }
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    // ── 55. A layer split's journal keeps the per-image resume sets ALIGNED: the split copies
    // each checkpoint's kind to its stage parts, so the main's and the stage's selections pick
    // the same lengths, the fold lands the same chain on every image, and the engine's
    // conversation_checkpoints_merge finds them aligned (a disagreeing selection made merge
    // fail and the engine dropped the whole restored chain — the cd647d10 regression).
    {
        const auto rdir = dir / "split_resume_set";
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        auto rng = [](int32_t base, size_t n) {
            std::vector<int32_t> v;
            for (size_t i = 0; i < n; ++i) v.push_back(base + int32_t(i));
            return v;
        };
        auto cat = [](std::vector<int32_t> a, const std::vector<int32_t>& b) {
            a.insert(a.end(), b.begin(), b.end());
            return a;
        };
        // a checkpoint WITH a stage part (what the engine's checks hold under a split)
        auto chained = [&](const std::vector<int32_t>& live, size_t len, uint8_t seed, CheckpointKind k) {
            ConversationCheckpoint c = jckpt(std::vector<int32_t>(live.begin(), live.begin() + len), seed);
            c.kind = k;
            ConversationCheckpoint part;
            part.ids = c.ids;
            part.gdn = {uint8_t(seed + 1)};
            part.ple = {uint8_t(seed + 2)};
            part.tails = {uint8_t(seed + 3)};
            part.dead = {uint8_t(seed + 4)};
            part.block_pos = {uint8_t(seed + 5)};
            c.stage_parts.push_back(std::move(part));
            return c;
        };
        // the same turn model as 54, one stage
        const std::vector<int32_t> sys = rng(3000, 400);
        const std::vector<int32_t> u1 = rng(3400, 60);
        const std::vector<int32_t> effort1 = rng(5000, 10);
        const std::vector<int32_t> header1 = rng(6000, 2);
        const std::vector<int32_t> P1 = cat(cat(cat(sys, u1), effort1), header1);
        const std::vector<int32_t> a1 = rng(9000, 80);
        const std::vector<int32_t> live1 = cat(P1, a1);            // 552
        const std::vector<int32_t> a1r = rng(3100, 40);
        const std::vector<int32_t> u2 = rng(3200, 50);
        const std::vector<int32_t> effort2 = rng(5010, 10);
        const std::vector<int32_t> header2 = rng(6002, 2);
        const std::vector<int32_t> P2 = cat(cat(cat(cat(cat(sys, u1), a1r), u2), effort2), header2);
        const std::vector<int32_t> a2 = rng(9100, 60);
        const std::vector<int32_t> live2 = cat(P2, a2);            // 622
        // park 1's chain: {periodic 256, pin 400, T1 460, E1 471}; park 2 adds {T2 550, E2 561}
        auto chain_of = [&](const std::vector<int32_t>& live, bool with_t2) {
            std::vector<ConversationCheckpoint> checks = {
                chained(live, 256, 70, CheckpointKind::periodic),
                chained(live, 400, 61, CheckpointKind::pin),
                chained(live, 460, 62, CheckpointKind::turn),
                chained(live, 471, 63, CheckpointKind::end_of_prompt),
            };
            checks[1].pinned = true;
            if (with_t2) {
                checks.push_back(chained(live, 550, 64, CheckpointKind::turn));
                checks.push_back(chained(live, 561, 65, CheckpointKind::end_of_prompt));
            }
            return checks;
        };
        // the engine's park: split the chain apart, main gets stage0, the stage image parts[0]
        auto split_park = [&](const std::vector<int32_t>& live, uint8_t seed, bool with_t2) {
            strata::core::ConversationCheckpointSplit cs =
                strata::core::conversation_checkpoints_split(chain_of(live, with_t2), 1);
            SavedConversation main = ids_conv_image(live, seed);
            main.checkpoints = std::move(cs.stage0);
            SavedConversation stage = ids_conv_image(live, uint8_t(seed + 8));
            stage.layer_lo = 40; stage.layer_hi = 48;   // the other card's carve
            stage.live = main.live;
            stage.checkpoints = std::move(cs.parts[0]);
            main.stage_images.push_back(std::move(stage));
            return main;
        };
        // the next prompt re-renders without the effort turn: it shares [0, 550) of live2
        // (declared at this scope: the restart blocks below ask best() the same question)
        const std::vector<int32_t> a2r = rng(3300, 35);
        const std::vector<int32_t> u3 = rng(3500, 40);
        const std::vector<int32_t> effort3 = rng(5020, 10);
        const std::vector<int32_t> header3 = rng(6004, 2);
        const std::vector<int32_t> P3 =
            cat(cat(cat(cat(cat(cat(cat(sys, u1), a1r), u2), a2r), u3), effort3), header3);
        // the selections agree before anything is written (the split copies the kind)
        {
            strata::core::ConversationCheckpointSplit cs =
                strata::core::conversation_checkpoints_split(chain_of(live2, true), 1);
            std::vector<const ConversationCheckpoint*> sel_main, sel_stage;
            spill_resume_checkpoints(cs.stage0, sel_main);
            spill_resume_checkpoints(cs.parts[0], sel_stage);
            check(sel_main.size() == sel_stage.size() && sel_main.size() == 4,
                  "55: the main's and the stage's sets agree (4: periodic, T1, T2, E2)");
            bool same = true;
            for (size_t i = 0; same && i < sel_main.size(); ++i)
                same = sel_main[i]->ids.size() == sel_stage[i]->ids.size() &&
                       sel_main[i]->kind == sel_stage[i]->kind;
            check(same, "55: the two sets are the same lengths and kinds");
        }
        // through the store: full write → one delta append → take → the ENGINE's merge
        {
            ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
            // the pinned root FIRST: without it find_delta_base has no pin_len and the
            // re-rendered park 2 (whose cut is neither the base's end nor its checkpoint)
            // cannot chain — a full write would land instead of the journal record
            SavedConversation pfx = ids_conv_image(sys, 9);
            ConversationCheckpoint pin55 = jckpt(sys, 61);
            pin55.pinned = true;
            pin55.kind = CheckpointKind::pin;
            pfx.checkpoints = {pin55};
            check(store.put(std::move(pfx), 1, id, {}, true), "55: the pfx entry lands");
            SavedConversation park1 = split_park(live1, 1, false);
            check(store.put(std::move(park1), 2, id), "55: park 1 (main + stage) lands");
            SavedConversation park2 = split_park(live2, 2, true);
            check(store.put_async(std::move(park2), 3, id), "55: park 2 queues");
            store.flush_pending();
            check(fs::exists(rdir / "conv-2.dlt"), "55: park 2 appended a journal record");
            const auto m = store.best(P3, {}, false);
            check(m.tokens == 550, "55: best() promises T2 (550) through the split journal");
            auto back = store.take(m, id);
            check(back.has_value() && back->stage_images.size() == 1, "55: the take restores both images");
            check(back->checkpoints.size() == 4 && back->stage_images[0].checkpoints.size() == 4,
                  "55: both folded chains carry the same 4-checkpoint set");
            bool aligned = true;
            for (size_t i = 0; aligned && i < 4; ++i)
                aligned = back->checkpoints[i].ids == back->stage_images[0].checkpoints[i].ids;
            check(aligned, "55: the two sets are index-aligned (merge's requirement)");
            // the engine's own reassembly (generate.cpp after a restore)
            strata::core::ConversationCheckpointSplit cs;
            cs.stage0 = std::move(back->checkpoints);
            cs.parts.push_back(std::move(back->stage_images[0].checkpoints));
            std::vector<ConversationCheckpoint> merged;
            check(strata::core::conversation_checkpoints_merge(std::move(cs), merged) && merged.size() == 4,
                  "55: conversation_checkpoints_merge succeeds — the restored chain survives");
            check(merged[1].kind == CheckpointKind::turn && merged[2].kind == CheckpointKind::turn &&
                      merged[3].kind == CheckpointKind::end_of_prompt && merged[1].stage_parts.size() == 1,
                  "55: the merged chain keeps the kinds and the stage parts");
        }
        // ── 56. The RESTART the whole resume-set work exists for: a FRESH store on the same
        // dir reloads the journal entry and best() still promises the turn — first through the
        // .idx fast path (the append refreshed it), then through the SLOW path (the .idx gone:
        // parse the .bin + fold the .dlt + the .rst — the self-healing a version bump or a
        // lost .idx takes). The 2026-11-02 Kaggle log: the post-restart restore fell back to
        // the pfx because the capture had lost every kind and the manifest promised only
        // ckpt_lens=[n-1]; the slow path must arrive at the SAME set the fast path carries,
        // or an .idx bump silently costs the conversation its resume points.
        {
            ConversationDiskStore fresh(rdir.string(), 1ull << 30, &id);
            const size_t loaded = fresh.reload_from_disk(id);
            check(loaded == 2 && fresh.size() == 2,
                  "56: fast path: the restart reloads the pfx and the journal entry");
            const auto m = fresh.best(P3, {}, false);
            check(m.tokens == 550, "56: fast path: best() still promises T2 (550) after a restart");
            auto back = fresh.take(m, id);
            check(back.has_value() && back->checkpoints.size() == 4 &&
                      back->checkpoints[2].kind == CheckpointKind::turn,
                  "56: fast path: the restored chain is the resume set with its kinds");
        }
        {
            fs::remove(rdir / "conv-2.idx");   // a lost/stale/bumped .idx: the slow path's door
            ConversationDiskStore fresh(rdir.string(), 1ull << 30, &id);
            const size_t loaded = fresh.reload_from_disk(id);
            check(loaded == 2 && fresh.size() == 2,
                  "56: slow path: the .bin is parsed and the journal folded back in");
            const auto m = fresh.best(P3, {}, false);
            check(m.tokens == 550, "56: slow path: best() still promises T2 (550) after a restart");
            auto back = fresh.take(m, id);
            check(back.has_value() && back->stage_images.size() == 1 &&
                      back->checkpoints.size() == 4 &&
                      back->checkpoints[2].kind == CheckpointKind::turn &&
                      back->stage_images[0].checkpoints.size() == 4,
                  "56: slow path: the fold lands the set with kinds on both images");
            bool aligned = true;
            for (size_t i = 0; aligned && i < back->checkpoints.size(); ++i)
                aligned = back->checkpoints[i].ids == back->stage_images[0].checkpoints[i].ids;
            check(aligned, "56: slow path: the two chains stay index-aligned (merge's requirement)");
        }
        junsetenv("STRATA_SPILL_JOURNAL");
    }

    fs::remove_all(dir);
    std::printf("conversation_disk_store_test: %d checks passed\n", checks);
    return 0;
}
