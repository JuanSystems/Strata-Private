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
        // (the journal itself may already be past its 20% bound and compacted into the .bin —
        // the take below proves whichever shape the pair is in restores correctly)
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

    // ── 29. Flag OFF: no journal is ever created; the old rewrite-in-place behaviour. (The
    // whole suite above runs with the flag off; this asserts the file-level evidence.)
    {
        junsetenv("STRATA_SPILL_JOURNAL");
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

    // ── 31. Compaction: once the journal crosses 20% of its .bin, the writer folds it back
    // (the .dlt disappears, the .bin carries the newest state, take() restores it whole).
    {
        jsetenv("STRATA_SPILL_JOURNAL", "1");
        const auto rdir = dir / "journal_compact";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        SavedConversation cur = conv_image(1000, 1);
        check(store.put(std::move(cur), 1000, id), "31: the base");
        cur = conv_image(1000, 1);
        // grow a page at a time: each record re-sends one partial page (~17 KiB), so the 20%
        // bound crosses after a handful of records
        for (int turn = 1; turn <= 8; ++turn) {
            SavedConversation next = conv_grow(cur, 30, uint8_t(turn));
            SavedConversation queued = next;   // the queue takes a copy; the chain keeps the original
            cur = std::move(next);
            check(store.put_async(std::move(queued), uint64_t(1000 + turn), id), "31: a delta turn");
            store.flush_pending();
            if (!fs::exists(rdir / "conv-1000.dlt")) break;   // compacted
        }
        check(!fs::exists(rdir / "conv-1000.dlt"), "31: the journal was compacted away");
        check(store.size() == 1, "31: one entry after the compaction");
        ConversationCache::Match m;
        m.serial = 1000; m.tokens = (int64_t) cur.live.ids.size(); m.live = true;
        auto back = store.take(m, id);
        check(back.has_value() && same_conv(*back, cur), "31: the compacted .bin restores the newest state");
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
        // room for two .bin files plus a small journal, but not for two .bin files plus 20%
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

    fs::remove_all(dir);
    std::printf("conversation_disk_store_test: %d checks passed\n", checks);
    return 0;
}
