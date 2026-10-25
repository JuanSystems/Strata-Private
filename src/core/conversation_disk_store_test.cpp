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

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};

    // ── 1. Round trip: put → best → take returns the same image, file is gone after take ──────
    {
        ConversationDiskStore store(dir.string() + "/rt", 1ull << 30, &id);
        check(store.enabled(), "store enabled with non-zero budget");
        SavedConversation original = sample(1000);
        // best() uses conversation_prefix() which requires the prompt's image list to match the
        // entry's live.imgs (one image at start=3 here). Pass the same image list so the prefix matches.
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        const uint64_t serial = 42;
        check(store.put(std::move(original), serial, id), "put succeeds");
        check(store.size() == 1, "manifest holds one entry");
        // The manifest's prefix matches the original's live ids (1000 tokens).
        std::vector<int32_t> prompt;
        for (int i = 0; i < 1000; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more: the next turn's token
        const auto m = store.best(prompt, imgs, /*cvec=*/false);
        check(m.tokens == 1000, "best() matches the live prefix (1000 tokens)");
        check(m.serial == serial, "best() returns the correct serial");
        auto back = store.take(m, id);
        check(back.has_value(), "take returns the image");
        check(same(sample(1000), *back), "round trip is identical");
        check(store.size() == 0, "take removes the entry");
        check(store.bytes() == 0, "take accounts bytes");
        check(!fs::exists(dir / "rt" / ("conv-" + std::to_string(serial) + ".bin")), "file deleted after take");
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

    fs::remove_all(dir);
    std::printf("conversation_disk_store_test: %d checks passed\n", checks);
    return 0;
}
