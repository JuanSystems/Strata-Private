// CPU-only tests for the delta journal codec (include/strata/core/conversation_delta.hpp).
// Same style as conversation_disk_store_test.cpp: one main(), anonymous-namespace builders, a
// check() that exits on failure, a scratch dir under fs::temp_directory_path().
//
// The builders model what a real capture produces: cells are the token count rounded UP to a
// page, every K/V part is exactly per-cell bytes wide, and a "grown" image keeps the base's
// bytes below the completed-page boundary of the kept prefix (conversation_kv_save's reuse
// rule) while everything above it is fresh data. apply(base + records) must then reproduce
// the grown image byte-exactly.
#include "strata/core/conversation_delta.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

constexpr int64_t kPage = 64;      // the K/V page size of the test images
constexpr int64_t kHeads = 2, kHeadDim = 64, kIdxDim = 32, kIdxBlock = 4;

int64_t cells_of(size_t tokens) { return ((int64_t) tokens + kPage - 1) / kPage * kPage; }

ConversationBuffer fill(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed * 7u);
        return true;
    });
    return b;
}

ConversationCheckpoint ckpt_of(const std::vector<int32_t>& ids, uint8_t seed) {
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

// A capture-consistent image: 2 QSA layers with scales and a pooled indexer + a draft layer
// without one, every part exactly per-cell bytes wide.
SavedConversation image_of(size_t tokens, uint8_t seed) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(200 + i);
    s.layer_lo = 0; s.layer_hi = 40;
    s.cvec = false;
    std::vector<int32_t> ids;
    for (size_t i = 0; i < tokens; ++i) ids.push_back(int32_t(3000 + i));
    s.live = ckpt_of(ids, seed);
    s.live.imgs = {{7, 0xabcd0000ull + seed}};
    s.checkpoints.push_back(ckpt_of(std::vector<int32_t>(ids.begin(), ids.begin() + tokens / 2),
                                    uint8_t(seed + 50)));
    const int64_t cells = cells_of(tokens);
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 1; kv.cells = cells; kv.heads = kHeads; kv.head_dim = kHeadDim;
        kv.page_size = kPage;
        kv.pooled_rows = layer < 2 && tokens > 0 ? (int64_t) tokens / kIdxBlock + 1 : 0;
        kv.idx_dim = kIdxDim;
        kv.k = fill(size_t(cells) * size_t(kHeads * kHeadDim * 2), uint8_t(layer * 10 + seed));
        kv.v = fill(size_t(cells) * size_t(kHeads * kHeadDim * 2), uint8_t(layer * 10 + seed + 1));
        kv.k_scale = layer < 2 ? fill(size_t(cells) * size_t(kHeads * 2), uint8_t(layer * 10 + seed + 2))
                               : ConversationBuffer{};
        kv.v_scale = layer < 2 ? fill(size_t(cells) * size_t(kHeads * 2), uint8_t(layer * 10 + seed + 3))
                               : ConversationBuffer{};
        kv.pooled = layer < 2 && tokens > 0
                        ? fill(size_t(kv.pooled_rows) * size_t(kIdxDim * 4), uint8_t(layer * 10 + seed + 4))
                        : ConversationBuffer{};
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// The next turn of `base`: `extra` more tokens, fresh bytes from the completed-page boundary
// of base's length on (what a capture re-sends), everything below it identical.
SavedConversation grow(const SavedConversation& base, size_t extra, uint8_t seed) {
    SavedConversation s = base;
    const size_t old_tokens = s.live.ids.size();
    for (size_t i = 0; i < extra; ++i) s.live.ids.push_back(int32_t(3000 + old_tokens + i));
    s.live = ckpt_of(s.live.ids, seed);
    s.live.imgs = {{7, 0xabcd0000ull + seed}};
    s.checkpoints.clear();
    s.checkpoints.push_back(ckpt_of(
        std::vector<int32_t>(s.live.ids.begin(), s.live.ids.begin() + (std::ptrdiff_t) (s.live.ids.size() / 2)),
        uint8_t(seed + 50)));
    const int64_t whole = (int64_t) (old_tokens / kPage) * kPage;
    const int64_t cells = cells_of(s.live.ids.size());
    const int64_t pooled_keep = old_tokens / kIdxBlock;
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv& kv = s.kv[layer];
        kv.cells = cells;
        kv.pooled_rows = layer < 2 ? (int64_t) s.live.ids.size() / kIdxBlock + 1 : 0;
        auto extend = [&](ConversationBuffer& buf, size_t size, size_t at) {
            buf.resize(size);   // keeps [0, at) — the base's completed pages
            if (size > at)
                buf.visit(at, size - at, [&](uint8_t* p, size_t c, size_t) {
                    for (size_t i = 0; i < c; ++i) p[i] = uint8_t(i * 17u + seed + layer);
                    return true;
                });
        };
        extend(kv.k, size_t(cells) * size_t(kHeads * kHeadDim * 2), size_t(whole) * size_t(kHeads * kHeadDim * 2));
        extend(kv.v, size_t(cells) * size_t(kHeads * kHeadDim * 2), size_t(whole) * size_t(kHeads * kHeadDim * 2));
        if (layer < 2) {
            extend(kv.k_scale, size_t(cells) * size_t(kHeads * 2), size_t(whole) * size_t(kHeads * 2));
            extend(kv.v_scale, size_t(cells) * size_t(kHeads * 2), size_t(whole) * size_t(kHeads * 2));
            extend(kv.pooled, size_t(kv.pooled_rows) * size_t(kIdxDim * 4), pooled_keep * size_t(kIdxDim * 4));
        }
    }
    return s;
}

// The client's re-render: the new history keeps only base's deepest-checkpoint prefix and
// writes a fresh (shorter) turn on top — base's live ids are NOT a prefix of the result.
SavedConversation rerender(const SavedConversation& base, size_t new_tail, uint8_t seed) {
    const std::vector<int32_t> anchor = base.checkpoints.back().ids;
    SavedConversation s = base;
    const size_t keep = anchor.size();
    const int64_t whole = (int64_t) (keep / kPage) * kPage;
    const int64_t pooled_keep = keep / kIdxBlock;
    s.live.ids = anchor;
    for (size_t i = 0; i < new_tail; ++i) s.live.ids.push_back(int32_t(9000 + i));
    s.live = ckpt_of(s.live.ids, seed);
    s.live.imgs = {{7, 0xabcd0000ull + seed}};
    s.checkpoints.clear();
    s.checkpoints.push_back(ckpt_of(
        std::vector<int32_t>(s.live.ids.begin(), s.live.ids.begin() + (std::ptrdiff_t) (s.live.ids.size() / 2)),
        uint8_t(seed + 50)));
    const int64_t cells = cells_of(s.live.ids.size());
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv& kv = s.kv[layer];
        kv.cells = cells;
        kv.pooled_rows = layer < 2 ? (int64_t) s.live.ids.size() / kIdxBlock + 1 : 0;
        auto cut = [&](ConversationBuffer& buf, size_t size, size_t at) {
            buf.resize(at);   // keep the base's completed pages of the anchor prefix
            buf.resize(size);
            if (size > at)
                buf.visit(at, size - at, [&](uint8_t* p, size_t c, size_t) {
                    for (size_t i = 0; i < c; ++i) p[i] = uint8_t(i * 23u + seed + layer);
                    return true;
                });
        };
        cut(kv.k, size_t(cells) * size_t(kHeads * kHeadDim * 2), size_t(whole) * size_t(kHeads * kHeadDim * 2));
        cut(kv.v, size_t(cells) * size_t(kHeads * kHeadDim * 2), size_t(whole) * size_t(kHeads * kHeadDim * 2));
        if (layer < 2) {
            cut(kv.k_scale, size_t(cells) * size_t(kHeads * 2), size_t(whole) * size_t(kHeads * 2));
            cut(kv.v_scale, size_t(cells) * size_t(kHeads * 2), size_t(whole) * size_t(kHeads * 2));
            cut(kv.pooled, size_t(kv.pooled_rows) * size_t(kIdxDim * 4), pooled_keep * size_t(kIdxDim * 4));
        }
    }
    return s;
}

// A re-render that diverges at an ARBITRARY cut — neither the base's end nor its checkpoint:
// the selector's common-prefix case. Below the completed-page boundary of `cut` the base's own
// bytes (the same conversation's cells); above it, a fresh tail.
SavedConversation lcp_turn(const SavedConversation& base, size_t cut, size_t new_tail, uint8_t seed) {
    SavedConversation s = base;
    const int64_t whole = (int64_t) (cut / kPage) * kPage;
    const int64_t pooled_keep = cut / kIdxBlock;
    s.live.ids.assign(base.live.ids.begin(), base.live.ids.begin() + (std::ptrdiff_t) cut);
    for (size_t i = 0; i < new_tail; ++i) s.live.ids.push_back(int32_t(9000 + i));
    s.live = ckpt_of(s.live.ids, seed);
    s.live.imgs = {{7, 0xabcd0000ull + seed}};
    s.checkpoints.clear();
    s.checkpoints.push_back(ckpt_of(
        std::vector<int32_t>(s.live.ids.begin(), s.live.ids.begin() + (std::ptrdiff_t) (s.live.ids.size() / 2)),
        uint8_t(seed + 50)));
    const int64_t cells = cells_of(s.live.ids.size());
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv& kv = s.kv[layer];
        kv.cells = cells;
        kv.pooled_rows = layer < 2 ? (int64_t) s.live.ids.size() / kIdxBlock + 1 : 0;
        auto cut_buf = [&](ConversationBuffer& buf, size_t size, size_t at) {
            buf.resize(at);   // keep the base's completed pages of the kept prefix
            buf.resize(size);
            if (size > at)
                buf.visit(at, size - at, [&](uint8_t* p, size_t c, size_t) {
                    for (size_t i = 0; i < c; ++i) p[i] = uint8_t(i * 23u + seed + layer);
                    return true;
                });
        };
        cut_buf(kv.k, size_t(cells) * size_t(kHeads * kHeadDim * 2),
                size_t(whole) * size_t(kHeads * kHeadDim * 2));
        cut_buf(kv.v, size_t(cells) * size_t(kHeads * kHeadDim * 2),
                size_t(whole) * size_t(kHeads * kHeadDim * 2));
        if (layer < 2) {
            cut_buf(kv.k_scale, size_t(cells) * size_t(kHeads * 2), size_t(whole) * size_t(kHeads * 2));
            cut_buf(kv.v_scale, size_t(cells) * size_t(kHeads * 2), size_t(whole) * size_t(kHeads * 2));
            cut_buf(kv.pooled, size_t(kv.pooled_rows) * size_t(kIdxDim * 4),
                    size_t(pooled_keep) * size_t(kIdxDim * 4));
        }
    }
    return s;
}

bool same_image(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi ||
        a.cvec != b.cvec || a.live.ids != b.live.ids || a.live.imgs != b.live.imgs ||
        a.live.gdn != b.live.gdn || a.live.ple != b.live.ple || a.live.tails != b.live.tails ||
        a.live.dead != b.live.dead || a.live.block_pos != b.live.block_pos ||
        a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i) {
        const auto& x = a.checkpoints[i]; const auto& y = b.checkpoints[i];
        if (x.ids != y.ids || x.imgs != y.imgs || x.gdn != y.gdn || x.ple != y.ple ||
            x.tails != y.tails || x.dead != y.dead || x.block_pos != y.block_pos || x.used != y.used)
            return false;
    }
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads ||
            x.head_dim != y.head_dim || x.page_size != y.page_size ||
            x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) ||
            !(x.v_scale == y.v_scale) || !(x.pooled == y.pooled))
            return false;
    }
    return true;
}

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-delta-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const std::string journal = (dir / "conv-1.dlt").string();

    // ── 1. Growth: pack → append → apply reproduces the grown image byte-exactly ───────────
    {
        SavedConversation base = image_of(1000, 1);
        SavedConversation next = grow(base, 300, 2);
        DeltaPlan plan;
        check(conversation_delta_plan(plan, next, 1000, 1000), "1: plan for a grown turn");
        check(plan.applies_to_len == 1000 && plan.keep_tokens == 1000 && plan.new_tokens == 1300,
              "1: plan's chain fields");
        // the boundary rule: at = per-cell bytes × the completed pages of the kept prefix
        const size_t per_k = size_t(kHeads * kHeadDim * 2);
        check(plan.images[0].parts[0][0].at == 960 * per_k, "1: k's at is the completed-page boundary");
        check(plan.images[0].parts[0][1].at == 960 * per_k, "1: v's at is the completed-page boundary");
        // pooled: idx_block derived as an upper bound from the new image (1300/(326-1)=4)
        check(plan.images[0].parts[0][4].at == (1000 / 4) * size_t(kIdxDim * 4), "1: pooled's at is completed rows");
        // the draft layer has no pooled part at all
        check(plan.images[0].parts[2][4].at == 0 && plan.images[0].parts[2][4].n == 0, "1: the draft has no pooled part");
        uint64_t rec = 0;
        check(conversation_delta_append(journal, next, plan, rec), "1: append lands");
        check(rec == plan.record_bytes, "1: the record's size is the plan's");
        check(fs::file_size(journal) == rec, "1: the journal holds exactly one record");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 1 && !fold.torn && !fold.dropped, "1: one record applied, nothing torn");
        check(fold.journal_bytes == rec, "1: the fold's byte count matches");
        check(same_image(back, next), "1: base + record == the grown image, byte-exactly");
    }

    // ── 2. Chained growth: three turns, one journal; the fold applies them in order ────────
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 40, 2);
        SavedConversation t2 = grow(t1, 60, 3);
        SavedConversation t3 = grow(t2, 200, 4);
        uint64_t rec1 = 0, rec2 = 0, rec3 = 0;
        DeltaPlan p1, p2, p3;
        check(conversation_delta_plan(p1, t1, 1000, 1000) && conversation_delta_append(journal, t1, p1, rec1),
              "2: record 1 lands");
        check(conversation_delta_plan(p2, t2, t1.live.ids.size(), t1.live.ids.size()) &&
              conversation_delta_append(journal, t2, p2, rec2), "2: record 2 lands");
        check(conversation_delta_plan(p3, t3, t2.live.ids.size(), t2.live.ids.size()) &&
              conversation_delta_append(journal, t3, p3, rec3), "2: record 3 lands");
        check(fs::file_size(journal) == rec1 + rec2 + rec3, "2: the journal holds exactly three records");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 3 && !fold.torn, "2: three records applied");
        check(fold.records == 3 && fold.journal_bytes == rec1 + rec2 + rec3, "2: the fold's accounting");
                check(same_image(back, t3), "2: base + 3 records == the third turn's image");
    }

    // ── 3. Re-render: keep = the deepest checkpoint's cut, the branch above it is fresh ────
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        SavedConversation next = rerender(base, 400, 5);
        check(next.live.ids.size() == 500 + 400, "3: the re-rendered turn is anchored at the checkpoint");
        DeltaPlan plan;
        check(conversation_delta_plan(plan, next, 1000, 500), "3: plan for a re-rendered turn");
        check(plan.keep_tokens == 500 && plan.applies_to_len == 1000, "3: the anchor is the ckpt cut");
        uint64_t rec = 0;
        check(conversation_delta_append(journal, next, plan, rec), "3: append lands");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 1, "3: one record applied");
        check(same_image(back, next), "3: base + record == the re-rendered image");
    }

    // ── 4. Torn tail: garbage past the last good record truncates the fold there ───────────
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 40, 2);
        SavedConversation t2 = grow(t1, 60, 3);
        uint64_t rec1 = 0, rec2 = 0;
        DeltaPlan p1, p2;
        check(conversation_delta_plan(p1, t1, 1000, 1000) && conversation_delta_append(journal, t1, p1, rec1),
              "4: record 1 lands");
        check(conversation_delta_plan(p2, t2, t1.live.ids.size(), t1.live.ids.size()) &&
              conversation_delta_append(journal, t2, p2, rec2), "4: record 2 lands");
        // a torn third record: half a header is enough to look like a record that never landed
        {
            std::FILE* f = std::fopen(journal.c_str(), "ab");
            const char half[64] = "SDLT\x01\x00\x00\x00";   // magic + version, nothing else
            std::fwrite(half, 1, sizeof(half), f);
            std::fclose(f);
        }
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 2 && fold.torn, "4: the fold stops at the last good record");
        check(fold.valid_end == rec1 + rec2, "4: the valid end is the last good record's end");
        check(same_image(back, t2), "4: the image is the second turn's");
        // after the caller truncates, the fold is clean
        std::error_code ec;
        fs::resize_file(journal, fold.valid_end, ec);
        check(!ec, "4: truncate works");
        SavedConversation again = base;
        const DeltaFold fold2 = conversation_delta_apply(journal, again);
        check(fold2.applied == 2 && !fold2.torn, "4: after the heal, nothing is torn");
        check(same_image(again, t2), "4: the healed journal yields the same state");
    }

    // ── 5. A journal that does not chain onto the base is dropped whole ────────────────────
    // (the compaction-crash case: the .bin was already folded past the journal's anchor)
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 40, 2);
        uint64_t rec1 = 0;
        DeltaPlan p1;
        check(conversation_delta_plan(p1, t1, 1000, 1000) && conversation_delta_append(journal, t1, p1, rec1),
              "5: record lands against the 1000-token base");
        // a DIFFERENT base (the compacted one, 1040 tokens): the record does not chain
        SavedConversation compacted = t1;
        const DeltaFold fold = conversation_delta_apply(journal, compacted);
        check(fold.applied == 0 && fold.dropped, "5: nothing chains — the journal is stale");
        check(same_image(compacted, t1), "5: the image is untouched");
    }

    // ── 6. A corrupted record (the hash fails) truncates the fold at the last good one ────
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 40, 2);
        SavedConversation t2 = grow(t1, 60, 3);
        uint64_t rec1 = 0, rec2 = 0;
        DeltaPlan p1, p2;
        check(conversation_delta_plan(p1, t1, 1000, 1000) && conversation_delta_append(journal, t1, p1, rec1),
              "6: record 1 lands");
        check(conversation_delta_plan(p2, t2, t1.live.ids.size(), t1.live.ids.size()) &&
              conversation_delta_append(journal, t2, p2, rec2), "6: record 2 lands");
        // flip one byte inside record 2's payload region (past its header)
        {
            std::FILE* f = std::fopen(journal.c_str(), "r+b");
            std::fseek(f, (long) rec1 + 200, SEEK_SET);
            const int before = std::fgetc(f);
            std::fseek(f, (long) rec1 + 200, SEEK_SET);   // rewrite at the SAME position
            std::fputc(before ^ 0xff, f);
            std::fclose(f);
        }
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 1 && fold.torn, "6: the corrupted record is not applied");
        check(same_image(back, t1), "6: the image is the last good record's state");
    }

    // ── 7. The plan refuses what cannot be a delta ────────────────────────────────────────
    {
        SavedConversation base = image_of(100, 1);
        SavedConversation next = grow(base, 900, 2);   // the base carries a tenth of the new
        DeltaPlan plan;
        check(!conversation_delta_plan(plan, next, 100, 100),
              "7: a base below half the new image is refused");
        SavedConversation misaligned = image_of(1000, 1);
        misaligned.kv[0].k.resize(misaligned.kv[0].k.size() + 1);   // not cell-aligned any more
        check(!conversation_delta_plan(plan, misaligned, 1000, 1000),
              "7: a buffer that is not cell-aligned is refused");
        SavedConversation geom = image_of(1000, 1);
        geom.geometry[3] ^= 1;   // a different geometry echo
        check(conversation_delta_plan(plan, geom, 1000, 1000),
              "7: the plan itself only checks the image's self-consistency");
        // a keep past the chaining state is refused
        check(!conversation_delta_plan(plan, grow(image_of(1000, 1), 40, 2), 500, 900),
              "7: keep_tokens beyond the chaining state is refused");
    }

    // ── 8. An empty or absent journal folds to nothing, cleanly ───────────────────────────
    {
        SavedConversation base = image_of(1000, 1);
        const std::string nothing = (dir / "conv-none.dlt").string();
        const DeltaFold fold = conversation_delta_apply(nothing, base);
        check(!fold.had_journal && fold.applied == 0 && !fold.torn, "8: no journal, no fold");
        const std::string empty = (dir / "conv-empty.dlt").string();
        std::FILE* f = std::fopen(empty.c_str(), "wb");
        std::fclose(f);
        const DeltaFold fold2 = conversation_delta_apply(empty, base);
        check(fold2.had_journal && fold2.applied == 0 && !fold2.torn && !fold2.dropped,
              "8: an empty journal folds to nothing");
    }

    // ── 9. Layer split (the 2-GPU shape): a record carries one section per image, and the
    // fold reproduces the main AND the stage byte-exactly ────────────────────────────────
    {
        fs::remove(journal);
        // a 2-image spill: the main holds layers [0, 40), the stage [40, 48) — different
        // carve, different bytes, same live state (one capture, two devices)
        SavedConversation base = image_of(1000, 1);
        base.layer_lo = 0; base.layer_hi = 40;
        {
            SavedConversation stage = image_of(1000, 9);
            stage.layer_lo = 40; stage.layer_hi = 48;   // the other card's carve
            stage.live = base.live;                      // the capture's one live state
            stage.checkpoints = base.checkpoints;
            base.stage_images.push_back(std::move(stage));
        }
        SavedConversation next = grow(base, 300, 2);
        next.stage_images.clear();
        {
            SavedConversation nstage = grow(base.stage_images[0], 300, 3);
            nstage.layer_lo = 40; nstage.layer_hi = 48;
            nstage.live = next.live;
            nstage.checkpoints = next.checkpoints;
            next.stage_images.push_back(std::move(nstage));
        }
        DeltaPlan plan;
        check(conversation_delta_plan(plan, next, 1000, 1000), "9: plan for a 2-image turn");
        check(plan.images.size() == 2, "9: the plan covers main + stage");
        check(plan.images[1].geometry_hash != plan.images[0].geometry_hash,
              "9: each section carries its own carve's echo");
        uint64_t rec = 0;
        check(conversation_delta_append(journal, next, plan, rec), "9: append lands");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 1 && !fold.torn, "9: one record applied over both images");
        check(same_image(back, next), "9: main + record == the grown main");
        check(back.stage_images.size() == 1 && same_image(back.stage_images[0], next.stage_images[0]),
              "9: stage + record == the grown stage");
    }

    // ── 10. Torn tail with stages: the fold stops at the last good record for BOTH images ──
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        {
            SavedConversation stage = image_of(1000, 9);
            stage.layer_lo = 40; stage.layer_hi = 48;
            stage.live = base.live; stage.checkpoints = base.checkpoints;
            base.stage_images.push_back(std::move(stage));
        }
        SavedConversation t1 = grow(base, 40, 2);
        t1.stage_images.clear();
        {
            SavedConversation nstage = grow(base.stage_images[0], 40, 3);
            nstage.layer_lo = 40; nstage.layer_hi = 48;
            nstage.live = t1.live; nstage.checkpoints = t1.checkpoints;
            t1.stage_images.push_back(std::move(nstage));
        }
        uint64_t rec1 = 0;
        DeltaPlan p1;
        check(conversation_delta_plan(p1, t1, 1000, 1000) && conversation_delta_append(journal, t1, p1, rec1),
              "10: record 1 lands over both images");
        // garbage past it
        {
            std::FILE* f = std::fopen(journal.c_str(), "ab");
            const char junk[64] = "SDLT";
            std::fwrite(junk, 1, sizeof(junk), f);
            std::fclose(f);
        }
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 1 && fold.torn && fold.valid_end == rec1,
              "10: the fold stops at the last good record");
        check(same_image(back, t1) && same_image(back.stage_images[0], t1.stage_images[0]),
              "10: both images hold the first turn's state");
    }

    // ── 11. An image-count mismatch is a torn tail: a 2-image journal cannot apply onto the
    // main alone (a stage file lost) — nothing moves, the base restores by itself ────────
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        {
            SavedConversation stage = image_of(1000, 9);
            stage.layer_lo = 40; stage.layer_hi = 48;
            stage.live = base.live; stage.checkpoints = base.checkpoints;
            base.stage_images.push_back(std::move(stage));
        }
        SavedConversation next = grow(base, 40, 2);
        next.stage_images.clear();
        {
            SavedConversation nstage = grow(base.stage_images[0], 40, 3);
            nstage.layer_lo = 40; nstage.layer_hi = 48;
            nstage.live = next.live; nstage.checkpoints = next.checkpoints;
            next.stage_images.push_back(std::move(nstage));
        }
        uint64_t rec = 0;
        DeltaPlan p;
        check(conversation_delta_plan(p, next, 1000, 1000) && conversation_delta_append(journal, next, p, rec),
              "11: a 2-image record lands");
        // apply onto a copy WITHOUT the stage
        SavedConversation main_only = base;
        main_only.stage_images.clear();
        const DeltaFold fold = conversation_delta_apply(journal, main_only);
        check(fold.applied == 0 && fold.torn, "11: the image count does not match — nothing applies");
        check(same_image(main_only, base), "11: the main is untouched");
        // the plan also refuses images whose live states disagree (not one capture)
        SavedConversation odd = image_of(1000, 1);
        {
            SavedConversation stage = image_of(999, 9);   // a different length: not a delta pair
            stage.layer_lo = 40; stage.layer_hi = 48;
            odd.stage_images.push_back(std::move(stage));
        }
        DeltaPlan refuse;
        check(!conversation_delta_plan(refuse, odd, 1000, 1000),
              "11: images that do not share one live state are refused");
    }

    // ── 12. The intermediate cut (the selector's new case): keep_tokens is NEITHER the base's
    // end NOR a checkpoint — 700 of a 1000-id base whose deepest checkpoint sits at 500. The
    // plan builds, the record appends, and apply reconstructs the image byte-exactly from the
    // truncated base + the re-sent tail (the partial page at the cut is re-sent, not trusted).
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        check(base.checkpoints.back().ids.size() == 500, "12: the base's checkpoint is at 500");
        SavedConversation next = lcp_turn(base, 700, 300, 5);
        check(next.live.ids.size() == 1000, "12: the re-rendered park is 1000 ids");
        check(next.live.ids.size() != base.checkpoints.back().ids.size(),
              "12: the cut is not the checkpoint");
        DeltaPlan plan;
        check(conversation_delta_plan(plan, next, 1000, 700), "12: plan for the intermediate cut");
        check(plan.keep_tokens == 700 && plan.applies_to_len == 1000 && plan.new_tokens == 1000,
              "12: the plan's chain fields");
        // the payload starts at the completed pages of the CUT (640), not of the base (960):
        // the 36 cells of the cut's partial page are re-sent, never trusted
        const size_t per_k = size_t(kHeads * kHeadDim * 2);
        check(plan.images[0].parts[0][0].at == 640 * per_k, "12: k's at is the CUT's page boundary");
        check(plan.images[0].parts[0][1].at == 640 * per_k, "12: v's at is the CUT's page boundary");
        check(plan.images[0].parts[0][4].at == (700 / kIdxBlock) * size_t(kIdxDim * 4),
              "12: pooled's at is the cut's completed rows");
        uint64_t rec = 0;
        check(conversation_delta_append(journal, next, plan, rec), "12: append lands");
        check(rec == plan.record_bytes, "12: the record's size is the plan's");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 1 && !fold.torn && !fold.dropped, "12: one record applied, nothing torn");
        check(same_image(back, next), "12: truncated base + record == the re-rendered image");
    }

    // ── 13. A capture with a long checkpoint chain: the record carries ONLY the checkpoint a
    // full write would store (the chain's deepest; with a pinned shared prefix in the chain,
    // the deepest PINNED one) — serializing the whole chain re-sent every checkpoint's full
    // running state on every turn's append, and the .dlt outgrew the .bin it extends. The
    // fold must land exactly that one checkpoint (the apply replaces the chain).
    {
        fs::remove(journal);
        SavedConversation base = image_of(1000, 1);
        SavedConversation next = grow(base, 300, 2);   // one checkpoint at 650 (the deepest)
        check(next.checkpoints.size() == 1 && next.checkpoints[0].ids.size() == 650,
              "13: the grown capture starts with its one checkpoint");
        // the real park shape: the pinned shared prefix + a periodic checkpoint deeper than it
        ConversationCheckpoint shallow = ckpt_of(
            std::vector<int32_t>(next.live.ids.begin(), next.live.ids.begin() + 200), 61);
        shallow.pinned = true;
        ConversationCheckpoint deep = ckpt_of(
            std::vector<int32_t>(next.live.ids.begin(), next.live.ids.begin() + 900), 62);
        next.checkpoints.push_back(shallow);
        next.checkpoints.push_back(deep);
        // with a pin present, the full-write selection is the deepest PINNED one (at 200)
        SavedConversation trimmed_pin = next;
        trimmed_pin.checkpoints.assign(1, shallow);
        DeltaPlan plan, plan_trimmed;
        check(conversation_delta_plan(plan, next, 1000, 1000), "13: plan for the chained capture");
        check(conversation_delta_plan(plan_trimmed, trimmed_pin, 1000, 1000),
              "13: plan for the trimmed capture");
        check(plan.record_bytes == plan_trimmed.record_bytes,
              "13: the record carries ONE checkpoint, not the chain");
        uint64_t rec = 0;
        check(conversation_delta_append(journal, next, plan, rec), "13: append lands");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back);
        check(fold.applied == 1 && !fold.torn, "13: one record applied");
        check(back.checkpoints.size() == 1 && back.checkpoints[0].ids == shallow.ids,
              "13: the folded chain is the record's one checkpoint");
        check(same_image(back, trimmed_pin), "13: fold == a full write of the park");
        // no pin in the chain: the selection is the chain's deepest (at 900)
        fs::remove(journal);
        SavedConversation nopin = next;
        nopin.checkpoints[1].pinned = false;
        SavedConversation trimmed_deep = nopin;
        trimmed_deep.checkpoints.assign(1, deep);
        DeltaPlan plan2, plan2_trimmed;
        check(conversation_delta_plan(plan2, nopin, 1000, 1000) &&
              conversation_delta_plan(plan2_trimmed, trimmed_deep, 1000, 1000), "13: no-pin plans");
        check(plan2.record_bytes == plan2_trimmed.record_bytes,
              "13: without a pin the record carries the deepest checkpoint alone");
        uint64_t rec2 = 0;
        check(conversation_delta_append(journal, nopin, plan2, rec2), "13: no-pin append lands");
        SavedConversation back2 = base;
        const DeltaFold fold2 = conversation_delta_apply(journal, back2);
        check(fold2.applied == 1 && !fold2.torn, "13: no-pin fold");
        check(back2.checkpoints.size() == 1 && back2.checkpoints[0].ids == deep.ids,
              "13: the folded chain is the deepest checkpoint");
        check(same_image(back2, trimmed_deep), "13: no-pin fold == a full write of the park");
    }

    // ── 14. v3 round-trip, multi-turn: the records carry NO state (the sidecar does), so the
    // fold must land the state of the LAST turn's image byte-exactly — four turns of growth
    // and re-render, each append publishing its .rst, one apply at the end ─────────────────
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 200, 2);
        SavedConversation t2 = grow(t1, 150, 3);
        SavedConversation t3 = rerender(t2, 120, 4);          // keep < base_len (the ckpt cut)
        SavedConversation t4 = grow(t3, 260, 5);
        const SavedConversation* chain[4] = {&t1, &t2, &t3, &t4};
        uint64_t total = 0;
        for (size_t turn = 0; turn < 4; ++turn) {
            const uint64_t prev = turn == 0 ? 1000 : chain[turn - 1]->live.ids.size();
            const uint64_t keep = turn == 2
                ? (uint64_t) t2.checkpoints.back().ids.size()   // the re-render's anchor
                : prev;
            DeltaPlan plan;
            check(conversation_delta_plan(plan, *chain[turn], prev, keep, turn + 1),
                  "14: v3 plan lands");
            check(plan.version == kDeltaVersion && plan.seq == turn + 1, "14: the plan is v3");
            uint64_t rec = 0, hash = 0;
            check(conversation_delta_append(journal, *chain[turn], plan, rec, &hash),
                  "14: v3 append lands");
            uint64_t rst_bytes = 0;
            check(conversation_delta_state_write(rst, *chain[turn], turn + 1, hash, rst_bytes),
                  "14: the state sidecar lands");
            check(rst_bytes == conversation_delta_state_bytes(*chain[turn]),
                  "14: the sidecar's size is the estimate's");
            // the size bound: the record is the K/V it sends + a fixed slack, NOT O(state)
            uint64_t kv_sent = 0;
            for (const auto& layer : plan.images[0].parts)
                for (const auto& part : layer) kv_sent += part.n;
            check(plan.record_bytes <= kv_sent + kv_sent / 20 + 64 * 1024,
                  "14: the record is the K/V + 5% + 64 KiB, nothing else");
            total += rec;
        }
        // the record's size is INDEPENDENT of the live state: the same turn with a 64 KiB
        // state array plans the same record bytes as with a 2 B one (the state is in the
        // sidecar, and only there)
        {
            SavedConversation big = t4, small = t4;
            big.live.gdn.assign(64 * 1024, 0xab);
            small.live.gdn.assign(2, 0xcd);
            DeltaPlan pb, ps;
            check(conversation_delta_plan(pb, big, t3.live.ids.size(), t3.live.ids.size(), 5) &&
                  conversation_delta_plan(ps, small, t3.live.ids.size(), t3.live.ids.size(), 5),
                  "14: both state shapes plan");
            check(pb.record_bytes == ps.record_bytes,
                  "14: the record size does not grow with the state (it is in the sidecar)");
            check(conversation_delta_state_bytes(big) > conversation_delta_state_bytes(small),
                  "14: the SIDECAR grows with the state (that is where it lives)");
        }
        check(fs::file_size(journal) == total, "14: the journal holds exactly four records");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back, rst);
        check(fold.applied == 4 && !fold.torn && !fold.dropped && !fold.failed,
              "14: four v3 records applied cleanly");
        check(same_image(back, t4), "14: base + 4 light records + sidecar == turn 4, byte-exactly");
        // the sidecar holds turn 4's state: same image regardless of how many records precede
        check(conversation_delta_state_bytes(t4) == conversation_delta_state_bytes(t4),
              "14: the sidecar size is a function of the image alone");
    }

    // ── 15. v3 layer-split round-trip: one record per turn covers BOTH images, the sidecar
    // carries both states, the fold reproduces main AND stage byte-exactly ─────────────────
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        base.layer_lo = 0; base.layer_hi = 40;
        {
            SavedConversation stage = image_of(1000, 9);
            stage.layer_lo = 40; stage.layer_hi = 48;
            stage.live = base.live; stage.checkpoints = base.checkpoints;
            base.stage_images.push_back(std::move(stage));
        }
        SavedConversation t1 = grow(base, 120, 2);
        t1.stage_images.clear();
        {
            SavedConversation nstage = grow(base.stage_images[0], 120, 3);
            nstage.layer_lo = 40; nstage.layer_hi = 48;
            nstage.live = t1.live; nstage.checkpoints = t1.checkpoints;
            t1.stage_images.push_back(std::move(nstage));
        }
        SavedConversation t2 = grow(t1, 90, 4);
        t2.stage_images.clear();
        {
            SavedConversation nstage = grow(t1.stage_images[0], 90, 5);
            nstage.layer_lo = 40; nstage.layer_hi = 48;
            nstage.live = t2.live; nstage.checkpoints = t2.checkpoints;
            t2.stage_images.push_back(std::move(nstage));
        }
        uint64_t rec1 = 0, hash1 = 0, rec2 = 0, hash2 = 0;
        DeltaPlan p1, p2;
        {
            uint64_t rst_bytes = 0;
            check(conversation_delta_plan(p1, t1, 1000, 1000, 1), "15: v3 plan 1 (2 images)");
            check(conversation_delta_append(journal, t1, p1, rec1, &hash1), "15: v3 append 1");
            check(conversation_delta_state_write(rst, t1, 1, hash1, rst_bytes), "15: sidecar 1");
        }
        {
            uint64_t rst_bytes = 0;
            check(conversation_delta_plan(p2, t2, t1.live.ids.size(), t1.live.ids.size(), 2),
                  "15: v3 plan 2");
            check(conversation_delta_append(journal, t2, p2, rec2, &hash2), "15: v3 append 2");
            check(conversation_delta_state_write(rst, t2, 2, hash2, rst_bytes), "15: sidecar 2");
        }
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back, rst);
        check(fold.applied == 2 && !fold.torn && !fold.failed, "15: two v3 records applied");
        check(same_image(back, t2), "15: the main reproduces turn 2 byte-exactly");
        check(back.stage_images.size() == 1 && same_image(back.stage_images[0], t2.stage_images[0]),
              "15: the stage reproduces turn 2 byte-exactly");
    }

    // ── 16. Mixed journal: two v2 records (seq=0) then two v3 — one file, one fold, the
    // state of the last applied record wins (records 1-2 carry theirs, 3-4 take the sidecar's)
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 100, 2);
        SavedConversation t2 = grow(t1, 110, 3);
        SavedConversation t3 = grow(t2, 120, 4);
        SavedConversation t4 = grow(t3, 130, 5);
        uint64_t rec = 0, hash = 0;
        DeltaPlan p;
        // v2 records first
        check(conversation_delta_plan(p, t1, 1000, 1000, 0) && p.version == kDeltaVersionV2,
              "16: plan 1 is v2");
        check(conversation_delta_append(journal, t1, p, rec), "16: v2 record 1 lands");
        check(conversation_delta_plan(p, t2, t1.live.ids.size(), t1.live.ids.size(), 0),
              "16: plan 2 is v2");
        check(conversation_delta_append(journal, t2, p, rec), "16: v2 record 2 lands");
        // then v3 records (seq 3 and 4 — their 1-based index within the journal)
        check(conversation_delta_plan(p, t3, t2.live.ids.size(), t2.live.ids.size(), 3),
              "16: plan 3 is v3");
        check(conversation_delta_append(journal, t3, p, rec, &hash), "16: v3 record 3 lands");
        uint64_t rst_bytes = 0;
        check(conversation_delta_state_write(rst, t3, 3, hash, rst_bytes), "16: sidecar at 3");
        check(conversation_delta_plan(p, t4, t3.live.ids.size(), t3.live.ids.size(), 4),
              "16: plan 4 is v3");
        check(conversation_delta_append(journal, t4, p, rec, &hash), "16: v3 record 4 lands");
        rst_bytes = 0;
        check(conversation_delta_state_write(rst, t4, 4, hash, rst_bytes), "16: sidecar at 4");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back, rst);
        check(fold.applied == 4 && !fold.torn && !fold.failed, "16: the mixed fold applies 4");
        check(same_image(back, t4), "16: v2 then v3 lands turn 4 byte-exactly");
    }

    // ── 17. v3 without a usable sidecar: an empty path, a corrupt body, a corrupt trailer
    // and a truncated file all make the first v3 record a torn tail — the journal drops whole
    // (nothing chains for the base) and the base restores alone ────────────────────────────
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 150, 2);
        uint64_t rec = 0, hash = 0;
        DeltaPlan p;
        check(conversation_delta_plan(p, t1, 1000, 1000, 1) &&
              conversation_delta_append(journal, t1, p, rec, &hash) &&
              conversation_delta_state_write(rst, t1, 1, hash, rec),
              "17: the v3 pair lands");
        // (a) no state_path at all
        SavedConversation back1 = base;
        const DeltaFold f1 = conversation_delta_apply(journal, back1);
        check(f1.applied == 0 && f1.torn && f1.dropped, "17: no sidecar → dropped whole");
        check(same_image(back1, base), "17: (a) the base is untouched");
        // (b) a corrupt byte in the sidecar's body
        for (const size_t at : {size_t(40), size_t(60)}) {
            const std::string bad = (dir / "conv-1.bad.rst").string();
            fs::copy_file(rst, bad, fs::copy_options::overwrite_existing);
            {
                std::FILE* f = std::fopen(bad.c_str(), "r+b");
                std::fseek(f, (long) at, SEEK_SET);
                const int c = std::fgetc(f);
                std::fseek(f, (long) at, SEEK_SET);
                std::fputc(c ^ 0x5a, f);
                std::fclose(f);
            }
            SavedConversation back2 = base;
            const DeltaFold f2 = conversation_delta_apply(journal, back2, bad);
            check(f2.applied == 0 && f2.dropped, "17: a corrupt sidecar body → dropped");
        }
        // (c) a corrupt byte in the sidecar's trailer (the hash word)
        {
            constexpr size_t kSidecarTrailer = 20;   // u64 hash + u64 total + "ERST"
            const std::string bad = (dir / "conv-1.bad.rst").string();
            fs::copy_file(rst, bad, fs::copy_options::overwrite_existing);
            const uint64_t size = fs::file_size(bad);
            std::FILE* f = std::fopen(bad.c_str(), "r+b");
            std::fseek(f, (long) (size - kSidecarTrailer), SEEK_SET);
            const int c = std::fgetc(f);
            std::fseek(f, (long) (size - kSidecarTrailer), SEEK_SET);
            std::fputc(c ^ 0xff, f);
            std::fclose(f);
            SavedConversation back3 = base;
            const DeltaFold f3 = conversation_delta_apply(journal, back3, bad);
            check(f3.applied == 0 && f3.dropped, "17: a corrupt sidecar trailer → dropped");
        }
        // (d) a truncated sidecar
        {
            const std::string bad = (dir / "conv-1.bad.rst").string();
            fs::copy_file(rst, bad, fs::copy_options::overwrite_existing);
            std::error_code ec;
            fs::resize_file(bad, fs::file_size(bad) / 2, ec);
            check(!ec, "17: truncation works");
            SavedConversation back4 = base;
            const DeltaFold f4 = conversation_delta_apply(journal, back4, bad);
            check(f4.applied == 0 && f4.dropped, "17: a truncated sidecar → dropped");
            fs::remove(bad);
        }
        // (e) the intact sidecar still folds (the corruptions above touched copies only)
        SavedConversation back5 = base;
        const DeltaFold f5 = conversation_delta_apply(journal, back5, rst);
        check(f5.applied == 1 && !f5.failed, "17: the intact pair still folds");
        check(same_image(back5, t1), "17: (e) the image is turn 1's");
    }

    // ── 18. The crash window between the record append and the sidecar's rename: the .dlt
    // holds N records but the .rst still describes N-1. The fold applies 1..N-1, flags the
    // tail torn at record N (its seq passes the sidecar's coverage), and the state loads for
    // N-1 — the caller truncates the .dlt and the pair is coherent again ──────────────────
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 100, 2);
        SavedConversation t2 = grow(t1, 110, 3);
        uint64_t rec1 = 0, hash1 = 0, rec2 = 0, rst_bytes = 0;
        DeltaPlan p1, p2;
        check(conversation_delta_plan(p1, t1, 1000, 1000, 1) &&
              conversation_delta_append(journal, t1, p1, rec1, &hash1) &&
              conversation_delta_state_write(rst, t1, 1, hash1, rst_bytes),
              "18: the turn-1 pair lands");
        // the crash: record 2 appended, the sidecar's rewrite never happened
        check(conversation_delta_plan(p2, t2, t1.live.ids.size(), t1.live.ids.size(), 2) &&
              conversation_delta_append(journal, t2, p2, rec2),
              "18: record 2 lands without its sidecar");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back, rst);
        check(fold.applied == 1 && fold.torn && !fold.failed, "18: the fold ends one record early");
        check(fold.valid_end == rec1, "18: the valid end is record 1's");
        check(same_image(back, t1), "18: the state is turn 1's (the sidecar's)");
        // after the caller truncates, the fold is clean
        std::error_code ec;
        fs::resize_file(journal, fold.valid_end, ec);
        check(!ec, "18: truncate works");
        SavedConversation again = base;
        const DeltaFold fold2 = conversation_delta_apply(journal, again, rst);
        check(fold2.applied == 1 && !fold2.torn && !fold2.failed, "18: the healed pair folds clean");
        check(same_image(again, t1), "18: the healed pair yields turn 1");
    }

    // ── 19. A sidecar from ANOTHER generation: its (seq, record_hash) describes a record 2
    // the journal does not hold (or holds in another form) — the stamp mismatch is `failed`,
    // never a silent state/K-V mismatch ──────────────────────────────────────────────────
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 100, 2);
        SavedConversation t2 = grow(t1, 110, 3);
        uint64_t rec = 0, hash = 0;
        DeltaPlan p;
        check(conversation_delta_plan(p, t1, 1000, 1000, 1) &&
              conversation_delta_append(journal, t1, p, rec, &hash),
              "19: record 1 lands");
        // a sidecar stamped seq=1 but the hash of a DIFFERENT record (another generation's
        // record 1 — e.g. the .dlt was rewritten by a restore under the same serial)
        uint64_t other_hash = hash ^ 0xdeadbeefdeadbeefull;
        uint64_t rst_bytes = 0;
        check(conversation_delta_state_write(rst, t1, 1, other_hash, rst_bytes),
              "19: the foreign sidecar lands");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back, rst);
        check(fold.applied == 1 && fold.failed, "19: the stamp mismatch is failed");
    }

    // ── 20. Corruption in the middle: record k < N's payload is corrupt while the sidecar
    // describes N. The fold tears at k, and the state for k no longer exists anywhere (the
    // v2 records that carried it are gone) — failed, the images are suspect (Fase 4's case)
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 100, 2);
        SavedConversation t2 = grow(t1, 110, 3);
        SavedConversation t3 = grow(t2, 120, 4);
        uint64_t rec1 = 0, hash1 = 0, rec2 = 0, hash2 = 0, rec3 = 0, hash3 = 0, rst_bytes = 0;
        DeltaPlan p;
        check(conversation_delta_plan(p, t1, 1000, 1000, 1) &&
              conversation_delta_append(journal, t1, p, rec1, &hash1) &&
              conversation_delta_state_write(rst, t1, 1, hash1, rst_bytes), "20: turn 1 lands");
        check(conversation_delta_plan(p, t2, t1.live.ids.size(), t1.live.ids.size(), 2) &&
              conversation_delta_append(journal, t2, p, rec2, &hash2) &&
              conversation_delta_state_write(rst, t2, 2, hash2, rst_bytes), "20: turn 2 lands");
        check(conversation_delta_plan(p, t3, t2.live.ids.size(), t2.live.ids.size(), 3) &&
              conversation_delta_append(journal, t3, p, rec3, &hash3) &&
              conversation_delta_state_write(rst, t3, 3, hash3, rst_bytes), "20: turn 3 lands");
        // corrupt record 2's payload (inside rec2's span, past its header)
        {
            std::FILE* f = std::fopen(journal.c_str(), "r+b");
            std::fseek(f, (long) (rec1 + 200), SEEK_SET);
            const int c = std::fgetc(f);
            std::fseek(f, (long) (rec1 + 200), SEEK_SET);
            std::fputc(c ^ 0xff, f);
            std::fclose(f);
        }
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back, rst);
        check(fold.applied == 1 && fold.torn && fold.failed,
              "20: mid-corruption tears at 1 and fails (the state for 1 is gone)");
    }

    // ── 21. state_stale: a journal of v2 records with a sidecar beside it — the .rst
    // describes a v3 generation nothing reaches, the fold reports it, the state comes from
    // the records and the caller deletes the orphan ──────────────────────────────────────
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 100, 2);
        SavedConversation t2 = grow(t1, 110, 3);
        uint64_t rec = 0;
        DeltaPlan p;
        check(conversation_delta_plan(p, t1, 1000, 1000, 0) &&
              conversation_delta_append(journal, t1, p, rec), "21: v2 record 1 lands");
        check(conversation_delta_plan(p, t2, t1.live.ids.size(), t1.live.ids.size(), 0) &&
              conversation_delta_append(journal, t2, p, rec), "21: v2 record 2 lands");
        uint64_t rst_bytes = 0;
        check(conversation_delta_state_write(rst, t2, 9, 0, rst_bytes),
              "21: an orphan sidecar lands beside the v2 journal");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back, rst);
        check(fold.applied == 2 && !fold.failed && fold.state_stale && !fold.dropped,
              "21: the v2 fold succeeds and flags the sidecar stale");
        check(same_image(back, t2), "21: the state comes from the records");
        fs::remove(rst);
        SavedConversation again = base;
        const DeltaFold fold2 = conversation_delta_apply(journal, again, rst);
        check(fold2.applied == 2 && !fold2.state_stale, "21: without the orphan, no stale flag");
    }

    // ── 22. An empty journal with a sidecar: nothing applies, the sidecar is stale ───────
    {
        const std::string empty = (dir / "conv-empty2.dlt").string();
        const std::string rst = (dir / "conv-empty2.rst").string();
        { std::FILE* f = std::fopen(empty.c_str(), "wb"); std::fclose(f); }
        SavedConversation img = image_of(1000, 1);
        uint64_t rst_bytes = 0;
        check(conversation_delta_state_write(rst, img, 1, 0, rst_bytes), "22: the sidecar lands");
        const DeltaFold fold = conversation_delta_apply(empty, img, rst);
        check(fold.had_journal && fold.applied == 0 && !fold.dropped && fold.state_stale,
              "22: an empty journal with a sidecar reports state_stale");
    }

    // ── 23. The Phase 0 instrument: describe() returns the per-image breakdown with the
    // markers the [delta/diag] log greps for ─────────────────────────────────────────────
    {
        SavedConversation base = image_of(1000, 1);
        SavedConversation next = grow(base, 300, 2);
        DeltaPlan plan;
        check(conversation_delta_plan(plan, next, 1000, 1000), "23: the plan lands");
        const std::string d = conversation_delta_describe(next, plan);
        check(d.find("img0") != std::string::npos && d.find("live=") != std::string::npos &&
              d.find("kv_sent=") != std::string::npos && d.find("kv_expected=") != std::string::npos,
              "23: describe carries the breakdown markers");
    }

    // ── 24. The v3 → v2 transition (STRATA_SPILL_JOURNAL_V3=0 after v3 records): a v2 record
    // appended after v3 ones carries its own state, but the EARLIER records still chain
    // through the sidecar — the fold applies all of them, takes record 3's state, and the
    // sidecar is NOT stale (a tail tear could yet land the fold on record 2) ────────────────
    {
        fs::remove(journal);
        const std::string rst = (dir / "conv-1.rst").string();
        fs::remove(rst);
        SavedConversation base = image_of(1000, 1);
        SavedConversation t1 = grow(base, 100, 2);
        SavedConversation t2 = grow(t1, 110, 3);
        SavedConversation t3 = grow(t2, 120, 4);
        uint64_t rec = 0, hash = 0, rst_bytes = 0;
        DeltaPlan p;
        check(conversation_delta_plan(p, t1, 1000, 1000, 1) &&
              conversation_delta_append(journal, t1, p, rec, &hash) &&
              conversation_delta_state_write(rst, t1, 1, hash, rst_bytes), "24: v3 record 1");
        check(conversation_delta_plan(p, t2, t1.live.ids.size(), t1.live.ids.size(), 2) &&
              conversation_delta_append(journal, t2, p, rec, &hash) &&
              conversation_delta_state_write(rst, t2, 2, hash, rst_bytes), "24: v3 record 2");
        // the escape hatch lands a v2 record (seq = 0) and the sidecar stays at seq 2
        check(conversation_delta_plan(p, t3, t2.live.ids.size(), t2.live.ids.size(), 0) &&
              p.version == kDeltaVersionV2 && conversation_delta_append(journal, t3, p, rec),
              "24: v2 record 3 lands");
        SavedConversation back = base;
        const DeltaFold fold = conversation_delta_apply(journal, back, rst);
        check(fold.applied == 3 && !fold.torn && !fold.failed && !fold.state_stale,
              "24: the mixed v3,v3,v2 fold applies all three and keeps the sidecar live");
        check(same_image(back, t3), "24: the state is record 3's own (it is v2)");
        // and a take that truncates back to the v3 pair still restores through the sidecar
        std::error_code ec;
        fs::resize_file(journal, fold.valid_end - rec, ec);
        SavedConversation again = base;
        const DeltaFold fold2 = conversation_delta_apply(journal, again, rst);
        check(fold2.applied == 2 && !fold2.failed && !fold2.state_stale,
              "24: a tail tear onto the v3 pair still folds through the sidecar");
        check(same_image(again, t2), "24: the sidecar's state lands (record 2's)");
    }

    fs::remove_all(dir);
    std::printf("conversation_delta_test: %d checks passed\n", checks);
    return 0;
}
