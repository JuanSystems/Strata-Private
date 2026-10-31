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

    fs::remove_all(dir);
    std::printf("conversation_delta_test: %d checks passed\n", checks);
    return 0;
}
