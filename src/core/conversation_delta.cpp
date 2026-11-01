// Delta journal codec implementation. See include/strata/core/conversation_delta.hpp for the
// record layout and the invariants. Pure host code: no CUDA. The reader never seeks beyond the
// two portable ones (end, to size the file, and back to 0): records are applied strictly
// front-to-back with sequential reads. The one filesystem dependency is the v3 state sidecar's
// atomic publish (temp + rename + directory fsync) — the same discipline the store's .bin and
// .idx writes already use.
#include "strata/core/conversation_delta.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#include <filesystem>

namespace strata::core {
namespace fs = std::filesystem;

namespace {

// ── little-endian scalar codec (the same style as the .idx sidecar's) ──────────────────────
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t((v >> (i * 8)) & 0xFF));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t((v >> (i * 8)) & 0xFF));
}
void put_i64(std::vector<uint8_t>& b, int64_t v) { put_u64(b, uint64_t(v)); }
void put_u8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }

bool get_u32(const uint8_t* p, size_t n, size_t& at, uint32_t& v) {
    if (at + 4 > n) return false;
    v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t(p[at + i]) << (i * 8);
    at += 4;
    return true;
}
bool get_u64(const uint8_t* p, size_t n, size_t& at, uint64_t& v) {
    if (at + 8 > n) return false;
    v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(p[at + i]) << (i * 8);
    at += 8;
    return true;
}
bool get_i64(const uint8_t* p, size_t n, size_t& at, int64_t& v) {
    uint64_t u = 0;
    if (!get_u64(p, n, at, u)) return false;
    v = int64_t(u);
    return true;
}
bool get_u8(const uint8_t* p, size_t n, size_t& at, uint8_t& v) {
    if (at + 1 > n) return false;
    v = p[at++];
    return true;
}

// FNV-1a 64, incremental (the .idx sidecar's checksum algorithm; the journal's record hash).
inline uint64_t fnv_update(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
inline uint64_t fnv_begin() { return 1469598103934665603ull; }

constexpr size_t kHeaderBytes = 129;   // the fixed header, header only (v2: n_images + sections_bytes)
constexpr size_t kTrailerBytes = 20;   // hash + record_bytes + "EDLT" (the .rst uses the same)
constexpr size_t kLayerHeaderBytes = 7 * 8;
constexpr size_t kPartTableBytesPerLayer = 5 * 16;
constexpr size_t kSectionFixedBytes = 4 + 8 + 8 + 40;       // v2: n_layers + geometry + ckpt + state_bytes[5]
constexpr size_t kSectionFixedBytesV3 = 4 + 8;              // v3: n_layers + geometry (state in the .rst)
constexpr size_t kStateHeaderBytes = 32;                    // "SRST" + version + seq + hash + n_images + rsv
constexpr uint32_t kMaxImages = 64;                         // a sane layer-split bound (63 later stages)

// One image's 5 K/V parts. The mutable overload is the apply side's writer.
const ConversationBuffer* part_of(const ConversationKv& kv, size_t part) {
    switch (part) {
        case 0: return &kv.k;
        case 1: return &kv.v;
        case 2: return &kv.k_scale;
        case 3: return &kv.v_scale;
        default: return &kv.pooled;
    }
}
ConversationBuffer* part_of(ConversationKv& kv, size_t part) {
    switch (part) {
        case 0: return &kv.k;
        case 1: return &kv.v;
        case 2: return &kv.k_scale;
        case 3: return &kv.v_scale;
        default: return &kv.pooled;
    }
}

// A running-state array by index 0..4 (gdn, ple, tails, dead, block_pos).
const std::vector<uint8_t>* state_of(const ConversationCheckpoint& c, size_t i) {
    switch (i) {
    case 0: return &c.gdn;
    case 1: return &c.ple;
    case 2: return &c.tails;
    case 3: return &c.dead;
    default: return &c.block_pos;
    }
}
std::vector<uint8_t>* state_of(ConversationCheckpoint& c, size_t i) {
    switch (i) {
    case 0: return &c.gdn;
    case 1: return &c.ple;
    case 2: return &c.tails;
    case 3: return &c.dead;
    default: return &c.block_pos;
    }
}

// The checkpoint a record carries: the same ONE the v1 session file stores
// (session_checkpoints_to_save → session_deepest_checkpoint, conversation_file.cpp) — the
// chain's deepest, or with a pinned shared prefix in the chain the deepest PINNED one.
// Serializing the whole capture chain here made every turn's append re-send up to
// --prompt-cache checkpoints (each with its full running state, once per image section), so
// the .dlt outgrew the .bin it extends — the opposite of the journal's purpose. One
// checkpoint per record keeps the invariant exact and cheap: fold(base + records) == a full
// write of the newest park, and the apply's replace semantics (img.checkpoints =
// sec.checkpoints) lands the folded image on exactly that chain.
const ConversationCheckpoint* record_checkpoint(const std::vector<ConversationCheckpoint>& chain) {
    const ConversationCheckpoint* deepest = nullptr;
    bool any_pinned = false;
    for (const auto& c : chain) any_pinned = any_pinned || c.pinned;
    for (const auto& c : chain)
        if ((!any_pinned || c.pinned) && (!deepest || c.ids.size() > deepest->ids.size()))
            deepest = &c;
    return deepest;
}

// The image list a record covers: the main first, then its stage images in order.
std::vector<const SavedConversation*> images_of(const SavedConversation& image) {
    std::vector<const SavedConversation*> out;
    out.reserve(1 + image.stage_images.size());
    out.push_back(&image);
    for (const auto& st : image.stage_images) out.push_back(&st);
    return out;
}

// The mutable twin: the apply side writes the records' state onto the images it holds.
std::vector<SavedConversation*> images_of(SavedConversation& image) {
    std::vector<SavedConversation*> out;
    out.reserve(1 + image.stage_images.size());
    out.push_back(&image);
    for (auto& st : image.stage_images) out.push_back(&st);
    return out;
}

// The shared region's byte size: the ids suffix + the image list (one live state, every image).
uint64_t shared_region_bytes(const SavedConversation& image, uint64_t keep_tokens) {
    uint64_t n = (uint64_t(image.live.ids.size()) - keep_tokens) * sizeof(int32_t);
    n += 8 + image.live.imgs.size() * 16;
    return n;
}

void put_shared_region(std::vector<uint8_t>& b, const SavedConversation& image, uint64_t keep_tokens) {
    for (size_t i = size_t(keep_tokens); i < image.live.ids.size(); ++i)
        put_u32(b, uint32_t(image.live.ids[i]));
    put_u64(b, image.live.imgs.size());
    for (const auto& k : image.live.imgs) {
        put_i64(b, k.start);
        put_u64(b, k.hash);
    }
}

// One checkpoint's identity head (ids + imgs, no state): the .rst writes it, then streams the
// state arrays after it — the record path reuses it through put_checkpoint.
void put_checkpoint_head(std::vector<uint8_t>& b, const ConversationCheckpoint& c) {
    put_u64(b, c.ids.size());
    for (int32_t v : c.ids) put_u32(b, uint32_t(v));
    put_u64(b, c.imgs.size());
    for (const auto& k : c.imgs) {
        put_i64(b, k.start);
        put_u64(b, k.hash);
    }
}

// One checkpoint block complete (the same layout the v1 small region used): head + the five
// state arrays (size + bytes each) + used.
void put_checkpoint(std::vector<uint8_t>& b, const ConversationCheckpoint& c) {
    put_checkpoint_head(b, c);
    for (size_t i = 0; i < 5; ++i) {
        const auto& s = *state_of(c, i);
        put_u64(b, s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    put_u64(b, c.used);
}

// The exact byte size of put_checkpoint's block (put_u64 sizes are 8 B each: 5 array sizes +
// used, plus the head's own counts).
uint64_t checkpoint_bytes(const ConversationCheckpoint& c) {
    uint64_t n = 8 + c.ids.size() * sizeof(int32_t) + 8 + c.imgs.size() * 16 + 40;
    for (size_t i = 0; i < 5; ++i) n += state_of(c, i)->size();
    return n + 8;
}

// The K/V layer headers + the parts table every section carries, both versions (the apply side
// sizes the record from them alone; the state never enters the table).
void put_layer_tables(std::vector<uint8_t>& b, const SavedConversation& img, const DeltaImagePlan& plan) {
    for (const auto& kv : img.kv) {
        put_i64(b, kv.format); put_i64(b, kv.cells); put_i64(b, kv.heads);
        put_i64(b, kv.head_dim); put_i64(b, kv.page_size);
        put_i64(b, kv.pooled_rows); put_i64(b, kv.idx_dim);
    }
    for (const auto& layer : plan.parts)
        for (const auto& part : layer) {
            put_u64(b, part.at);
            put_u64(b, part.n);
        }
}

// One section's front matter: v2 adds the live state + the record's ONE checkpoint; v3 carries
// only the fixed fields + the layer tables (the state lives in the .rst sidecar).
uint64_t section_front_bytes(const SavedConversation& img, bool v3) {
    uint64_t n = v3 ? kSectionFixedBytesV3 : kSectionFixedBytes;
    if (!v3) {
        for (size_t i = 0; i < 5; ++i) n += state_of(img.live, i)->size();
        n += 4;
        if (const ConversationCheckpoint* sel = record_checkpoint(img.checkpoints))
            n += checkpoint_bytes(*sel);
    }
    n += img.kv.size() * (kLayerHeaderBytes + kPartTableBytesPerLayer);
    return n;
}

void put_section_front(std::vector<uint8_t>& b, const SavedConversation& img, uint64_t geometry_hash,
                       const DeltaImagePlan& plan, bool v3) {
    put_u32(b, uint32_t(img.kv.size()));
    put_u64(b, geometry_hash);
    if (!v3) {
        const ConversationCheckpoint* sel = record_checkpoint(img.checkpoints);
        put_u64(b, sel ? (uint64_t) sel->ids.size() : 0);
        for (size_t i = 0; i < 5; ++i) put_u64(b, state_of(img.live, i)->size());
        for (size_t i = 0; i < 5; ++i) {
            const auto& s = *state_of(img.live, i);
            b.insert(b.end(), s.begin(), s.end());
        }
        put_u32(b, sel ? 1u : 0u);   // the record's ONE checkpoint: the full-write selection
        if (sel) put_checkpoint(b, *sel);
    }
    put_layer_tables(b, img, plan);
}

// A parsed section (the apply side). Every count is bounded against the front buffer's own
// size while parsing, so no vector can exceed the record's bytes.
struct ParsedSection {
    uint64_t geometry_hash = 0, ckpt_tokens = 0;
    std::array<std::vector<uint8_t>, 5> state;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<int64_t> layer_headers;                    // n_layers x 7
    std::vector<std::array<DeltaPart, 5>> table;           // n_layers x 5
};

// A parsed shared region.
struct ParsedShared {
    std::vector<int32_t> ids_suffix;
    std::vector<ConversationImageKey> imgs;
};

bool parse_shared_region(const uint8_t* p, size_t n, uint64_t keep_tokens, uint64_t new_tokens,
                         uint64_t n_imgs, ParsedShared& out) {
    size_t at = 0;
    const uint64_t suffix = new_tokens - keep_tokens;
    if (suffix > n / sizeof(int32_t)) return false;
    out.ids_suffix.resize(size_t(suffix));
    for (uint64_t i = 0; i < suffix; ++i) {
        uint32_t v = 0;
        if (!get_u32(p, n, at, v)) return false;
        out.ids_suffix[size_t(i)] = int32_t(v);
    }
    uint64_t imgs_count = 0;
    if (!get_u64(p, n, at, imgs_count) || imgs_count != n_imgs) return false;   // the region's own count
    if (n_imgs > (n - at) / 16) return false;
    out.imgs.resize(size_t(n_imgs));
    for (uint64_t i = 0; i < n_imgs; ++i) {
        int64_t start = 0;
        uint64_t hash = 0;
        if (!get_i64(p, n, at, start) || !get_u64(p, n, at, hash)) return false;
        out.imgs[size_t(i)] = ConversationImageKey{start, hash};
    }
    return at == n;
}

// The v2 section's state + checkpoint block (everything between geometry_hash and the layer
// headers). Extracted from parse_section_at verbatim so the v3 parser can skip it; the cursor
// contract and every bound are the original ones.
bool parse_state_and_checkpoints(const uint8_t* p, size_t n, size_t& at, ParsedSection& out) {
    if (!get_u64(p, n, at, out.ckpt_tokens)) return false;
    // the five state arrays are laid out sizes-first (the writer's put_section_front): 5 u64
    // sizes, then the 5 byte arrays back to back
    uint64_t state_sz[5] = {};
    for (size_t i = 0; i < 5; ++i)
        if (!get_u64(p, n, at, state_sz[i]) || state_sz[i] > n - at) return false;
    for (size_t i = 0; i < 5; ++i) {
        if (state_sz[i] > n - at) return false;
        out.state[i].assign(p + at, p + at + size_t(state_sz[i]));
        at += size_t(state_sz[i]);
    }
    uint32_t n_ckpt = 0;
    if (!get_u32(p, n, at, n_ckpt) || n_ckpt > (n - at) / 8) return false;
    out.checkpoints.resize(n_ckpt);
    for (uint32_t c = 0; c < n_ckpt; ++c) {
        ConversationCheckpoint& k = out.checkpoints[c];
        uint64_t n_ids = 0;
        if (!get_u64(p, n, at, n_ids) || n_ids > (n - at) / sizeof(int32_t)) return false;
        k.ids.resize(size_t(n_ids));
        for (uint64_t i = 0; i < n_ids; ++i) {
            uint32_t v = 0;
            if (!get_u32(p, n, at, v)) return false;
            k.ids[size_t(i)] = int32_t(v);
        }
        uint64_t n_cimgs = 0;
        if (!get_u64(p, n, at, n_cimgs) || n_cimgs > (n - at) / 16) return false;
        k.imgs.resize(size_t(n_cimgs));
        for (uint64_t i = 0; i < n_cimgs; ++i) {
            int64_t start = 0;
            uint64_t hash = 0;
            if (!get_i64(p, n, at, start) || !get_u64(p, n, at, hash)) return false;
            k.imgs[size_t(i)] = ConversationImageKey{start, hash};
        }
        for (size_t i = 0; i < 5; ++i) {
            uint64_t sz = 0;
            if (!get_u64(p, n, at, sz) || sz > n - at) return false;
            state_of(k, i)->assign(p + at, p + at + size_t(sz));
            at += size_t(sz);
        }
        if (!get_u64(p, n, at, k.used)) return false;
    }
    return true;
}

// Parse one section starting at the cursor `at` (updated past what it consumed). Every count
// is bounded against the buffer's own size, so no vector can exceed the record's bytes. A v3
// section carries no state and no checkpoint (they live in the .rst sidecar).
bool parse_section_at(const uint8_t* p, size_t n, size_t& at, ParsedSection& out, bool v3) {
    uint32_t n_layers = 0;
    if (!get_u32(p, n, at, n_layers) || n_layers == 0 || n_layers > (1u << 12)) return false;
    if (!get_u64(p, n, at, out.geometry_hash)) return false;
    if (!v3 && !parse_state_and_checkpoints(p, n, at, out)) return false;
    out.layer_headers.resize(size_t(n_layers) * 7);
    for (size_t i = 0; i < out.layer_headers.size(); ++i)
        if (!get_i64(p, n, at, out.layer_headers[i])) return false;
    out.table.resize(size_t(n_layers));
    for (auto& layer : out.table)
        for (auto& part : layer) {
            if (!get_u64(p, n, at, part.at) || !get_u64(p, n, at, part.n)) return false;
        }
    return true;
}

// The fixed header as the reader parses it. v3 overloads state_bytes[0] with the record's seq.
struct RecordHeader {
    uint32_t version = 0;
    uint64_t seq = 0;
    uint64_t record_bytes = 0, applies_to_len = 0, keep_tokens = 0, new_tokens = 0, small_bytes = 0;
    uint64_t geometry_hash = 0, n_imgs = 0, ckpt_tokens = 0, sections_bytes = 0;
    uint64_t state_bytes[5]{};
    uint32_t n_layers = 0, n_images = 0;
    uint8_t cvec = 0;
};

bool parse_header(const uint8_t* p, size_t n, RecordHeader& h) {
    size_t at = 0;
    uint32_t version = 0;
    if (n < kHeaderBytes) return false;
    for (int i = 0; i < 4; ++i)
        if (p[i] != uint8_t(kDeltaMagic[i])) return false;
    at = 4;
    if (!get_u32(p, n, at, version) ||
        (version != kDeltaVersionV2 && version != kDeltaVersion)) return false;
    h.version = version;
    if (!get_u64(p, n, at, h.record_bytes) || !get_u64(p, n, at, h.applies_to_len) ||
        !get_u64(p, n, at, h.keep_tokens) || !get_u64(p, n, at, h.new_tokens) ||
        !get_u64(p, n, at, h.small_bytes) || !get_u64(p, n, at, h.geometry_hash))
        return false;
    if (!get_u8(p, n, at, h.cvec)) return false;
    if (!get_u32(p, n, at, h.n_images)) return false;
    if (!get_u32(p, n, at, h.n_layers)) return false;
    if (!get_u64(p, n, at, h.n_imgs) || !get_u64(p, n, at, h.ckpt_tokens)) return false;
    for (size_t i = 0; i < 5; ++i)
        if (!get_u64(p, n, at, h.state_bytes[i])) return false;
    if (!get_u64(p, n, at, h.sections_bytes)) return false;
    // v3: state_bytes[0] is the record's 1-based index within the journal (see the .hpp)
    if (h.version >= kDeltaVersion) h.seq = h.state_bytes[0];
    // sanity: the counts must fit the record's own bytes, and the chain fields must line up
    const uint64_t min_record = kHeaderBytes + h.small_bytes + h.sections_bytes + kTrailerBytes;
    if (h.record_bytes < min_record || h.record_bytes > (1ull << 40)) return false;
    if (h.keep_tokens > h.applies_to_len || h.new_tokens < h.keep_tokens) return false;
    if (h.new_tokens - h.keep_tokens > h.small_bytes / sizeof(int32_t)) return false;
    if (h.n_images == 0 || h.n_images > kMaxImages) return false;
    if (h.n_layers == 0 || h.n_layers > (1u << 12)) return false;
    if (h.n_imgs > (1u << 20)) return false;
    return true;
}

// Read the record's front (shared region + every section's front matter) from the stream's
// current position, which is just past the fixed header. Sequential reads only.
bool read_record_front(std::FILE* f, const RecordHeader& h, std::vector<uint8_t>& front) {
    const uint64_t bytes = h.small_bytes + h.sections_bytes;
    if (bytes > h.record_bytes - kHeaderBytes - kTrailerBytes) return false;
    try {
        front.resize(size_t(bytes));
    } catch (const std::exception&) {
        return false;
    }
    if (std::fread(front.data(), 1, front.size(), f) != front.size()) return false;
    return true;
}

// ── The v3 state sidecar's reader helpers (sequential reads, bounded against the file) ──────
bool read_exact(std::FILE* f, void* dst, size_t n) { return n == 0 || std::fread(dst, 1, n, f) == n; }
bool read_u32(std::FILE* f, uint32_t& v) {
    uint8_t t[4]; size_t at = 0;
    return read_exact(f, t, 4) && get_u32(t, 4, at, v);
}
bool read_u64(std::FILE* f, uint64_t& v) {
    uint8_t t[8]; size_t at = 0;
    return read_exact(f, t, 8) && get_u64(t, 8, at, v);
}

// What state_file_probe validated: the sidecar's stamp (seq + record_hash) and exact size.
struct StateFileInfo {
    uint64_t seq = 0, record_hash = 0, total = 0;
    uint32_t n_images = 0;
};

// Validate a .rst whole (magic, version, streaming FNV over the body, trailer hash + size +
// end magic) and report its stamp. The load path re-reads without re-hashing: the file cannot
// move under us (the writer is drained first), so the probe's verdict holds.
bool state_file_probe(const std::string& path, StateFileInfo& info) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = false;
    do {
        if (std::fseek(f, 0, SEEK_END) != 0) break;
        const long long size = std::ftell(f);
        if (size < (long long) (kStateHeaderBytes + kTrailerBytes) ||
            std::fseek(f, 0, SEEK_SET) != 0)
            break;
        uint8_t head[kStateHeaderBytes];
        if (std::fread(head, 1, sizeof(head), f) != sizeof(head)) break;
        bool magic_ok = true;
        for (int i = 0; i < 4; ++i) magic_ok = magic_ok && head[i] == uint8_t(kStateMagic[i]);
        if (!magic_ok) break;
        {
            size_t at = 4;
            uint32_t version = 0, reserved = 0;
            if (!get_u32(head, sizeof(head), at, version) || version != kStateVersion) break;
            if (!get_u64(head, sizeof(head), at, info.seq) ||
                !get_u64(head, sizeof(head), at, info.record_hash) ||
                !get_u32(head, sizeof(head), at, info.n_images) ||
                !get_u32(head, sizeof(head), at, reserved))
                break;
            if (info.n_images == 0 || info.n_images > kMaxImages) break;
        }
        uint64_t h = fnv_update(fnv_begin(), head, sizeof(head));
        std::vector<uint8_t> chunk;
        try {
            chunk.resize(64 * 1024);
        } catch (const std::exception&) {
            break;
        }
        bool streamed = true;
        for (uint64_t left = uint64_t(size) - sizeof(head) - kTrailerBytes; left > 0 && streamed;) {
            const size_t take = size_t(std::min<uint64_t>(left, chunk.size()));
            streamed = std::fread(chunk.data(), 1, take, f) == take;
            h = fnv_update(h, chunk.data(), take);
            left -= take;
        }
        if (!streamed) break;
        uint8_t trailer[kTrailerBytes];
        if (std::fread(trailer, 1, sizeof(trailer), f) != sizeof(trailer)) break;
        size_t at = 0;
        uint64_t stored_hash = 0, stored_total = 0;
        if (!get_u64(trailer, sizeof(trailer), at, stored_hash) ||
            !get_u64(trailer, sizeof(trailer), at, stored_total))
            break;
        bool end_ok = true;
        for (int i = 0; i < 4; ++i)
            end_ok = end_ok && trailer[kTrailerBytes - 4 + i] == uint8_t(kStateEndMagic[i]);
        if (!end_ok || stored_hash != h || stored_total != uint64_t(size)) break;
        info.total = uint64_t(size);
        ok = true;
    } while (false);
    std::fclose(f);
    return ok;
}

// One checkpoint block from the sidecar (the writer's put_checkpoint_head + arrays + used),
// bounded by `left` (the bytes the probe said the body holds).
bool read_checkpoint(std::FILE* f, uint64_t& left, ConversationCheckpoint& c) {
    auto take_n = [&](uint64_t n) { if (n > left) return false; left -= n; return true; };
    uint64_t n_ids = 0, n_imgs = 0;
    if (!take_n(8) || !read_u64(f, n_ids) || !take_n(n_ids * 4)) return false;
    c.ids.resize(size_t(n_ids));
    for (uint64_t i = 0; i < n_ids; ++i) {
        uint32_t v = 0;
        if (!read_u32(f, v)) return false;
        c.ids[size_t(i)] = int32_t(v);
    }
    if (!take_n(8) || !read_u64(f, n_imgs) || !take_n(n_imgs * 16)) return false;
    c.imgs.resize(size_t(n_imgs));
    for (uint64_t i = 0; i < n_imgs; ++i) {
        uint64_t start = 0, hash = 0;
        if (!read_u64(f, start) || !read_u64(f, hash)) return false;
        c.imgs[size_t(i)] = ConversationImageKey{int64_t(start), hash};
    }
    for (size_t i = 0; i < 5; ++i) {
        uint64_t sz = 0;
        if (!take_n(8) || !read_u64(f, sz) || !take_n(sz)) return false;
        auto& dst = *state_of(c, i);
        dst.resize(size_t(sz));
        if (!read_exact(f, dst.data(), dst.size())) return false;
    }
    return take_n(8) && read_u64(f, c.used);
}

// Load a PROBED sidecar onto the images the reader holds (main + stages, in order): the live
// running state and the full-write checkpoint of every image, each guarded by its geometry
// echo. The ids/dims/K/V are NOT here — the records carry them. `left` must land exactly on
// zero (a layout drift is a corrupt sidecar).
bool state_file_load(const std::string& path, const StateFileInfo& info,
                     const std::vector<SavedConversation*>& images) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = std::fseek(f, (long long) kStateHeaderBytes, SEEK_SET) == 0;
    uint64_t left = info.total - kStateHeaderBytes - kTrailerBytes;
    auto take_n = [&](uint64_t n) { if (n > left) return false; left -= n; return true; };
    for (size_t ii = 0; ok && ii < images.size(); ++ii) {
        SavedConversation& img = *images[ii];
        uint64_t geom = 0, sz[5] = {};
        ok = take_n(8) && read_u64(f, geom) &&
             geom == conversation_delta_geometry_hash(img.geometry, img.layer_lo, img.layer_hi);
        for (size_t i = 0; ok && i < 5; ++i) ok = take_n(8) && read_u64(f, sz[i]);
        for (size_t i = 0; ok && i < 5; ++i) {
            ok = take_n(sz[i]);
            if (!ok) break;
            auto& dst = *state_of(img.live, i);
            dst.resize(size_t(sz[i]));
            ok = read_exact(f, dst.data(), dst.size());
        }
        uint32_t n_ckpt = 0;
        ok = ok && take_n(4) && read_u32(f, n_ckpt) && n_ckpt <= 1;
        img.checkpoints.clear();
        if (ok && n_ckpt == 1) {
            img.checkpoints.resize(1);
            ok = read_checkpoint(f, left, img.checkpoints[0]);
        }
    }
    std::fclose(f);
    return ok && left == 0;
}

} // namespace

uint64_t conversation_delta_geometry_hash(const std::array<int64_t, 18>& geometry, int64_t layer_lo,
                                          int64_t layer_hi) {
    uint64_t h = fnv_begin();
    for (int64_t v : geometry) h = fnv_update(h, &v, sizeof(v));
    h = fnv_update(h, &layer_lo, sizeof(layer_lo));
    h = fnv_update(h, &layer_hi, sizeof(layer_hi));
    return h;
}

bool conversation_delta_plan(DeltaPlan& plan, const SavedConversation& image, uint64_t base_len,
                             uint64_t keep_tokens, uint64_t seq) {
    plan = DeltaPlan{};
    plan.seq = seq;
    plan.version = seq > 0 ? kDeltaVersion : kDeltaVersionV2;
    const bool v3 = seq > 0;
    const std::vector<const SavedConversation*> images = images_of(image);
    if (images.empty() || images[0]->kv.empty()) return false;
    for (const SavedConversation* img : images)
        if (img->kv.empty()) return false;
    if (keep_tokens > base_len || keep_tokens > image.live.ids.size() || base_len == 0) return false;
    if (keep_tokens * 2 < (uint64_t) image.live.ids.size()) return false;   // the base must carry half
    // One live state: every image of the spill carries the same ids and image list (one capture
    // builds them from one `live`). A mismatch is not a delta-able capture — the full write
    // keeps whatever each image holds.
    for (size_t k = 1; k < images.size(); ++k) {
        if (images[k]->live.ids != image.live.ids || images[k]->live.imgs != image.live.imgs) return false;
    }
    plan.applies_to_len = base_len;
    plan.keep_tokens = keep_tokens;
    plan.new_tokens = (uint64_t) image.live.ids.size();
    plan.geometry_hash = conversation_delta_geometry_hash(image.geometry, image.layer_lo, image.layer_hi);
    plan.cvec = image.cvec;
    plan.images.resize(images.size());
    uint64_t payload = 0;
    for (size_t ii = 0; ii < images.size(); ++ii) {
        const SavedConversation& img = *images[ii];
        DeltaImagePlan& ip = plan.images[ii];
        ip.geometry_hash = conversation_delta_geometry_hash(img.geometry, img.layer_lo, img.layer_hi);
        ip.n_layers = img.kv.size();
        ip.parts.assign(img.kv.size(), {});
        for (size_t j = 0; j < img.kv.size(); ++j) {
            const ConversationKv& kv = img.kv[j];
            if (kv.page_size <= 0 || kv.cells < 0) return false;
            // completed pages of the kept prefix: cells below this boundary are byte-identical in
            // the base (conversation_kv_save's reuse rule — "completed pages strictly before the
            // first rewritten token remain identical")
            const uint64_t whole_cells = keep_tokens / uint64_t(kv.page_size) * uint64_t(kv.page_size);
            if (whole_cells > uint64_t(kv.cells)) return false;
            // the pooled boundary needs idx_block, which no image carries: derive an UPPER bound
            // from this image's own pooled_rows (pooled_rows = tokens/idx_block + 1 for an index
            // layer). An over-estimate of idx_block under-counts completed rows, so the record
            // only ever re-sends more bytes — it never keeps bytes that may differ.
            uint64_t pooled_keep_rows = 0;
            if (kv.pooled_rows > 1) {
                const uint64_t idx_block_ub = plan.new_tokens / uint64_t(kv.pooled_rows - 1);
                if (idx_block_ub == 0) return false;
                pooled_keep_rows = keep_tokens / idx_block_ub;
            }
            for (size_t part = 0; part < 5; ++part) {
                const ConversationBuffer& buf = *part_of(kv, part);
                uint64_t at = 0;
                if (part < 4) {
                    if (kv.cells > 0) {
                        if (buf.size() % uint64_t(kv.cells) != 0) return false;   // not cell-aligned
                        at = (buf.size() / uint64_t(kv.cells)) * whole_cells;
                    }
                } else {
                    at = pooled_keep_rows * (uint64_t(kv.idx_dim) * 4);
                }
                if (at > buf.size()) return false;
                ip.parts[j][part] = DeltaPart{at, buf.size() - at};
                payload += buf.size() - at;
            }
        }
    }
    const uint64_t small = shared_region_bytes(image, keep_tokens);
    uint64_t sections = 0;
    for (size_t ii = 0; ii < images.size(); ++ii)
        sections += section_front_bytes(*images[ii], v3);
    plan.record_bytes = kHeaderBytes + small + sections + payload + kTrailerBytes;
    return true;
}

bool conversation_delta_append(const std::string& path, const SavedConversation& image, const DeltaPlan& plan,
                               uint64_t& record_bytes, uint64_t* record_hash) {
    record_bytes = 0;
    if (record_hash) *record_hash = 0;
    const bool v3 = plan.version >= kDeltaVersion;
    if (v3 && plan.seq == 0) return false;   // a v3 record is identified by its journal index
    const std::vector<const SavedConversation*> images = images_of(image);
    if (plan.images.size() != images.size() || images.empty()) return false;
    for (size_t ii = 0; ii < images.size(); ++ii) {
        if (plan.images[ii].n_layers != images[ii]->kv.size() ||
            plan.images[ii].n_layers != plan.images[ii].parts.size())
            return false;
    }
    if (plan.keep_tokens > image.live.ids.size()) return false;
    const uint64_t small = shared_region_bytes(image, plan.keep_tokens);
    uint64_t payload = 0, sections = 0;
    for (const auto& ip : plan.images)
        for (const auto& layer : ip.parts)
            for (const auto& part : layer) payload += part.n;
    for (size_t ii = 0; ii < images.size(); ++ii)
        sections += section_front_bytes(*images[ii], v3);
    if (plan.record_bytes != kHeaderBytes + small + sections + payload + kTrailerBytes)
        return false;
    std::FILE* f = std::fopen(path.c_str(), "ab");
    if (!f) return false;
    bool ok = false;
    uint64_t h = fnv_begin();
    do {
        // header + shared region + sections' front matter, hashed and written in one go. In v3
        // the front holds no state (the .rst sidecar carries it), so this buffer is O(tables),
        // not O(model state) — the writer's RAM peak of v2 is gone.
        std::vector<uint8_t> front;
        try {
            front.reserve(kHeaderBytes + size_t(small) + size_t(sections));
        } catch (const std::exception&) {
            break;
        }
        for (int i = 0; i < 4; ++i) front.push_back(uint8_t(kDeltaMagic[i]));
        put_u32(front, plan.version);
        put_u64(front, plan.record_bytes);
        put_u64(front, plan.applies_to_len);
        put_u64(front, plan.keep_tokens);
        put_u64(front, plan.new_tokens);
        put_u64(front, small);
        put_u64(front, plan.geometry_hash);
        put_u8(front, image.cvec ? 1 : 0);
        put_u32(front, uint32_t(plan.images.size()));
        put_u32(front, uint32_t(plan.images[0].n_layers));
        put_u64(front, image.live.imgs.size());
        const ConversationCheckpoint* main_ckpt = record_checkpoint(image.checkpoints);
        put_u64(front, main_ckpt ? (uint64_t) main_ckpt->ids.size() : 0);
        if (v3) {
            // v3: the state_bytes[5] slot carries the record's journal index; [1..4] are zero
            put_u64(front, plan.seq);
            for (size_t i = 1; i < 5; ++i) put_u64(front, 0);
        } else {
            for (size_t i = 0; i < 5; ++i) put_u64(front, state_of(image.live, i)->size());
        }
        put_u64(front, sections);
        put_shared_region(front, image, plan.keep_tokens);
        for (size_t ii = 0; ii < images.size(); ++ii)
            put_section_front(front, *images[ii], plan.images[ii].geometry_hash, plan.images[ii], v3);
        h = fnv_update(h, front.data(), front.size());
        if (std::fwrite(front.data(), 1, front.size(), f) != front.size()) break;
        // K/V payloads, streamed straight from each image's buffers (section order)
        bool streamed = true;
        for (size_t ii = 0; ii < images.size() && streamed; ++ii) {
            const SavedConversation& img = *images[ii];
            for (size_t j = 0; j < img.kv.size() && streamed; ++j)
                for (size_t part = 0; part < 5 && streamed; ++part) {
                    const uint64_t at = plan.images[ii].parts[j][part].at, n = plan.images[ii].parts[j][part].n;
                    if (n == 0) continue;
                    streamed = part_of(img.kv[j], part)->visit(size_t(at), size_t(n),
                            [&](const uint8_t* p, size_t c, size_t) {
                                h = fnv_update(h, p, c);
                                return std::fwrite(p, 1, c, f) == c;
                            });
                }
        }
        if (!streamed) break;
        // trailer
        std::vector<uint8_t> tail;
        put_u64(tail, h);
        put_u64(tail, plan.record_bytes);
        for (int i = 0; i < 4; ++i) tail.push_back(uint8_t(kDeltaEndMagic[i]));
        if (std::fwrite(tail.data(), 1, tail.size(), f) != tail.size()) break;
        if (std::fflush(f) != 0) break;
#if !defined(_WIN32)
        // the record's bytes must survive a power cut the moment append returns — the same
        // discipline the .bin's publish applies to its rename
        (void) ::fsync(::fileno(f));
#endif
        ok = true;
    } while (false);
    if (std::fclose(f) != 0) ok = false;
    if (ok) {
        record_bytes = plan.record_bytes;
        if (record_hash) *record_hash = h;   // the sidecar binds to this record through it
    }
    return ok;
}

uint64_t conversation_delta_state_bytes(const SavedConversation& image) {
    uint64_t n = kStateHeaderBytes;
    for (const SavedConversation* img : images_of(image)) {
        n += 8 + 40 + 4;   // geometry echo + the 5 sizes + the checkpoint count
        for (size_t i = 0; i < 5; ++i) n += state_of(img->live, i)->size();
        if (const ConversationCheckpoint* sel = record_checkpoint(img->checkpoints))
            n += checkpoint_bytes(*sel);
    }
    return n + kTrailerBytes;
}

bool conversation_delta_state_write(const std::string& path, const SavedConversation& image, uint64_t seq,
                                    uint64_t record_hash, uint64_t& bytes) {
    bytes = 0;
    const std::vector<const SavedConversation*> images = images_of(image);
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    uint64_t h = fnv_begin();
    uint64_t total = 0;
    bool ok = true;
    auto emit = [&](const void* p, size_t n) {
        if (n == 0) return true;
        h = fnv_update(h, p, n);
        total += n;
        return std::fwrite(p, 1, n, f) == n;
    };
    auto emit_u32 = [&](uint32_t v) { std::vector<uint8_t> t; put_u32(t, v); return emit(t.data(), t.size()); };
    auto emit_u64 = [&](uint64_t v) { std::vector<uint8_t> t; put_u64(t, v); return emit(t.data(), t.size()); };
    do {
        // the fixed header: magic, version, the record this state belongs to (seq + its
        // trailer hash), the image count, a reserved word
        std::vector<uint8_t> head;
        for (int i = 0; i < 4; ++i) head.push_back(uint8_t(kStateMagic[i]));
        put_u32(head, kStateVersion);
        put_u64(head, seq);
        put_u64(head, record_hash);
        put_u32(head, uint32_t(images.size()));
        put_u32(head, 0);
        if (!(ok = emit(head.data(), head.size()))) break;
        // per image: the geometry echo, the live running state streamed straight from the
        // vectors (no front buffer — the O(state) RAM peak of a v2 record never happens here),
        // and the full-write checkpoint selection
        for (const SavedConversation* img : images) {
            ok = emit_u64(conversation_delta_geometry_hash(img->geometry, img->layer_lo, img->layer_hi));
            for (size_t i = 0; ok && i < 5; ++i) ok = emit_u64(state_of(img->live, i)->size());
            for (size_t i = 0; ok && i < 5; ++i) {
                const auto& s = *state_of(img->live, i);
                ok = emit(s.data(), s.size());
            }
            const ConversationCheckpoint* sel = record_checkpoint(img->checkpoints);
            ok = ok && emit_u32(sel ? 1u : 0u);
            if (ok && sel) {
                std::vector<uint8_t> ck;
                put_checkpoint_head(ck, *sel);
                ok = emit(ck.data(), ck.size());
                for (size_t i = 0; ok && i < 5; ++i) {
                    const auto& s = *state_of(*sel, i);
                    ok = emit_u64(s.size()) && emit(s.data(), s.size());
                }
                ok = ok && emit_u64(sel->used);
            }
            if (!ok) break;
        }
        if (!ok) break;
        // trailer: hash over everything above + the exact file size + the end magic
        std::vector<uint8_t> tail;
        put_u64(tail, h);
        put_u64(tail, total + kTrailerBytes);
        for (int i = 0; i < 4; ++i) tail.push_back(uint8_t(kStateEndMagic[i]));
        if (std::fwrite(tail.data(), 1, tail.size(), f) != tail.size()) { ok = false; break; }
        total += tail.size();
        if (std::fflush(f) != 0) { ok = false; break; }
#if !defined(_WIN32)
        (void) ::fsync(::fileno(f));
#endif
    } while (false);
    if (std::fclose(f) != 0) ok = false;
    if (ok) {
        // the rename IS the commit: a crash before it leaves the old sidecar (and the record
        // it belongs to) intact; after it, the new pair is durable together
        std::error_code ec;
        fs::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) {
        std::error_code ec;
        fs::remove(tmp, ec);
        return false;
    }
#if !defined(_WIN32)
    // the rename must survive a power cut: flush the directory entry, the same discipline the
    // store applies to the .bin's publish (Windows has no directory fsync — same exception)
    const size_t slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
    const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }
#endif
    bytes = total;
    return true;
}

std::string conversation_delta_describe(const SavedConversation& image, const DeltaPlan& plan) {
    const std::vector<const SavedConversation*> images = images_of(image);
    auto mib = [](uint64_t v) { return double(v) / (1024.0 * 1024.0); };
    std::string out;
    for (size_t ii = 0; ii < images.size() && ii < plan.images.size(); ++ii) {
        const SavedConversation& img = *images[ii];
        uint64_t live = 0, ck_state = 0, ck_meta = 0, kv_sent = 0, kv_full = 0;
        bool pinned = false;
        size_t ck_ids = 0;
        for (size_t i = 0; i < 5; ++i) live += state_of(img.live, i)->size();
        if (const ConversationCheckpoint* sel = record_checkpoint(img.checkpoints)) {
            for (size_t i = 0; i < 5; ++i) ck_state += state_of(*sel, i)->size();
            ck_meta = sel->ids.size() * 4 + sel->imgs.size() * 16;
            ck_ids = sel->ids.size();
            pinned = sel->pinned;
        }
        for (size_t j = 0; j < img.kv.size(); ++j)
            for (size_t part = 0; part < 5; ++part) {
                kv_full += part_of(img.kv[j], part)->size();
                kv_sent += plan.images[ii].parts[j][part].n;
            }
        const double per_tok = plan.new_tokens ? double(kv_full) / double(plan.new_tokens) : 0.0;
        const double expected = per_tok * double(plan.new_tokens - plan.keep_tokens);
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      " img%zu{live=%.1f ckpt=%.1f(+%.2f meta, ids=%zu, pinned=%d) "
                      "kv_sent=%.1f kv_expected=%.1f kv_full=%.1f MiB}",
                      ii, mib(live), mib(ck_state), mib(ck_meta), ck_ids, pinned ? 1 : 0,
                      mib(kv_sent), mib(expected), mib(kv_full));
        out += buf;
    }
    return out;
}

DeltaFold conversation_delta_apply(const std::string& path, SavedConversation& image,
                                   const std::string& state_path) {
    DeltaFold fold;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fold;
    fold.had_journal = true;
    long long fsize = 0;
    uint64_t offset = 0;
    if (std::fseek(f, 0, SEEK_END) == 0 && (fsize = std::ftell(f)) >= 0 && std::fseek(f, 0, SEEK_SET) == 0) {
        uint64_t current_len = (uint64_t) image.live.ids.size();
        const std::vector<SavedConversation*> images = images_of(image);
        const uint64_t base_geometry =
            conversation_delta_geometry_hash(image.geometry, image.layer_lo, image.layer_hi);
        // ── v3: validate the state sidecar once (magic, version, streaming hash, size, the
        // image count). Its stamp (seq + record_hash) is checked per record below and once
        // more after the loop — a record past its coverage is a torn tail (the crash window
        // between the record append and the sidecar's rename), never a silent mismatch.
        StateFileInfo sinfo;
        const bool state_ok = !state_path.empty() && state_file_probe(state_path, sinfo) &&
                              sinfo.n_images == images.size();
        uint32_t last_version = 0;
        uint64_t last_hash = 0;
        bool any_v3 = false;   // a .rst is only stale when NO v3 record applied (a v2 record
                               // appended after v3 ones must not orphan their state source)
        std::vector<uint8_t> front;
        while (offset + kHeaderBytes + kTrailerBytes <= (uint64_t) fsize) {
            // ── header: bounds, magic, version, sanity ──────────────────────────────────────
            RecordHeader rh;
            uint8_t hdr[kHeaderBytes];
            if (std::fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr) || !parse_header(hdr, sizeof(hdr), rh)) {
                fold.torn = true;   // a short or malformed header reads as a torn tail
                break;
            }
            if (rh.record_bytes > (uint64_t) fsize - offset) { fold.torn = true; break; }
            // ── chain: this record must apply to the state we hold ──────────────────────────
            if (rh.applies_to_len != current_len) {
                // a first record that does not chain is a whole stale journal (the compaction
                // crash case, or a .bin replaced under a reused serial); a later one is a
                // broken tail: either way the valid chain ends before it
                fold.dropped = fold.applied == 0;
                fold.torn = true;
                break;
            }
            // ── the anti-misalignment echoes, before anything moves ────────────────────────
            if (rh.geometry_hash != base_geometry || (rh.cvec != 0) != image.cvec ||
                rh.n_layers != image.kv.size()) {
                fold.torn = true;
                break;
            }
            // the record's image count must be exactly what the reader holds: the main plus
            // its stage files (a stage lost or replaced is a torn tail — the base restores
            // alone once the caller truncates)
            if (rh.n_images != images.size()) { fold.torn = true; break; }
            // ── the v3 chain: a light record is only as good as the sidecar behind it. The
            // sidecar must exist (state_ok), this record must be the NEXT index of the chain
            // (seq == applied + 1: no gaps, no replays), and the sidecar must cover it
            // (seq <= sinfo.seq) — the crash window between the .dlt append and the .rst
            // rename lands here as a torn tail, so the fold ends one record early and the
            // caller truncates. Nothing is mutated before this point.
            if (rh.version >= kDeltaVersion &&
                (!state_ok || rh.seq != fold.applied + 1 || rh.seq > sinfo.seq)) {
                fold.torn = true;
                break;
            }
            // ── shared region + sections' front matter ──────────────────────────────────────
            if (!read_record_front(f, rh, front)) { fold.torn = true; break; }
            ParsedShared shared;
            if (!parse_shared_region(front.data(), size_t(rh.small_bytes), rh.keep_tokens, rh.new_tokens,
                                     rh.n_imgs, shared)) {
                fold.torn = true;
                break;
            }
            // the sections walk the rest of the front buffer with one cursor; a section that
            // overruns its share, or a cursor that does not land exactly on the buffer's end,
            // is a malformed record. A v3 section carries no state block (the .rst has it).
            std::vector<ParsedSection> sections(rh.n_images);
            {
                size_t sat = size_t(rh.small_bytes);
                bool secs_ok = true;
                for (uint32_t ii = 0; ii < rh.n_images && secs_ok; ++ii)
                    secs_ok = parse_section_at(front.data(), front.size(), sat, sections[ii],
                                               rh.version >= kDeltaVersion);
                if (!secs_ok || sat != front.size()) { fold.torn = true; break; }
            }
            // ── pre-checks against the images we hold: echoes, per-layer invariants, bounds ──
            bool fits = true;
            for (uint32_t ii = 0; ii < rh.n_images && fits; ++ii) {
                const SavedConversation& img = *images[ii];
                const ParsedSection& sec = sections[ii];
                if (sec.geometry_hash != conversation_delta_geometry_hash(img.geometry, img.layer_lo,
                                                                          img.layer_hi))
                    fits = false;   // this section was built for another carve
                if (sec.layer_headers.size() != img.kv.size() * 7) fits = false;
                for (size_t j = 0; j < img.kv.size() && fits; ++j) {
                    const ConversationKv& kv = img.kv[j];
                    const int64_t* lh = &sec.layer_headers[j * 7];
                    // format, heads, head_dim, page_size, idx_dim are conversation constants;
                    // cells and pooled_rows are the new image's
                    if (lh[0] != kv.format || lh[2] != kv.heads || lh[3] != kv.head_dim || lh[4] != kv.page_size ||
                        lh[6] != kv.idx_dim)
                        fits = false;
                    for (size_t part = 0; part < 5; ++part) {
                        const uint64_t at = sec.table[j][part].at, n = sec.table[j][part].n;
                        if (at > part_of(kv, part)->size()) fits = false;
                        if (at + n < at || at + n > (1ull << 40)) fits = false;
                    }
                }
            }
            if (!fits) { fold.torn = true; break; }
            // ── the payload bytes must add up to the record's size ──────────────────────────
            const uint64_t payload_bytes = rh.record_bytes - kHeaderBytes - rh.small_bytes -
                                           rh.sections_bytes - kTrailerBytes;
            uint64_t payload_sum = 0;
            for (const auto& sec : sections)
                for (const auto& layer : sec.table)
                    for (const auto& part : layer) payload_sum += part.n;
            if (payload_sum != payload_bytes) { fold.torn = true; break; }
            // ── phase 1: hash-validate the WHOLE record before anything is mutated ─────────
            // (a corrupted record must leave every image at the last good record's state, so
            // the hash runs over a streaming read first and the payload is re-read to apply)
            uint64_t h = fnv_begin();
            h = fnv_update(h, hdr, sizeof(hdr));
            h = fnv_update(h, front.data(), front.size());
            {
                std::vector<uint8_t> chunk;
                try {
                    chunk.resize(64 * 1024);
                } catch (const std::exception&) {
                    fold.torn = true;
                    break;
                }
                bool streamed = true;
                for (uint64_t left = payload_bytes; left > 0 && streamed;) {
                    const size_t take = size_t(std::min<uint64_t>(left, chunk.size()));
                    if (std::fread(chunk.data(), 1, take, f) != take) { streamed = false; break; }
                    h = fnv_update(h, chunk.data(), take);
                    left -= take;
                }
                if (!streamed) { fold.torn = true; break; }
            }
            uint8_t trailer[kTrailerBytes];
            uint64_t stored_hash = 0, stored_bytes = 0;
            bool trailer_ok = std::fread(trailer, 1, sizeof(trailer), f) == sizeof(trailer);
            if (trailer_ok) {
                size_t at = 0;
                trailer_ok = get_u64(trailer, sizeof(trailer), at, stored_hash) &&
                             get_u64(trailer, sizeof(trailer), at, stored_bytes);
                for (int i = 0; trailer_ok && i < 4; ++i)
                    trailer_ok = trailer[kTrailerBytes - 4 + i] == uint8_t(kDeltaEndMagic[i]);
            }
            if (!trailer_ok || stored_hash != h || stored_bytes != rh.record_bytes) { fold.torn = true; break; }
            // ── phase 2: apply — ids, images (every image shares them), then per section the
            // running state (v2 records only — a v3 record's state arrives with the sidecar
            // after the loop), the checkpoints (same), the dims and the K/V payload (always) ─
            for (SavedConversation* img : images) {
                img->live.ids.resize(size_t(rh.keep_tokens));
                img->live.ids.insert(img->live.ids.end(), shared.ids_suffix.begin(), shared.ids_suffix.end());
                img->live.imgs = shared.imgs;
            }
            for (uint32_t ii = 0; ii < rh.n_images; ++ii) {
                const ParsedSection& sec = sections[ii];
                SavedConversation& img = *images[ii];
                if (rh.version < kDeltaVersion) {
                    img.live.gdn = sec.state[0];
                    img.live.ple = sec.state[1];
                    img.live.tails = sec.state[2];
                    img.live.dead = sec.state[3];
                    img.live.block_pos = sec.state[4];
                    img.checkpoints = sec.checkpoints;
                }
                for (size_t j = 0; j < img.kv.size(); ++j) {
                    const int64_t* lh = &sec.layer_headers[j * 7];
                    ConversationKv& kv = img.kv[j];
                    kv.format = int(lh[0]); kv.cells = lh[1]; kv.heads = lh[2]; kv.head_dim = lh[3];
                    kv.page_size = lh[4]; kv.pooled_rows = lh[5]; kv.idx_dim = lh[6];
                }
            }
            // the payloads, re-read from where phase 1 found them (the file did not move: the
            // writer thread is drained before any read), section by section in record order
            {
                const uint64_t payload_start = offset + kHeaderBytes + rh.small_bytes + rh.sections_bytes;
                bool ok = std::fseek(f, (long long) payload_start, SEEK_SET) == 0;
                for (uint32_t ii = 0; ok && ii < rh.n_images; ++ii) {
                    SavedConversation& img = *images[ii];
                    for (size_t j = 0; ok && j < img.kv.size(); ++j) {
                        ConversationKv& kv = img.kv[j];
                        for (size_t part = 0; ok && part < 5; ++part) {
                            const uint64_t at = sections[ii].table[j][part].at, n = sections[ii].table[j][part].n;
                            if (n == 0) continue;
                            ConversationBuffer& buf = *part_of(kv, part);
                            buf.resize(size_t(at + n));
                            ok = buf.visit(size_t(at), size_t(n), [&](uint8_t* p, size_t c, size_t) {
                                return std::fread(p, 1, c, f) == c;
                            });
                        }
                    }
                }
                if (!ok) { fold.failed = true; break; }   // the images are suspect: the caller discards them
                if (std::fseek(f, (long long) (offset + rh.record_bytes), SEEK_SET) != 0) {
                    fold.failed = true;
                    break;
                }
            }
            // applied: advance the chain (and remember which formats the chain holds)
            current_len = rh.new_tokens;
            last_version = rh.version;
            last_hash = stored_hash;
            if (rh.version >= kDeltaVersion) any_v3 = true;
            offset += rh.record_bytes;
            ++fold.applied;
        }
        // ── v3: the running state and the checkpoints arrive ONCE, from the sidecar of the
        // last record the fold applied. The stamp must land exactly on that record (seq ==
        // applied, record_hash == the record's trailer hash): anything else means the state
        // and the K/V would disagree at a byte level the hashes cannot see, so the fold is
        // failed and the caller discards the images (the files stay on disk). A fold with NO
        // v3 record at all while a sidecar sits beside the journal reports state_stale: the
        // .rst describes a generation nothing reaches — the caller deletes it. (A v2 record
        // appended AFTER v3 ones does NOT make the sidecar stale: the earlier records still
        // need it, and a tail tear could yet land the fold on one of them.)
        if (!fold.failed) {
            if (last_version >= kDeltaVersion) {
                if (!state_ok || sinfo.seq != fold.applied || sinfo.record_hash != last_hash ||
                    !state_file_load(state_path, sinfo, images))
                    fold.failed = true;
            } else if (state_ok && !any_v3) {
                fold.state_stale = true;
            }
        }
    }
    std::fclose(f);
    fold.valid_end = offset;
    fold.journal_bytes = offset;
    fold.records = uint32_t(fold.applied);
    if (fold.applied == 0 && (uint64_t) fsize > 0) fold.dropped = true;
    fold.torn = fold.torn || (uint64_t) fsize > fold.valid_end;
    return fold;
}

} // namespace strata::core
