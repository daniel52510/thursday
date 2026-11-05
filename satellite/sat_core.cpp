/*
 * THURSDAY Satellite Native C++17 Audio, AEC & OpenWakeWord Preprocessor Core
 *
 * Replaces the Python/NumPy hot loops in satellite.py, aec.py, and openwakeword:
 *   1. High-pass (80 Hz rumble cut) + soft-knee tanh mic booster (sat_boost_pcm)
 *   2. Fast SIMD-friendly dBFS & 24k<->16k linear resampler (sat_dbfs, sat_resample_scale)
 *   3. 20 ms Player block engine: 4 ms gap softening + 24 kHz music mixer & ducker
 *   4. 30 s 16 kHz AEC Reference ring buffer + SpeexDSP Echo Canceller + lag cross-correlation
 *   5. Zero-allocation OpenWakeWord streaming feature buffer + Speex NS + <0.01 ms instant reset()
 *
 * Build on Pi (aarch64):
 *   g++ -O3 -march=armv8-a+crc+simd -shared -fPIC satellite/sat_core.cpp -lspeexdsp -lpthread -o satellite/_sat_core.so
 * Build on macOS (arm64):
 *   g++ -O3 -std=c++17 -shared -fPIC satellite/sat_core.cpp -lpthread -o satellite/_sat_core.so
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <mutex>
#include <vector>

namespace {

constexpr int MIC_RATE = 16000;
constexpr int SPEEX_FRAME = 160; // 10 ms at 16 kHz
constexpr int ECHO_SET_SAMPLING_RATE = 24;
constexpr int PP_SET_DENOISE = 0;
constexpr int PP_SET_NOISE_SUPPRESS = 18;
constexpr int PP_SET_ECHO_STATE = 24;

// Dynamic loader for libspeexdsp so the same C++ file compiles and runs even on hosts
// without libspeexdsp headers linked at build time.
struct SpeexApi {
    void* handle = nullptr;
    void* (*echo_state_init)(int, int) = nullptr;
    void (*echo_state_destroy)(void*) = nullptr;
    void (*echo_state_reset)(void*) = nullptr;
    void (*echo_cancellation)(void*, const int16_t*, const int16_t*, int16_t*) = nullptr;
    int (*echo_ctl)(void*, int, void*) = nullptr;
    void* (*preprocess_state_init)(int, int) = nullptr;
    void (*preprocess_state_destroy)(void*) = nullptr;
    int (*preprocess_ctl)(void*, int, void*) = nullptr;
    int (*preprocess_run)(void*, int16_t*) = nullptr;

    SpeexApi() {
        const char* names[] = {"libspeexdsp.so.1", "libspeexdsp.so", "libspeexdsp.dylib", nullptr};
        for (int i = 0; names[i]; ++i) {
            handle = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
            if (handle) break;
        }
        if (!handle) return;
        echo_state_init = reinterpret_cast<decltype(echo_state_init)>(dlsym(handle, "speex_echo_state_init"));
        echo_state_destroy = reinterpret_cast<decltype(echo_state_destroy)>(dlsym(handle, "speex_echo_state_destroy"));
        echo_state_reset = reinterpret_cast<decltype(echo_state_reset)>(dlsym(handle, "speex_echo_state_reset"));
        echo_cancellation = reinterpret_cast<decltype(echo_cancellation)>(dlsym(handle, "speex_echo_cancellation"));
        echo_ctl = reinterpret_cast<decltype(echo_ctl)>(dlsym(handle, "speex_echo_ctl"));
        preprocess_state_init = reinterpret_cast<decltype(preprocess_state_init)>(dlsym(handle, "speex_preprocess_state_init"));
        preprocess_state_destroy = reinterpret_cast<decltype(preprocess_state_destroy)>(dlsym(handle, "speex_preprocess_state_destroy"));
        preprocess_ctl = reinterpret_cast<decltype(preprocess_ctl)>(dlsym(handle, "speex_preprocess_ctl"));
        preprocess_run = reinterpret_cast<decltype(preprocess_run)>(dlsym(handle, "speex_preprocess_run"));
    }

    bool ok() const {
        return echo_state_init && echo_cancellation && echo_ctl && preprocess_state_init && preprocess_ctl && preprocess_run;
    }
};

SpeexApi& speex_api() {
    static SpeexApi api;
    return api;
}

inline int16_t clip_i16(float x) {
    if (x >= 32767.0f) return 32767;
    if (x <= -32768.0f) return -32768;
    return static_cast<int16_t>(std::lrintf(x));
}

} // namespace

extern "C" {

// -----------------------------------------------------------------------------
// 1. High-Pass Rumble Filter (80 Hz) + Soft-Knee Far-Field Mic Booster
// -----------------------------------------------------------------------------

void sat_boost_pcm(const int16_t* in_pcm, int16_t* out_pcm, int n_samples, float gain, float* hp_state) {
    if (gain == 1.0f && !hp_state) {
        std::memcpy(out_pcm, in_pcm, n_samples * sizeof(int16_t));
        return;
    }
    // 1st-order 80 Hz high-pass at 16 kHz: removes DC offset and HVAC/room rumble
    // before boosting, so across-the-room speech gets clean headroom.
    constexpr float alpha = 0.9695f;
    float prev_x = hp_state ? hp_state[0] : 0.0f;
    float prev_y = hp_state ? hp_state[1] : 0.0f;
    constexpr float knee = 20000.0f;
    constexpr float headroom = 12767.0f;
    constexpr float inv_headroom = 1.0f / headroom;

    for (int i = 0; i < n_samples; ++i) {
        float x = static_cast<float>(in_pcm[i]);
        float y = x;
        if (hp_state) {
            y = alpha * (prev_y + x - prev_x);
            prev_x = x;
            prev_y = y;
        }
        float s = y * gain;
        if (s > knee) {
            s = knee + headroom * std::tanh((s - knee) * inv_headroom);
        } else if (s < -knee) {
            s = -knee + headroom * std::tanh((s + knee) * inv_headroom);
        }
        out_pcm[i] = clip_i16(s);
    }
    if (hp_state) {
        hp_state[0] = prev_x;
        hp_state[1] = prev_y;
    }
}

double sat_dbfs(const int16_t* pcm, int n_samples) {
    if (n_samples <= 0) return -120.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        double v = static_cast<double>(pcm[i]);
        sum_sq += v * v;
    }
    double mean_sq = sum_sq / static_cast<double>(n_samples);
    return 10.0 * std::log10(mean_sq / (32768.0 * 32768.0) + 1e-12);
}

// -----------------------------------------------------------------------------
// 2. Resampling, Scaling, Gap Softening, and Music Mixing for Player
// -----------------------------------------------------------------------------

int sat_resample_scale(const int16_t* in_pcm, int in_samples, int in_rate,
                       int16_t* out_pcm, int max_out_samples, int out_rate, float gain) {
    if (in_samples <= 0) return 0;
    if (in_rate == out_rate) {
        int n = std::min(in_samples, max_out_samples);
        if (gain == 1.0f) {
            std::memcpy(out_pcm, in_pcm, n * sizeof(int16_t));
        } else {
            for (int i = 0; i < n; ++i) {
                out_pcm[i] = clip_i16(static_cast<float>(in_pcm[i]) * gain);
            }
        }
        return n;
    }
    int out_n = std::min(static_cast<int>(static_cast<int64_t>(in_samples) * out_rate / in_rate), max_out_samples);
    double step = static_cast<double>(in_rate) / static_cast<double>(out_rate);
    for (int i = 0; i < out_n; ++i) {
        double pos = i * step;
        int idx = static_cast<int>(pos);
        float frac = static_cast<float>(pos - idx);
        float s0 = static_cast<float>(in_pcm[idx]);
        float s1 = static_cast<float>(in_pcm[std::min(idx + 1, in_samples - 1)]);
        float val = (s0 + frac * (s1 - s0)) * gain;
        out_pcm[i] = clip_i16(val);
    }
    return out_n;
}

void sat_soften_block(int16_t* pcm, int n_samples, int fade_samples, int was_dry, int runs_dry) {
    int n = std::min(fade_samples, n_samples);
    if (n <= 1) return;
    float inv = 1.0f / static_cast<float>(n - 1);
    if (was_dry) {
        for (int i = 0; i < n; ++i) {
            pcm[i] = static_cast<int16_t>(static_cast<float>(pcm[i]) * (static_cast<float>(i) * inv));
        }
    }
    if (runs_dry) {
        for (int i = 0; i < n; ++i) {
            float g = static_cast<float>(n - 1 - i) * inv;
            pcm[n_samples - n + i] = static_cast<int16_t>(static_cast<float>(pcm[n_samples - n + i]) * g);
        }
    }
}

float sat_mix_music_block(const int16_t* voice_pcm, const int16_t* music_pcm, int music_valid_samples,
                          int16_t* out_pcm, int block_samples, int fade_out_music,
                          float cur_mgain, float target_mgain) {
    float next_mgain = cur_mgain + std::max(-0.1f, std::min(0.1f, target_mgain - cur_mgain));
    float inv_block = block_samples > 1 ? 1.0f / static_cast<float>(block_samples - 1) : 0.0f;
    float inv_valid = (fade_out_music && music_valid_samples > 1) ? 1.0f / static_cast<float>(music_valid_samples - 1) : 0.0f;

    for (int i = 0; i < block_samples; ++i) {
        float m = (i < music_valid_samples) ? static_cast<float>(music_pcm[i]) : 0.0f;
        if (fade_out_music && i < music_valid_samples && music_valid_samples > 1) {
            m *= static_cast<float>(music_valid_samples - 1 - i) * inv_valid;
        }
        float ramp = cur_mgain + (next_mgain - cur_mgain) * (static_cast<float>(i) * inv_block);
        float mixed = static_cast<float>(voice_pcm[i]) + m * ramp;
        out_pcm[i] = clip_i16(mixed);
    }
    return next_mgain;
}

// -----------------------------------------------------------------------------
// 3. C++ Reference Ring Buffer & SpeexDSP Echo Canceller
// -----------------------------------------------------------------------------

struct SatReference {
    static constexpr int RING_SAMPLES = 30 * MIC_RATE; // 30 s at 16 kHz
    std::vector<int16_t> ring;
    int64_t mic_n = 0;
    std::deque<double> anchors;
    double t0 = 0.0;
    bool has_t0 = false;
    int64_t active_until = 0;
    std::mutex mu;

    SatReference() : ring(RING_SAMPLES, 0) {}

    int64_t mic_arrived(int samples, double now_mono) {
        std::lock_guard<std::mutex> lock(mu);
        int64_t start = mic_n;
        mic_n += samples;
        double anchor = now_mono - static_cast<double>(mic_n) / MIC_RATE;
        anchors.push_back(anchor);
        if (anchors.size() > 64) anchors.pop_front();
        t0 = *std::min_element(anchors.begin(), anchors.end());
        has_t0 = true;
        return start;
    }

    int64_t index_at(double at_mono, int* ok) {
        std::lock_guard<std::mutex> lock(mu);
        if (!has_t0) {
            if (ok) *ok = 0;
            return 0;
        }
        if (ok) *ok = 1;
        return std::llround((at_mono - t0) * MIC_RATE);
    }

    void add_samples_locked(const int16_t* x, int len, int64_t i) {
        if (i < mic_n) {
            int64_t drop = mic_n - i;
            if (drop >= len) return;
            x += drop;
            len -= static_cast<int>(drop);
            i = mic_n;
        }
        len = std::min(len, RING_SAMPLES - MIC_RATE);
        if (len <= 0) return;
        int j = static_cast<int>(((i % RING_SAMPLES) + RING_SAMPLES) % RING_SAMPLES);
        int k = std::min(len, RING_SAMPLES - j);
        std::memcpy(ring.data() + j, x, k * sizeof(int16_t));
        if (len > k) {
            std::memcpy(ring.data(), x + k, (len - k) * sizeof(int16_t));
        }
        active_until = std::max(active_until, i + len);
    }

    void add_at(const int16_t* pcm, int n_samples, int rate, int64_t index) {
        std::vector<int16_t> tmp;
        const int16_t* ptr = pcm;
        int len = n_samples;
        if (rate != MIC_RATE && n_samples > 0) {
            int out_n = static_cast<int>(static_cast<int64_t>(n_samples) * MIC_RATE / rate);
            tmp.resize(out_n);
            len = sat_resample_scale(pcm, n_samples, rate, tmp.data(), out_n, MIC_RATE, 1.0f);
            ptr = tmp.data();
        }
        std::lock_guard<std::mutex> lock(mu);
        add_samples_locked(ptr, len, index);
    }

    void add_time(const int16_t* pcm, int n_samples, int rate, double at_mono) {
        std::vector<int16_t> tmp;
        const int16_t* ptr = pcm;
        int len = n_samples;
        if (rate != MIC_RATE && n_samples > 0) {
            int out_n = static_cast<int>(static_cast<int64_t>(n_samples) * MIC_RATE / rate);
            tmp.resize(out_n);
            len = sat_resample_scale(pcm, n_samples, rate, tmp.data(), out_n, MIC_RATE, 1.0f);
            ptr = tmp.data();
        }
        std::lock_guard<std::mutex> lock(mu);
        if (!has_t0) return;
        int64_t idx = std::llround((at_mono - t0) * MIC_RATE);
        add_samples_locked(ptr, len, idx);
    }

    void cut(double at_mono) {
        std::lock_guard<std::mutex> lock(mu);
        if (!has_t0) return;
        int64_t i = std::max<int64_t>(std::llround((at_mono - t0) * MIC_RATE), mic_n);
        if (i < active_until) {
            for (int64_t p = i; p < active_until; ++p) {
                ring[((p % RING_SAMPLES) + RING_SAMPLES) % RING_SAMPLES] = 0;
            }
        }
        active_until = std::min(active_until, i);
    }

    void read(int64_t start, int n, int16_t* out) {
        std::lock_guard<std::mutex> lock(mu);
        for (int k = 0; k < n; ++k) {
            int idx = static_cast<int>((((start + k) % RING_SAMPLES) + RING_SAMPLES) % RING_SAMPLES);
            out[k] = ring[idx];
            ring[idx] = 0;
        }
    }
};

struct SatCanceller {
    SatReference* ref = nullptr;
    int tail = 1920;
    void* echo_st = nullptr;
    void* pp_st = nullptr;

    SatCanceller(SatReference* r, int filter_ms) : ref(r) {
        tail = filter_ms * MIC_RATE / 1000;
        auto& api = speex_api();
        if (!api.ok()) return;
        echo_st = api.echo_state_init(SPEEX_FRAME, tail);
        int rate = MIC_RATE;
        api.echo_ctl(echo_st, ECHO_SET_SAMPLING_RATE, &rate);
        pp_st = api.preprocess_state_init(SPEEX_FRAME, MIC_RATE);
        int off = 0;
        api.preprocess_ctl(pp_st, PP_SET_DENOISE, &off);
        api.preprocess_ctl(pp_st, PP_SET_ECHO_STATE, echo_st);
    }

    ~SatCanceller() {
        auto& api = speex_api();
        if (pp_st && api.preprocess_state_destroy) api.preprocess_state_destroy(pp_st);
        if (echo_st && api.echo_state_destroy) api.echo_state_destroy(echo_st);
    }

    bool ok() const { return echo_st != nullptr && pp_st != nullptr; }

    // Returns:
    //   0 = inactive (past active_until + tail; mic copied straight through)
    //   1 = active (cancelled; ref_out filled with reference frames, stats filled)
    int process_chunk(const int16_t* mic_in, int16_t* mic_out, int16_t* ref_out, int n_samples,
                      double now_mono, double* stats_out) {
        int64_t start = ref->mic_arrived(n_samples, now_mono);
        int64_t active_until = 0;
        {
            std::lock_guard<std::mutex> lock(ref->mu);
            active_until = ref->active_until;
        }
        if (start > active_until + tail || !ok()) {
            std::memcpy(mic_out, mic_in, n_samples * sizeof(int16_t));
            return 0;
        }
        ref->read(start, n_samples, ref_out);
        auto& api = speex_api();
        for (int f = 0; f + SPEEX_FRAME <= n_samples; f += SPEEX_FRAME) {
            api.echo_cancellation(echo_st, mic_in + f, ref_out + f, mic_out + f);
            api.preprocess_run(pp_st, mic_out + f);
        }
        bool any_ref = false;
        for (int i = 0; i < n_samples; ++i) {
            if (ref_out[i] != 0) {
                any_ref = true;
                break;
            }
        }
        if (stats_out) {
            stats_out[0] = any_ref ? 1.0 : 0.0;
            if (any_ref) {
                double mic_e = 0.0, out_e = 0.0;
                for (int i = 0; i < n_samples; ++i) {
                    double m = static_cast<double>(mic_in[i]);
                    double o = static_cast<double>(mic_out[i]);
                    mic_e += m * m;
                    out_e += o * o;
                }
                stats_out[1] = mic_e;
                stats_out[2] = out_e;
                stats_out[3] = 10.0 * std::log10(mic_e / n_samples / (32768.0 * 32768.0) + 1e-12);
                stats_out[4] = 10.0 * std::log10(out_e / n_samples / (32768.0 * 32768.0) + 1e-12);
            } else {
                stats_out[1] = stats_out[2] = stats_out[3] = stats_out[4] = 0.0;
            }
        }
        return 1;
    }
};

void* sat_ref_create() { return new SatReference(); }
void sat_ref_destroy(void* p) { delete static_cast<SatReference*>(p); }
int64_t sat_ref_mic_arrived(void* p, int samples, double now_mono) {
    return static_cast<SatReference*>(p)->mic_arrived(samples, now_mono);
}
int64_t sat_ref_index_at(void* p, double at_mono, int* ok) {
    return static_cast<SatReference*>(p)->index_at(at_mono, ok);
}
void sat_ref_add_at(void* p, const int16_t* pcm, int n_samples, int rate, int64_t index) {
    static_cast<SatReference*>(p)->add_at(pcm, n_samples, rate, index);
}
void sat_ref_add_time(void* p, const int16_t* pcm, int n_samples, int rate, double at_mono) {
    static_cast<SatReference*>(p)->add_time(pcm, n_samples, rate, at_mono);
}
void sat_ref_cut(void* p, double at_mono) {
    static_cast<SatReference*>(p)->cut(at_mono);
}
void sat_ref_read(void* p, int64_t start, int n, int16_t* out) {
    static_cast<SatReference*>(p)->read(start, n, out);
}

void* sat_aec_create(void* ref_ptr, int filter_ms) {
    auto* c = new SatCanceller(static_cast<SatReference*>(ref_ptr), filter_ms);
    if (!c->ok()) {
        delete c;
        return nullptr;
    }
    return c;
}
void sat_aec_destroy(void* p) { delete static_cast<SatCanceller*>(p); }
int sat_aec_process(void* p, const int16_t* mic_in, int16_t* mic_out, int16_t* ref_out,
                    int n_samples, double now_mono, double* stats_out) {
    return static_cast<SatCanceller*>(p)->process_chunk(mic_in, mic_out, ref_out, n_samples, now_mono, stats_out);
}

// -----------------------------------------------------------------------------
// 4. C++ OpenWakeWord Streaming Preprocessor & <0.01 ms Instant Reset
// -----------------------------------------------------------------------------

struct SatWakePrep {
    float raw_tail[480];            // 30 ms overlap before each 1280-sample chunk
    float melspec_in[1760];         // [1, 1760] float32 input for melspectrogram.onnx
    float mel_window[76 * 32];      // [1, 76, 32, 1] float32 input for embedding_model.onnx
    float feat_window[16 * 96];     // [1, 16, 96] float32 input for wakeword.onnx
    float init_feat_window[16 * 96];// baseline features captured once at init for <0.01ms reset()
    int frames_since_reset = 0;
    void* ns_st = nullptr;

    SatWakePrep(bool enable_speex_ns) {
        std::memset(raw_tail, 0, sizeof(raw_tail));
        std::memset(melspec_in, 0, sizeof(melspec_in));
        std::fill(std::begin(mel_window), std::end(mel_window), 1.0f);
        std::memset(feat_window, 0, sizeof(feat_window));
        std::memset(init_feat_window, 0, sizeof(init_feat_window));
        if (enable_speex_ns) {
            auto& api = speex_api();
            if (api.preprocess_state_init && api.preprocess_ctl && api.preprocess_run) {
                ns_st = api.preprocess_state_init(SPEEX_FRAME, MIC_RATE);
                if (ns_st) {
                    int on = 1;
                    int suppress_db = -15;
                    api.preprocess_ctl(ns_st, PP_SET_DENOISE, &on);
                    api.preprocess_ctl(ns_st, PP_SET_NOISE_SUPPRESS, &suppress_db);
                }
            }
        }
    }

    ~SatWakePrep() {
        auto& api = speex_api();
        if (ns_st && api.preprocess_state_destroy) {
            api.preprocess_state_destroy(ns_st);
        }
    }

    void set_init_features(const float* feats_16x96) {
        std::memcpy(init_feat_window, feats_16x96, sizeof(init_feat_window));
        std::memcpy(feat_window, feats_16x96, sizeof(feat_window));
    }

    void reset() {
        std::memset(raw_tail, 0, sizeof(raw_tail));
        std::fill(std::begin(mel_window), std::end(mel_window), 1.0f);
        std::memcpy(feat_window, init_feat_window, sizeof(feat_window));
        frames_since_reset = 0;
    }

    // Prepares [1, 1760] float32 input from 1280 int16 samples (running Speex NS in C++ if enabled).
    const float* prepare_melspec_input(const int16_t* pcm_1280) {
        int16_t clean[1280];
        const int16_t* src = pcm_1280;
        if (ns_st) {
            std::memcpy(clean, pcm_1280, sizeof(clean));
            auto& api = speex_api();
            for (int f = 0; f < 1280; f += SPEEX_FRAME) {
                api.preprocess_run(ns_st, clean + f);
            }
            src = clean;
        }
        std::memcpy(melspec_in, raw_tail, 480 * sizeof(float));
        for (int i = 0; i < 1280; ++i) {
            melspec_in[480 + i] = static_cast<float>(src[i]);
        }
        std::memcpy(raw_tail, melspec_in + 1280, 480 * sizeof(float));
        return melspec_in;
    }

    // Pushes `n_frames` (8 for 1760 samples) of 32 mel bins, applies `x / 10.0f + 2.0f`,
    // and returns the [1, 76, 32, 1] float32 window.
    const float* push_mels(const float* raw_mels, int n_frames) {
        if (n_frames <= 0) return mel_window;
        if (n_frames >= 76) {
            const float* tail = raw_mels + (n_frames - 76) * 32;
            for (int i = 0; i < 76 * 32; ++i) {
                mel_window[i] = tail[i] * 0.1f + 2.0f;
            }
        } else {
            int keep = (76 - n_frames) * 32;
            std::memmove(mel_window, mel_window + n_frames * 32, keep * sizeof(float));
            int add = n_frames * 32;
            for (int i = 0; i < add; ++i) {
                mel_window[keep + i] = raw_mels[i] * 0.1f + 2.0f;
            }
        }
        return mel_window;
    }

    // Pushes 1 embedding vector of 96 floats and returns the [1, 16, 96] float32 window.
    const float* push_embedding(const float* emb_96, int* out_ready) {
        std::memmove(feat_window, feat_window + 96, 15 * 96 * sizeof(float));
        std::memcpy(feat_window + 15 * 96, emb_96, 96 * sizeof(float));
        ++frames_since_reset;
        if (out_ready) {
            // Match openWakeWord's 5-frame warmup suppression after reset()
            *out_ready = (frames_since_reset > 5) ? 1 : 0;
        }
        return feat_window;
    }
};

void* sat_wake_create(int enable_speex_ns) {
    return new SatWakePrep(enable_speex_ns != 0);
}
void sat_wake_destroy(void* p) {
    delete static_cast<SatWakePrep*>(p);
}
void sat_wake_set_init_features(void* p, const float* feats_16x96) {
    static_cast<SatWakePrep*>(p)->set_init_features(feats_16x96);
}
void sat_wake_reset(void* p) {
    static_cast<SatWakePrep*>(p)->reset();
}
const float* sat_wake_prepare_melspec(void* p, const int16_t* pcm_1280) {
    return static_cast<SatWakePrep*>(p)->prepare_melspec_input(pcm_1280);
}
const float* sat_wake_push_mels(void* p, const float* raw_mels, int n_frames) {
    return static_cast<SatWakePrep*>(p)->push_mels(raw_mels, n_frames);
}
const float* sat_wake_push_emb(void* p, const float* emb_96, int* out_ready) {
    return static_cast<SatWakePrep*>(p)->push_embedding(emb_96, out_ready);
}

} // extern "C"
