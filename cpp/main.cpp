// hb_audio: ONNX Runtime (CPU) C++ worker for the audio model. One process = one loaded model; commands arrive on stdin.
//   hb_audio MODEL.onnx THREADS
// stdin : JOB <file> <warmup> <repeat> <print_logits>      or      QUIT
// JOB file (binary, little endian): int32 n_batches, then per batch: int32 B, int32 T, float32[B*T] (already normalised audio).
// stdout: READY once; per JOB: [LOGITS a b ...] one line per item (only if print_logits=1, taken from the first untimed pass),
//         then one "TIME ms1 ms2 ..." line per timed pass (one number per batch), then END. Failures print "ERROR text" then END.
// `warmup` untimed passes run first (at least one if print_logits=1). Only session.Run() is inside the timer.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Batch { int32_t B = 0, T = 0; std::vector<float> x; };

static std::vector<Batch> read_job(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n < 1) throw std::runtime_error("job file has no batches");
    std::vector<Batch> v(static_cast<size_t>(n));
    for (auto& b : v) {
        f.read(reinterpret_cast<char*>(&b.B), 4);
        f.read(reinterpret_cast<char*>(&b.T), 4);
        if (!f || b.B < 1 || b.T < 1) throw std::runtime_error("bad batch header in job file");
        b.x.resize(static_cast<size_t>(b.B) * static_cast<size_t>(b.T));
        f.read(reinterpret_cast<char*>(b.x.data()), static_cast<std::streamsize>(b.x.size() * sizeof(float)));
        if (!f) throw std::runtime_error("job file is truncated");
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: hb_audio MODEL.onnx THREADS\n"; return 2; }
    try {
        const int threads = std::max(1, std::atoi(argv[2]));
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "hb_audio");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(threads);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  // same default as the Python session
        Ort::Session session(env, argv[1], so);
        Ort::AllocatorWithDefaultOptions alloc;
        const std::string in_name = session.GetInputNameAllocated(0, alloc).get();
        const std::string out_name = session.GetOutputNameAllocated(0, alloc).get();
        const char* in_names[] = {in_name.c_str()};
        const char* out_names[] = {out_name.c_str()};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::cout << "READY\n" << std::flush;

        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream cmd(line);
            std::string op;
            cmd >> op;
            if (op == "QUIT") break;
            if (op != "JOB") continue;
            try {
                std::string path;
                int warmup = 0, repeat = 0, print_logits = 0;
                cmd >> path >> warmup >> repeat >> print_logits;
                std::vector<Batch> batches = read_job(path);
                std::vector<Ort::Value> inputs;  // tensors are built once, outside every timer
                for (auto& b : batches) {
                    const int64_t shape[2] = {b.B, b.T};
                    inputs.push_back(Ort::Value::CreateTensor<float>(mem, b.x.data(), b.x.size(), shape, 2));
                }
                std::ostringstream logits_text;
                logits_text << std::setprecision(9);
                auto run_once = [&](size_t i, bool collect, double* ms) {
                    const auto t0 = std::chrono::steady_clock::now();
                    auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &inputs[i], 1, out_names, 1);
                    const auto t1 = std::chrono::steady_clock::now();
                    if (ms) *ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (collect) {
                        const auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
                        const size_t C = static_cast<size_t>(shape.at(1));
                        const float* lg = out[0].GetTensorData<float>();
                        for (int32_t b = 0; b < batches[i].B; ++b) {
                            logits_text << "LOGITS";
                            for (size_t c = 0; c < C; ++c) logits_text << " " << lg[static_cast<size_t>(b) * C + c];
                            logits_text << "\n";
                        }
                    }
                };
                const int untimed = std::max(warmup, print_logits ? 1 : 0);
                for (int p = 0; p < untimed; ++p)
                    for (size_t i = 0; i < batches.size(); ++i) run_once(i, print_logits && p == 0, nullptr);
                std::ostringstream times_text;
                times_text << std::setprecision(9);
                for (int r = 0; r < repeat; ++r) {
                    times_text << "TIME";
                    std::vector<double> ms(batches.size());
                    for (size_t i = 0; i < batches.size(); ++i) run_once(i, false, &ms[i]);
                    for (double m : ms) times_text << " " << m;  // printed only after the whole pass
                    times_text << "\n";
                }
                std::cout << logits_text.str() << times_text.str();
            } catch (const std::exception& e) {
                std::cout << "ERROR " << e.what() << "\n";
            }
            std::cout << "END\n" << std::flush;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cout << "ERROR " << e.what() << "\nEND\n" << std::flush;
        return 1;
    }
}
