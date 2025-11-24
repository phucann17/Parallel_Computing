#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <iomanip>

// Configuration setups
const int MAP_SIZE = 4000; // 4000x4000
const double CELL_SIZE = 10.0; // 10 meters
const double CENTER = 2000.0; // Center coordinate
const double YIELD_W = 5000000000; // 5000 kilotons
const double SPEED_SOUND = 343.0; // Speed of sound in m/s
const int SIM_DURATION = 100; // 100 seconds
const int NUM_THREADS = 4;

// Kingery-Bulmash Coefficient
const double C[] = {
    2.611369, -1.690128, 0.00805, 0.336743,
    -0.005162, -0.080923, -0.004785, 0.007930, 0.000768
};

double calculate_pressure_at_cell(int r, int c) {
    double dy = (r - CENTER) * CELL_SIZE;
    double dx = (c - CENTER) * CELL_SIZE;
    // Distance from the blast (meters)
    double R = sqrt(dx * dx + dy * dy);

    if (R < 1.0) R = 1.0; // Prevent divide 0 issues

    double t_arrival = R / SPEED_SOUND;
    // if (t_arrival > SIM_DURATION) return 0.0;

    // Calculate Scaled Distance Z = R * W^(-1/3)
    double Z = R * pow(YIELD_W, -1.0 / 3.0); 
    // Calculate intermediate value U
    double U = -0.21436 + 1.35034 * std::log10(Z);
    // Calculate log10(Pso)
    double log10Pso = 0.0;
    for (int i = 0; i <= 8; ++i) {
        log10Pso += C[i] * pow(U, i);
    }

    return pow(10.0, log10Pso); // Pressure in kPa
}

bool verify_results(const std::vector<std::vector<double>>& seq, 
                    const std::vector<std::vector<double>>& par) {
    int error_count = 0;
    for (int i = 0; i < MAP_SIZE; ++i) {
        for (int j = 0; j < MAP_SIZE; ++j) {
            if (std::abs(seq[i][j] - par[i][j]) > 1e-9) {
                if (error_count < 5) { // Chỉ in 5 lỗi đầu tiên để debug
                    std::cout << "Mismatch at [" << i << "][" << j << "]: "
                              << "Seq=" << seq[i][j] << " vs Par=" << par[i][j] << "\n";
                }
                error_count++;
            }
        }
    }
    
    if (error_count > 0) {
        std::cout << "FAILED: Found " << error_count << " mismatches!\n";
        return false;
    }
    std::cout << "PASSED: All " << MAP_SIZE*MAP_SIZE << " cells match perfectly.\n";
    return true;
}

// Work Pool Model
struct Task {
    int start_row;
    int end_row;
    Task(int s, int e) : start_row(s), end_row(e) {}
};

class TaskQueue {
    private:
        std::mutex mtx;
        std::queue<Task*> tasks;
        std::condition_variable cv;
        bool finished = false;
        static TaskQueue* instance;
    public:
        void enqueue(Task* task) {
            std::lock_guard<std::mutex> lock(mtx);
            tasks.push(task);
            cv.notify_one(); // awake one waiting worker
        }
        Task* dequeue() {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return !tasks.empty() || finished; });
            if (tasks.empty()) return nullptr; // done or no more tasks

            Task* task = tasks.front();
            tasks.pop();
            return task;
        }

        void finish() {
            std::lock_guard<std::mutex> lock(mtx);
            finished = true;
            cv.notify_all();
        }

        void reset() {
            std::lock_guard<std::mutex> lock(mtx);
            finished = false;
            while (!tasks.empty()) {
                delete tasks.front();
                tasks.pop();
            }
        }

        static TaskQueue* get() {
            if (instance == nullptr) {
                instance = new TaskQueue();
            }
            return instance;
        }
};

TaskQueue* TaskQueue::instance = nullptr;

class Worker {
    private:
        std::thread t;
        int id;
        std::vector<std::vector<double>>* shared_map;
        void run() {
            while (true) {
                Task* task = TaskQueue::get()->dequeue();
                if (task == nullptr) break; // no more tasks
            
                for (int r = task->start_row; r < task->end_row; ++r) {
                    for (int c = 0; c < MAP_SIZE; ++c) {
                        double pressure = calculate_pressure_at_cell(r, c);
                        (*shared_map)[r][c] = pressure;
                    }
                }

                delete task;
            }
        }
    public:
        Worker (int id, std::vector<std::vector<double>>* map_ptr) : id(id), shared_map(map_ptr) {
            t = std::thread(&Worker::run, this);
        }

        void join() {
            if (t.joinable()) t.join();
        }
};

int main() {
    auto map_sequence = new std::vector<std::vector<double>>(MAP_SIZE, std::vector<double>(MAP_SIZE, 0.0));
    auto map_parallel = new std::vector<std::vector<double>>(MAP_SIZE, std::vector<double>(MAP_SIZE, 0.0));

    std::cout << "=== NUCLEAR BLAST SIMULATION (" << MAP_SIZE << "x" << MAP_SIZE << ") ===\n";
    std::cout << "Threads available: " << NUM_THREADS << "\n\n";    

    std::cout << "1. Running Sequential implementation...\n";
    auto start_seq = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < MAP_SIZE; ++r) {
        for (int c = 0; c < MAP_SIZE; ++c) {
            double pressure = calculate_pressure_at_cell(r, c);
            (*map_sequence)[r][c] = pressure;
        }
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;
    std::cout << "Sequential simulation completed in " << duration_seq.count() << " seconds.\n\n";

    std::cout << "2. Running Parallel implementation with Work Pool Model...\n";
    TaskQueue::get()->reset();
    auto start_par = std::chrono::high_resolution_clock::now();
    std::vector<Worker*> workers;
    for (int i = 0; i < NUM_THREADS; ++i) {
        workers.push_back(new Worker(i, map_parallel));
    }
    int chunk_size = 50;
    for (int r = 0; r < MAP_SIZE; r += chunk_size) {
        int end_row = std::min(r + chunk_size, MAP_SIZE);
        TaskQueue::get()->enqueue(new Task(r, end_row));
    }
    TaskQueue::get()->finish();
    for (auto& worker : workers) {
        worker->join();
        delete worker;
    }
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_par = end_par - start_par;
    
    std::cout << "=== COMPARISON ===\n";
    std::cout << "Sequential Time: " << duration_seq.count() << "s\n";
    std::cout << "Parallel Time:   " << duration_par.count() << "s\n";
    std::cout << "Speedup:         " << duration_seq.count() / duration_par.count() << "x\n";

    double val_seq = (*map_sequence)[2050][2050];
    double val_par = (*map_parallel)[2050][2050];
    std::cout << "Verifying random cell near center (2050, 2050):\n";
    std::cout << "Sequence Value: " << val_seq << "\n";
    std::cout << "Parallel Value: " << val_par << "\n";
    if (std::abs(val_seq - val_par) < 1e-9) {
        std::cout << "Result: MATCHED (Correct implementation).\n";
    } else {
        std::cout << "Result: MISMATCHED (Check synchronization).\n";
    }
    // verify_results(*map_sequence, *map_parallel);
    delete map_sequence;
    delete map_parallel;

    return 0;
}