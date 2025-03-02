#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <thread>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <cuda_runtime.h>
#include <cuda.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Memory access tracking for LRU implementation
struct MemoryAccessInfo {
    std::chrono::steady_clock::time_point lastAccess;
    size_t accessCount;
    bool dirtyFlag; // Indicates if CPU memory has been modified and needs sync with GPU
};

// Memory location enum
enum class MemoryLocation {
    CPU,
    GPU,
    BOTH
};

// Memory block structure
struct MemoryBlock {
    void* cpuPtr;
    void* gpuPtr;
    size_t size;
    MemoryLocation location;
    std::string identifier;
    
    MemoryBlock() : cpuPtr(nullptr), gpuPtr(nullptr), size(0), location(MemoryLocation::CPU) {}
    
    ~MemoryBlock() {
        if (cpuPtr) free(cpuPtr);
        if (gpuPtr) CUDA_CHECK(cudaFree(gpuPtr));
    }
    
    // Final memory stats
    memoryManager.printMemoryStats();
}

// Memory pool for pre-allocated memory blocks
class MemoryPool {
private:
    HybridMemoryManager& memoryManager;
    std::vector<std::string> availableBlocks;
    std::vector<std::string> inUseBlocks;
    size_t blockSize;
    std::mutex poolMutex;
    
public:
    MemoryPool(HybridMemoryManager& manager, size_t numBlocks, size_t sizePerBlock)
        : memoryManager(manager), blockSize(sizePerBlock) {
        
        // Pre-allocate blocks
        for (size_t i = 0; i < numBlocks; ++i) {
            std::string blockId = "pool_block_" + std::to_string(i);
            if (memoryManager.allocate(blockId, blockSize, MemoryLocation::CPU)) {
                availableBlocks.push_back(blockId);
            }
        }
    }
    
    ~MemoryPool() {
        // Free all blocks
        for (const auto& blockId : availableBlocks) {
            memoryManager.free(blockId);
        }
        
        for (const auto& blockId : inUseBlocks) {
            memoryManager.free(blockId);
        }
    }
    
    // Acquire a block from the pool
    std::string acquireBlock() {
        std::lock_guard<std::mutex> lock(poolMutex);
        
        if (availableBlocks.empty()) {
            return ""; // No blocks available
        }
        
        std::string blockId = availableBlocks.back();
        availableBlocks.pop_back();
        inUseBlocks.push_back(blockId);
        
        return blockId;
    }
    
    // Release a block back to the pool
    bool releaseBlock(const std::string& blockId) {
        std::lock_guard<std::mutex> lock(poolMutex);
        
        auto it = std::find(inUseBlocks.begin(), inUseBlocks.end(), blockId);
        if (it == inUseBlocks.end()) {
            return false; // Block not found in the in-use list
        }
        
        inUseBlocks.erase(it);
        availableBlocks.push_back(blockId);
        
        return true;
    }
    
    // Get current pool statistics
    void getPoolStats(size_t& available, size_t& inUse) {
        std::lock_guard<std::mutex> lock(poolMutex);
        
        available = availableBlocks.size();
        inUse = inUseBlocks.size();
    }
};

// Advanced cache policy that combines LRU and frequency
class AdvancedCachePolicy : public EvictionPolicy {
private:
    float frequencyWeight; // Weight for frequency component (0.0 to 1.0)
    
public:
    AdvancedCachePolicy(float freqWeight = 0.5f) : frequencyWeight(freqWeight) {}
    
    std::string selectVictim(const std::unordered_map<std::string, MemoryAccessInfo>& accessInfo) override {
        std::string victim;
        float lowestScore = std::numeric_limits<float>::max();
        
        auto now = std::chrono::steady_clock::now();
        
        // Find max access count for normalization
        size_t maxAccessCount = 1; // Avoid division by zero
        for (const auto& entry : accessInfo) {
            maxAccessCount = std::max(maxAccessCount, entry.second.accessCount);
        }
        
        // Find oldest access for normalization
        auto oldestAccess = now;
        for (const auto& entry : accessInfo) {
            if (entry.second.lastAccess < oldestAccess) {
                oldestAccess = entry.second.lastAccess;
            }
        }
        
        // Calculate time range in seconds
        double timeRangeSeconds = std::chrono::duration<double>(now - oldestAccess).count();
        if (timeRangeSeconds < 0.1) timeRangeSeconds = 0.1; // Avoid division by very small numbers
        
        for (const auto& entry : accessInfo) {
            // Normalize access count (0.0 to 1.0, higher is more frequent)
            float normalizedFrequency = static_cast<float>(entry.second.accessCount) / maxAccessCount;
            
            // Normalize recency (0.0 to 1.0, higher is more recent)
            float normalizedRecency = 1.0f - static_cast<float>(
                std::chrono::duration<double>(now - entry.second.lastAccess).count() / timeRangeSeconds
            );
            
            // Combined score (higher is better to keep)
            float score = (normalizedFrequency * frequencyWeight) + 
                         (normalizedRecency * (1.0f - frequencyWeight));
            
            // Select block with lowest score as victim
            if (score < lowestScore) {
                lowestScore = score;
                victim = entry.first;
            }
        }
        
        return victim;
    }
};

// Class to handle pinned (page-locked) memory for faster transfers
class PinnedMemoryManager {
private:
    std::unordered_map<std::string, void*> pinnedBlocks;
    std::mutex pinnedMutex;
    
public:
    PinnedMemoryManager() {}
    
    ~PinnedMemoryManager() {
        // Free all pinned memory
        for (auto& block : pinnedBlocks) {
            CUDA_CHECK(cudaFreeHost(block.second));
        }
        pinnedBlocks.clear();
    }
    
    // Allocate pinned memory
    bool allocatePinned(const std::string& identifier, size_t size) {
        std::lock_guard<std::mutex> lock(pinnedMutex);
        
        // Check if already exists
        if (pinnedBlocks.find(identifier) != pinnedBlocks.end()) {
            std::cerr << "Pinned memory block '" << identifier << "' already exists" << std::endl;
            return false;
        }
        
        // Allocate pinned memory
        void* ptr = nullptr;
        cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        pinnedBlocks[identifier] = ptr;
        return true;
    }
    
    // Free pinned memory
    bool freePinned(const std::string& identifier) {
        std::lock_guard<std::mutex> lock(pinnedMutex);
        
        auto it = pinnedBlocks.find(identifier);
        if (it == pinnedBlocks.end()) {
            std::cerr << "Pinned memory block '" << identifier << "' not found" << std::endl;
            return false;
        }
        
        CUDA_CHECK(cudaFreeHost(it->second));
        pinnedBlocks.erase(it);
        
        return true;
    }
    
    // Get pointer to pinned memory
    void* getPinnedPtr(const std::string& identifier) {
        std::lock_guard<std::mutex> lock(pinnedMutex);
        
        auto it = pinnedBlocks.find(identifier);
        if (it == pinnedBlocks.end()) {
            return nullptr;
        }
        
        return it->second;
    }
};

// Main function to demonstrate memory management
int main() {
    // Print CUDA device information
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, i));
        
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << (deviceProp.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Memory Clock Rate: " << (deviceProp.memoryClockRate / 1000) << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
    }
    
    // Run the hybrid computation example
    try {
        std::cout << "\n=== Running Hybrid Computation Example ===" << std::endl;
        runHybridComputation();
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    }
    
    // Advanced usage example with memory pool and pinned memory
    try {
        std::cout << "\n=== Advanced Memory Management Example ===" << std::endl;
        
        // Create memory manager with advanced cache policy
        HybridMemoryManager memoryManager(1ULL * 1024 * 1024 * 1024, 4ULL * 1024 * 1024 * 1024);
        memoryManager.setEvictionPolicy(std::make_unique<AdvancedCachePolicy>(0.7f)); // 70% weight on frequency
        
        // Create a memory pool
        const size_t blockSize = 4 * 1024 * 1024; // 4 MB blocks
        const size_t numBlocks = 10;
        MemoryPool memPool(memoryManager, numBlocks, blockSize);
        
        // Create pinned memory manager
        PinnedMemoryManager pinnedManager;
        pinnedManager.allocatePinned("transfer_buffer", 32 * 1024 * 1024); // 32 MB transfer buffer
        
        // Acquire blocks from pool for processing
        std::vector<std::string> activeBlocks;
        for (int i = 0; i < 5; ++i) {
            std::string blockId = memPool.acquireBlock();
            if (!blockId.empty()) {
                activeBlocks.push_back(blockId);
                
                // Use the memory block (example: initialize it)
                void* cpuPtr = memoryManager.getCpuPtr(blockId, true);
                if (cpuPtr) {
                    std::memset(cpuPtr, i, blockSize);
                    std::cout << "Initialized block " << blockId << " with value " << i << std::endl;
                }
            }
        }
        
        // Print memory stats
        memoryManager.printMemoryStats();
        
        // Release blocks back to pool
        for (const auto& blockId : activeBlocks) {
            memPool.releaseBlock(blockId);
            std::cout << "Released block " << blockId << " back to pool" << std::endl;
        }
        
        // Get pool statistics
        size_t available, inUse;
        memPool.getPoolStats(available, inUse);
        std::cout << "Pool stats: " << available << " blocks available, " 
                  << inUse << " blocks in use" << std::endl;
                  
        // Clean up pinned memory
        pinnedManager.freePinned("transfer_buffer");
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in advanced example: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nAll memory management examples completed successfully." << std::endl;
    return 0;
};

// Cache eviction policy interface
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    virtual std::string selectVictim(const std::unordered_map<std::string, MemoryAccessInfo>& accessInfo) = 0;
};

// LRU (Least Recently Used) eviction policy
class LRUEvictionPolicy : public EvictionPolicy {
public:
    std::string selectVictim(const std::unordered_map<std::string, MemoryAccessInfo>& accessInfo) override {
        std::string victim;
        std::chrono::steady_clock::time_point oldest = std::chrono::steady_clock::now();
        
        for (const auto& entry : accessInfo) {
            if (entry.second.lastAccess < oldest) {
                oldest = entry.second.lastAccess;
                victim = entry.first;
            }
        }
        
        return victim;
    }
};

// LFU (Least Frequently Used) eviction policy
class LFUEvictionPolicy : public EvictionPolicy {
public:
    std::string selectVictim(const std::unordered_map<std::string, MemoryAccessInfo>& accessInfo) override {
        std::string victim;
        size_t minCount = std::numeric_limits<size_t>::max();
        
        for (const auto& entry : accessInfo) {
            if (entry.second.accessCount < minCount) {
                minCount = entry.second.accessCount;
                victim = entry.first;
            }
        }
        
        return victim;
    }
};

// Hybrid CPU-GPU Memory Manager
class HybridMemoryManager {
private:
    // Memory allocation tracking
    std::unordered_map<std::string, std::shared_ptr<MemoryBlock>> memoryBlocks;
    std::unordered_map<std::string, MemoryAccessInfo> accessInfo;
    
    // Cache configuration
    size_t maxGpuMemory;
    size_t maxCpuMemory;
    size_t currentGpuUsage;
    size_t currentCpuUsage;
    
    // Synchronization
    std::mutex memoryMutex;
    std::mutex transferMutex;
    
    // Background transfer queue
    std::queue<std::function<void()>> transferQueue;
    std::condition_variable transferCondition;
    std::atomic<bool> isRunning;
    std::thread transferThread;
    
    // Eviction policy
    std::unique_ptr<EvictionPolicy> evictionPolicy;
    
    // CUDA stream for asynchronous operations
    cudaStream_t stream;
    
    // Background transfer thread function
    void transferThreadFunction() {
        while (isRunning) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(transferMutex);
                transferCondition.wait(lock, [this]{ 
                    return !transferQueue.empty() || !isRunning; 
                });
                
                if (!isRunning && transferQueue.empty()) {
                    break;
                }
                
                if (!transferQueue.empty()) {
                    task = transferQueue.front();
                    transferQueue.pop();
                }
            }
            
            if (task) {
                task();
            }
        }
    }
    
    // Perform memory eviction when needed
    void evictIfNeeded(MemoryLocation targetLocation, size_t requiredSize) {
        if (targetLocation == MemoryLocation::CPU || targetLocation == MemoryLocation::BOTH) {
            while (currentCpuUsage + requiredSize > maxCpuMemory && !memoryBlocks.empty()) {
                std::string victimId = evictionPolicy->selectVictim(accessInfo);
                evictBlock(victimId, MemoryLocation::CPU);
            }
        }
        
        if (targetLocation == MemoryLocation::GPU || targetLocation == MemoryLocation::BOTH) {
            while (currentGpuUsage + requiredSize > maxGpuMemory && !memoryBlocks.empty()) {
                std::string victimId = evictionPolicy->selectVictim(accessInfo);
                evictBlock(victimId, MemoryLocation::GPU);
            }
        }
    }
    
    // Evict a specific memory block from a target location
    void evictBlock(const std::string& blockId, MemoryLocation targetLocation) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        auto it = memoryBlocks.find(blockId);
        if (it == memoryBlocks.end()) {
            return;
        }
        
        std::shared_ptr<MemoryBlock> block = it->second;
        
        // If dirty, sync before eviction
        auto accessIt = accessInfo.find(blockId);
        if (accessIt != accessInfo.end() && accessIt->second.dirtyFlag) {
            if (targetLocation == MemoryLocation::GPU && block->location == MemoryLocation::BOTH) {
                // Sync CPU to GPU before evicting from GPU
                CUDA_CHECK(cudaMemcpyAsync(block->gpuPtr, block->cpuPtr, block->size, 
                                         cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else if (targetLocation == MemoryLocation::CPU && block->location == MemoryLocation::BOTH) {
                // Sync GPU to CPU before evicting from CPU
                CUDA_CHECK(cudaMemcpyAsync(block->cpuPtr, block->gpuPtr, block->size, 
                                         cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            
            accessIt->second.dirtyFlag = false;
        }
        
        // Handle eviction based on current location and target location
        if (targetLocation == MemoryLocation::CPU && 
            (block->location == MemoryLocation::CPU || block->location == MemoryLocation::BOTH)) {
            
            // Free CPU memory
            if (block->cpuPtr) {
                free(block->cpuPtr);
                block->cpuPtr = nullptr;
                currentCpuUsage -= block->size;
            }
            
            if (block->location == MemoryLocation::BOTH) {
                block->location = MemoryLocation::GPU;
            } else {
                // If only in CPU, remove it entirely
                memoryBlocks.erase(it);
                accessInfo.erase(blockId);
            }
        } 
        else if (targetLocation == MemoryLocation::GPU && 
                (block->location == MemoryLocation::GPU || block->location == MemoryLocation::BOTH)) {
            
            // Free GPU memory
            if (block->gpuPtr) {
                CUDA_CHECK(cudaFree(block->gpuPtr));
                block->gpuPtr = nullptr;
                currentGpuUsage -= block->size;
            }
            
            if (block->location == MemoryLocation::BOTH) {
                block->location = MemoryLocation::CPU;
            } else {
                // If only in GPU, remove it entirely
                memoryBlocks.erase(it);
                accessInfo.erase(blockId);
            }
        }
    }
    
    // Update access info for a memory block
    void updateAccessInfo(const std::string& blockId) {
        auto it = accessInfo.find(blockId);
        if (it != accessInfo.end()) {
            it->second.lastAccess = std::chrono::steady_clock::now();
            it->second.accessCount++;
        }
    }
    
public:
    HybridMemoryManager(size_t maxGpuMem = 4ULL * 1024 * 1024 * 1024, // 4 GB default
                        size_t maxCpuMem = 16ULL * 1024 * 1024 * 1024) // 16 GB default
        : maxGpuMemory(maxGpuMem),
          maxCpuMemory(maxCpuMem),
          currentGpuUsage(0),
          currentCpuUsage(0),
          isRunning(true),
          evictionPolicy(std::make_unique<LRUEvictionPolicy>()) {
        
        // Initialize CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Start background transfer thread
        transferThread = std::thread(&HybridMemoryManager::transferThreadFunction, this);
    }
    
    ~HybridMemoryManager() {
        // Stop background thread
        isRunning = false;
        transferCondition.notify_all();
        if (transferThread.joinable()) {
            transferThread.join();
        }
        
        // Clear all memory allocations
        memoryBlocks.clear();
        
        // Destroy CUDA stream
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    
    // Change eviction policy
    void setEvictionPolicy(std::unique_ptr<EvictionPolicy> policy) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        evictionPolicy = std::move(policy);
    }
    
    // Allocate memory with a given identifier
    bool allocate(const std::string& identifier, size_t size, MemoryLocation location = MemoryLocation::CPU) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        // Check if already exists
        if (memoryBlocks.find(identifier) != memoryBlocks.end()) {
            std::cerr << "Memory block with identifier '" << identifier << "' already exists" << std::endl;
            return false;
        }
        
        // Ensure we have enough space, evict if necessary
        evictIfNeeded(location, size);
        
        // Create new memory block
        auto block = std::make_shared<MemoryBlock>();
        block->size = size;
        block->identifier = identifier;
        block->location = location;
        
        bool success = true;
        
        // Allocate memory based on location
        if (location == MemoryLocation::CPU || location == MemoryLocation::BOTH) {
            block->cpuPtr = malloc(size);
            if (!block->cpuPtr) {
                std::cerr << "Failed to allocate " << size << " bytes on CPU" << std::endl;
                success = false;
            } else {
                currentCpuUsage += size;
            }
        }
        
        if ((location == MemoryLocation::GPU || location == MemoryLocation::BOTH) && success) {
            cudaError_t err = cudaMalloc(&block->gpuPtr, size);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate " << size << " bytes on GPU: " 
                          << cudaGetErrorString(err) << std::endl;
                
                // Free CPU memory if it was allocated
                if (block->cpuPtr) {
                    free(block->cpuPtr);
                    block->cpuPtr = nullptr;
                    currentCpuUsage -= size;
                }
                
                success = false;
            } else {
                currentGpuUsage += size;
            }
        }
        
        if (success) {
            memoryBlocks[identifier] = block;
            
            // Initialize access info
            MemoryAccessInfo accessInfoEntry;
            accessInfoEntry.lastAccess = std::chrono::steady_clock::now();
            accessInfoEntry.accessCount = 0;
            accessInfoEntry.dirtyFlag = false;
            accessInfo[identifier] = accessInfoEntry;
            
            return true;
        }
        
        return false;
    }
    
    // Free allocated memory
    bool free(const std::string& identifier) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        auto it = memoryBlocks.find(identifier);
        if (it == memoryBlocks.end()) {
            std::cerr << "Memory block with identifier '" << identifier << "' not found" << std::endl;
            return false;
        }
        
        std::shared_ptr<MemoryBlock> block = it->second;
        
        // Update memory usage tracking
        if (block->cpuPtr) {
            currentCpuUsage -= block->size;
        }
        
        if (block->gpuPtr) {
            currentGpuUsage -= block->size;
        }
        
        // Remove the block (destructor will handle freeing memory)
        memoryBlocks.erase(it);
        accessInfo.erase(identifier);
        
        return true;
    }
    
    // Get CPU pointer for a memory block
    void* getCpuPtr(const std::string& identifier, bool markDirty = false) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        auto it = memoryBlocks.find(identifier);
        if (it == memoryBlocks.end()) {
            std::cerr << "Memory block with identifier '" << identifier << "' not found" << std::endl;
            return nullptr;
        }
        
        std::shared_ptr<MemoryBlock> block = it->second;
        updateAccessInfo(identifier);
        
        // If data is only on GPU, we need to transfer to CPU
        if (block->location == MemoryLocation::GPU) {
            // Allocate CPU memory if needed
            if (!block->cpuPtr) {
                block->cpuPtr = malloc(block->size);
                if (!block->cpuPtr) {
                    std::cerr << "Failed to allocate " << block->size << " bytes on CPU" << std::endl;
                    return nullptr;
                }
                currentCpuUsage += block->size;
            }
            
            // Transfer data from GPU to CPU
            CUDA_CHECK(cudaMemcpyAsync(block->cpuPtr, block->gpuPtr, block->size, 
                                     cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            block->location = MemoryLocation::BOTH;
        }
        
        // If marking as dirty, update the flag
        if (markDirty) {
            accessInfo[identifier].dirtyFlag = true;
        }
        
        return block->cpuPtr;
    }
    
    // Get GPU pointer for a memory block
    void* getGpuPtr(const std::string& identifier) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        auto it = memoryBlocks.find(identifier);
        if (it == memoryBlocks.end()) {
            std::cerr << "Memory block with identifier '" << identifier << "' not found" << std::endl;
            return nullptr;
        }
        
        std::shared_ptr<MemoryBlock> block = it->second;
        updateAccessInfo(identifier);
        
        // If data is only on CPU, we need to transfer to GPU
        if (block->location == MemoryLocation::CPU) {
            // Allocate GPU memory if needed
            if (!block->gpuPtr) {
                cudaError_t err = cudaMalloc(&block->gpuPtr, block->size);
                if (err != cudaSuccess) {
                    std::cerr << "Failed to allocate " << block->size << " bytes on GPU: " 
                              << cudaGetErrorString(err) << std::endl;
                    return nullptr;
                }
                currentGpuUsage += block->size;
            }
            
            // Transfer data from CPU to GPU
            CUDA_CHECK(cudaMemcpyAsync(block->gpuPtr, block->cpuPtr, block->size, 
                                     cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            block->location = MemoryLocation::BOTH;
        }
        
        // If CPU data was marked as dirty, we need to sync
        auto accessIt = accessInfo.find(identifier);
        if (accessIt != accessInfo.end() && accessIt->second.dirtyFlag) {
            CUDA_CHECK(cudaMemcpyAsync(block->gpuPtr, block->cpuPtr, block->size, 
                                     cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            accessIt->second.dirtyFlag = false;
        }
        
        return block->gpuPtr;
    }
    
    // Prefetch data to a specific location (useful for optimization)
    void prefetch(const std::string& identifier, MemoryLocation targetLocation) {
        auto queueTask = [this, identifier, targetLocation]() {
            std::lock_guard<std::mutex> lock(this->memoryMutex);
            
            auto it = this->memoryBlocks.find(identifier);
            if (it == this->memoryBlocks.end()) {
                return;
            }
            
            std::shared_ptr<MemoryBlock> block = it->second;
            
            if (targetLocation == MemoryLocation::GPU && block->location == MemoryLocation::CPU) {
                // Allocate GPU memory if needed
                if (!block->gpuPtr) {
                    cudaError_t err = cudaMalloc(&block->gpuPtr, block->size);
                    if (err != cudaSuccess) {
                        return;
                    }
                    this->currentGpuUsage += block->size;
                }
                
                // Transfer data to GPU
                CUDA_CHECK(cudaMemcpyAsync(block->gpuPtr, block->cpuPtr, block->size, 
                                         cudaMemcpyHostToDevice, this->stream));
                CUDA_CHECK(cudaStreamSynchronize(this->stream));
                
                block->location = MemoryLocation::BOTH;
            }
            else if (targetLocation == MemoryLocation::CPU && block->location == MemoryLocation::GPU) {
                // Allocate CPU memory if needed
                if (!block->cpuPtr) {
                    block->cpuPtr = malloc(block->size);
                    if (!block->cpuPtr) {
                        return;
                    }
                    this->currentCpuUsage += block->size;
                }
                
                // Transfer data to CPU
                CUDA_CHECK(cudaMemcpyAsync(block->cpuPtr, block->gpuPtr, block->size, 
                                         cudaMemcpyDeviceToHost, this->stream));
                CUDA_CHECK(cudaStreamSynchronize(this->stream));
                
                block->location = MemoryLocation::BOTH;
            }
        };
        
        // Queue the prefetch task
        {
            std::lock_guard<std::mutex> lock(transferMutex);
            transferQueue.push(queueTask);
        }
        
        transferCondition.notify_one();
    }
    
    // Synchronize memory between CPU and GPU
    void sync(const std::string& identifier, MemoryLocation direction = MemoryLocation::BOTH) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        auto it = memoryBlocks.find(identifier);
        if (it == memoryBlocks.end()) {
            std::cerr << "Memory block with identifier '" << identifier << "' not found" << std::endl;
            return;
        }
        
        std::shared_ptr<MemoryBlock> block = it->second;
        
        // If location is CPU or BOTH, and the memory block has both CPU and GPU pointers
        if ((direction == MemoryLocation::CPU || direction == MemoryLocation::BOTH) && 
            block->cpuPtr && block->gpuPtr) {
            
            // Transfer from GPU to CPU
            CUDA_CHECK(cudaMemcpyAsync(block->cpuPtr, block->gpuPtr, block->size, 
                                     cudaMemcpyDeviceToHost, stream));
        }
        
        // If location is GPU or BOTH, and the memory block has both CPU and GPU pointers
        if ((direction == MemoryLocation::GPU || direction == MemoryLocation::BOTH) && 
            block->cpuPtr && block->gpuPtr) {
            
            // Transfer from CPU to GPU
            CUDA_CHECK(cudaMemcpyAsync(block->gpuPtr, block->cpuPtr, block->size, 
                                     cudaMemcpyHostToDevice, stream));
            
            // Reset dirty flag
            accessInfo[identifier].dirtyFlag = false;
        }
        
        // Synchronize the stream to ensure transfers are complete
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    // Get memory usage statistics
    void getMemoryStats(size_t& cpuUsed, size_t& gpuUsed, size_t& cpuTotal, size_t& gpuTotal) {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        cpuUsed = currentCpuUsage;
        gpuUsed = currentGpuUsage;
        cpuTotal = maxCpuMemory;
        gpuTotal = maxGpuMemory;
    }
    
    // Print memory usage and diagnostics information
    void printMemoryStats() {
        std::lock_guard<std::mutex> lock(memoryMutex);
        
        std::cout << "=== Memory Manager Statistics ===" << std::endl;
        std::cout << "CPU Memory: " << currentCpuUsage / (1024.0 * 1024.0) << " MB / " 
                  << maxCpuMemory / (1024.0 * 1024.0) << " MB ("
                  << (currentCpuUsage * 100.0 / maxCpuMemory) << "%)" << std::endl;
        
        std::cout << "GPU Memory: " << currentGpuUsage / (1024.0 * 1024.0) << " MB / " 
                  << maxGpuMemory / (1024.0 * 1024.0) << " MB ("
                  << (currentGpuUsage * 100.0 / maxGpuMemory) << "%)" << std::endl;
        
        std::cout << "Total Memory Blocks: " << memoryBlocks.size() << std::endl;
        
        std::cout << "\nDetailed Memory Block Information:" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        std::cout << std::left << std::setw(20) << "Identifier" 
                  << std::setw(12) << "Size (MB)" 
                  << std::setw(10) << "Location" 
                  << std::setw(10) << "Access #" 
                  << std::setw(10) << "Dirty" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        
        for (const auto& entry : memoryBlocks) {
            const auto& block = entry.second;
            const auto& acc = accessInfo[entry.first];
            
            std::string locationStr;
            switch (block->location) {
                case MemoryLocation::CPU: locationStr = "CPU"; break;
                case MemoryLocation::GPU: locationStr = "GPU"; break;
                case MemoryLocation::BOTH: locationStr = "BOTH"; break;
            }
            
            std::cout << std::left << std::setw(20) << block->identifier 
                      << std::setw(12) << (block->size / (1024.0 * 1024.0))
                      << std::setw(10) << locationStr
                      << std::setw(10) << acc.accessCount
                      << std::setw(10) << (acc.dirtyFlag ? "Yes" : "No") << std::endl;
        }
        
        std::cout << "=============================" << std::endl;
    }
};

// Templated wrapper for typed data access
template<typename T>
class TypedMemoryBlock {
private:
    HybridMemoryManager& memoryManager;
    std::string identifier;
    size_t elementCount;

public:
    TypedMemoryBlock(HybridMemoryManager& manager, const std::string& id, size_t count)
        : memoryManager(manager), identifier(id), elementCount(count) {
        
        memoryManager.allocate(identifier, count * sizeof(T), MemoryLocation::CPU);
    }
    
    ~TypedMemoryBlock() {
        memoryManager.free(identifier);
    }
    
    // Get typed CPU pointer
    T* getCpuPtr(bool markDirty = false) {
        return static_cast<T*>(memoryManager.getCpuPtr(identifier, markDirty));
    }
    
    // Get typed GPU pointer
    T* getGpuPtr() {
        return static_cast<T*>(memoryManager.getGpuPtr(identifier));
    }
    
    // Initialize with value
    void initializeCpu(const T& value) {
        T* ptr = getCpuPtr(true);
        if (ptr) {
            std::fill(ptr, ptr + elementCount, value);
        }
    }
    
    // Sync memory
    void sync(MemoryLocation direction = MemoryLocation::BOTH) {
        memoryManager.sync(identifier, direction);
    }
    
    // Prefetch data
    void prefetch(MemoryLocation location) {
        memoryManager.prefetch(identifier, location);
    }
    
    // Get element count
    size_t size() const {
        return elementCount;
    }
};

// Example function to perform computation on CPU
template<typename T>
void computeOnCPU(TypedMemoryBlock<T>& data) {
    T* ptr = data.getCpuPtr(true); // Mark as dirty since we'll modify it
    
    if (ptr) {
        for (size_t i = 0; i < data.size(); ++i) {
            // Example computation
            ptr[i] = static_cast<T>(i * 2);
        }
    }
}

// Example CUDA kernel for GPU computation
template<typename T>
__global__ void computeOnGPUKernel(T* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Example computation
        data[idx] = static_cast<T>(idx * 3);
    }
}

// Example function to perform computation on GPU
template<typename T>
void computeOnGPU(TypedMemoryBlock<T>& data) {
    T* d_ptr = data.getGpuPtr(); // This will transfer data to GPU if needed
    
    if (d_ptr) {
        const int blockSize = 256;
        const int numBlocks = (data.size() + blockSize - 1) / blockSize;
        
        computeOnGPUKernel<<<numBlocks, blockSize>>>(d_ptr, data.size());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Example usage of the memory manager in a complex computation scenario
void runHybridComputation() {
    // Create memory manager
    HybridMemoryManager memoryManager(2ULL * 1024 * 1024 * 1024, // 2 GB GPU memory
                                      8ULL * 1024 * 1024 * 1024); // 8 GB CPU memory
    
    // Change to LFU policy if desired
    // memoryManager.setEvictionPolicy(std::make_unique<LFUEvictionPolicy>());
    
    // Create some data blocks
    const size_t size1 = 100 * 1024 * 1024; // 100M elements
    const size_t size2 = 50 * 1024 * 1024;  // 50M elements
    
    TypedMemoryBlock<float> dataBlock1(memoryManager, "computation1", size1);
    TypedMemoryBlock<float> dataBlock2(memoryManager, "computation2", size2);
    
    // Initialize data on CPU
    dataBlock1.initializeCpu(1.0f);
    dataBlock2.initializeCpu(2.0f);
    
    // Print initial memory stats
    memoryManager.printMemoryStats();
    
    // Start prefetching data block 2 to GPU while we work on data block 1
    dataBlock2.prefetch(MemoryLocation::GPU);
    
    // Compute on CPU with data block 1
    computeOnCPU(dataBlock1);
    
    // Sync data block 1 to GPU
    dataBlock1.sync(MemoryLocation::GPU);
    
    // Compute on GPU with data block 2
    computeOnGPU(dataBlock2);
    
    // Print memory stats after computation
    memoryManager.printMemoryStats();
    
    // Sync results back to CPU for verification
    dataBlock2.sync(MemoryLocation::CPU);
    
    // Access results on CPU
    float* result1 = dataBlock1.getCpuPtr();
    float* result2 = dataBlock2.getCpuPtr();
    
    // Verify some results
    if (result1 && result2) {
        std::cout << "Sample results from computation1: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << result1[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Sample results from computation2: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << result2[i] << " ";
        }
        std::cout << std::endl;
