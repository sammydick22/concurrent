import ctypes
import numpy as np
import os
import sys
import time
from enum import Enum
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define enums and structures to match C++
class MemoryLocation(Enum):
    CPU = 0
    GPU = 1
    BOTH = 2

# Load the shared library
def load_library():
    # Determine the appropriate file extension based on the platform
    if sys.platform.startswith('win'):
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hybrid_memory_lib.dll')
    elif sys.platform.startswith('darwin'):
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libhybrid_memory_lib.dylib')
    else:  # Linux and others
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libhybrid_memory_lib.so')
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Shared library not found at {lib_path}")
    
    # Load the library
    try:
        return ctypes.CDLL(lib_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load shared library: {e}")

# Python wrapper for the C++ HybridMemoryManager
class HybridMemoryManager:
    def __init__(self, max_gpu_memory=4*1024*1024*1024, max_cpu_memory=16*1024*1024*1024):
        """
        Initialize the Hybrid Memory Manager
        
        Args:
            max_gpu_memory: Maximum GPU memory in bytes (default: 4GB)
            max_cpu_memory: Maximum CPU memory in bytes (default: 16GB)
        """
        try:
            self.lib = load_library()
            
            # Define function signatures
            self.lib.CreateMemoryManager.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
            self.lib.CreateMemoryManager.restype = ctypes.c_void_p
            
            self.lib.DestroyMemoryManager.argtypes = [ctypes.c_void_p]
            self.lib.DestroyMemoryManager.restype = None
            
            self.lib.AllocateMemory.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
            self.lib.AllocateMemory.restype = ctypes.c_bool
            
            self.lib.FreeMemory.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.lib.FreeMemory.restype = ctypes.c_bool
            
            self.lib.GetCpuPtr.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
            self.lib.GetCpuPtr.restype = ctypes.c_void_p
            
            self.lib.GetGpuPtr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.lib.GetGpuPtr.restype = ctypes.c_void_p
            
            self.lib.SyncMemory.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            self.lib.SyncMemory.restype = None
            
            self.lib.PrefetchMemory.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            self.lib.PrefetchMemory.restype = None
            
            self.lib.GetMemoryStats.argtypes = [ctypes.c_void_p, 
                                              ctypes.POINTER(ctypes.c_size_t),
                                              ctypes.POINTER(ctypes.c_size_t),
                                              ctypes.POINTER(ctypes.c_size_t),
                                              ctypes.POINTER(ctypes.c_size_t)]
            self.lib.GetMemoryStats.restype = None
            
            self.lib.SetEvictionPolicy.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.SetEvictionPolicy.restype = None
            
            # Create the memory manager instance
            self.manager = self.lib.CreateMemoryManager(max_gpu_memory, max_cpu_memory)
            if not self.manager:
                raise RuntimeError("Failed to create memory manager")
            
            # Keep track of allocated blocks for safety
            self.allocated_blocks = {}
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HybridMemoryManager: {e}")
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        if hasattr(self, 'manager') and self.manager:
            self.lib.DestroyMemoryManager(self.manager)
            self.manager = None
    
    def allocate(self, identifier, size, location=MemoryLocation.CPU):
        """
        Allocate memory with a given identifier
        
        Args:
            identifier: String identifier for the memory block
            size: Size in bytes
            location: MemoryLocation (CPU, GPU, or BOTH)
            
        Returns:
            True if allocation succeeded, False otherwise
        """
        id_bytes = identifier.encode('utf-8')
        result = self.lib.AllocateMemory(self.manager, id_bytes, size, location.value)
        if result:
            self.allocated_blocks[identifier] = {
                'size': size,
                'location': location
            }
        return result
    
    def free(self, identifier):
        """
        Free allocated memory
        
        Args:
            identifier: String identifier for the memory block
            
        Returns:
            True if freeing succeeded, False otherwise
        """
        id_bytes = identifier.encode('utf-8')
        result = self.lib.FreeMemory(self.manager, id_bytes)
        if result and identifier in self.allocated_blocks:
            del self.allocated_blocks[identifier]
        return result
    
    def get_cpu_ptr(self, identifier, mark_dirty=False):
        """
        Get CPU pointer for a memory block
        
        Args:
            identifier: String identifier for the memory block
            mark_dirty: Whether to mark the memory as modified (will sync to GPU later)
            
        Returns:
            Memory address as integer
        """
        id_bytes = identifier.encode('utf-8')
        return self.lib.GetCpuPtr(self.manager, id_bytes, mark_dirty)
    
    def get_gpu_ptr(self, identifier):
        """
        Get GPU pointer for a memory block
        
        Args:
            identifier: String identifier for the memory block
            
        Returns:
            Memory address as integer
        """
        id_bytes = identifier.encode('utf-8')
        return self.lib.GetGpuPtr(self.manager, id_bytes)
    
    def sync(self, identifier, direction=MemoryLocation.BOTH):
        """
        Synchronize memory between CPU and GPU
        
        Args:
            identifier: String identifier for the memory block
            direction: Direction of synchronization (CPU, GPU, or BOTH)
        """
        id_bytes = identifier.encode('utf-8')
        self.lib.SyncMemory(self.manager, id_bytes, direction.value)
    
    def prefetch(self, identifier, target_location):
        """
        Prefetch data to a specific location
        
        Args:
            identifier: String identifier for the memory block
            target_location: Target location (CPU or GPU)
        """
        id_bytes = identifier.encode('utf-8')
        self.lib.PrefetchMemory(self.manager, id_bytes, target_location.value)
    
    def get_memory_stats(self):
        """
        Get memory usage statistics
        
        Returns:
            Tuple of (cpu_used, gpu_used, cpu_total, gpu_total) in bytes
        """
        cpu_used = ctypes.c_size_t()
        gpu_used = ctypes.c_size_t()
        cpu_total = ctypes.c_size_t()
        gpu_total = ctypes.c_size_t()
        
        self.lib.GetMemoryStats(self.manager, 
                              ctypes.byref(cpu_used),
                              ctypes.byref(gpu_used),
                              ctypes.byref(cpu_total),
                              ctypes.byref(gpu_total))
        
        return (cpu_used.value, gpu_used.value, cpu_total.value, gpu_total.value)
    
    def set_eviction_policy(self, policy_type):
        """
        Set eviction policy
        
        Args:
            policy_type: 0 for LRU, 1 for LFU, 2 for Advanced
        """
        self.lib.SetEvictionPolicy(self.manager, policy_type)
    
    def create_numpy_array(self, identifier, shape, dtype=np.float32, location=MemoryLocation.CPU):
        """
        Create a numpy array backed by memory managed by HybridMemoryManager
        
        Args:
            identifier: String identifier for the memory block
            shape: Shape of the numpy array
            dtype: Data type of the numpy array
            location: Initial memory location
            
        Returns:
            NumPy array
        """
        # Calculate size
        size = np.prod(shape) * np.dtype(dtype).itemsize
        
        # Allocate memory
        if not self.allocate(identifier, size, location):
            raise RuntimeError(f"Failed to allocate memory for array '{identifier}'")
        
        # Get CPU pointer
        ptr = self.get_cpu_ptr(identifier)
        if not ptr:
            raise RuntimeError(f"Failed to get CPU pointer for array '{identifier}'")
        
        # Create NumPy array using the allocated memory
        # Note: We're not specifying strides, assuming contiguous memory
        return np.ctypeslib.as_array((ctypes.c_byte * size).from_address(ptr)).view(dtype).reshape(shape)
    
    def array_to_gpu(self, identifier):
        """
        Transfer a numpy array to GPU
        
        Args:
            identifier: String identifier for the memory block
        """
        self.sync(identifier, MemoryLocation.GPU)
    
    def array_from_gpu(self, identifier):
        """
        Transfer data from GPU to the numpy array
        
        Args:
            identifier: String identifier for the memory block
        """
        self.sync(identifier, MemoryLocation.CPU)


# NumPy array wrapper class for easier usage
class HybridArray:
    def __init__(self, memory_manager, identifier, shape, dtype=np.float32, location=MemoryLocation.CPU):
        """
        Create a numpy array with hybrid memory management
        
        Args:
            memory_manager: HybridMemoryManager instance
            identifier: String identifier for the memory block
            shape: Shape of the numpy array
            dtype: Data type of the numpy array
            location: Initial memory location
        """
        self.memory_manager = memory_manager
        self.identifier = identifier
        self.shape = shape
        self.dtype = dtype
        self.location = location
        
        # Create the array
        self.array = memory_manager.create_numpy_array(identifier, shape, dtype, location)
    
    def to_gpu(self):
        """Transfer data to GPU"""
        self.memory_manager.array_to_gpu(self.identifier)
        self.location = MemoryLocation.BOTH
    
    def from_gpu(self):
        """Transfer data from GPU to CPU"""
        self.memory_manager.array_from_gpu(self.identifier)
    
    def get_gpu_ptr(self):
        """Get GPU pointer"""
        return self.memory_manager.get_gpu_ptr(self.identifier)
    
    def __getitem__(self, key):
        """Enable array-like indexing"""
        return self.array[key]
    
    def __setitem__(self, key, value):
        """Enable array-like value setting"""
        self.array[key] = value
        # Mark as dirty since we modified the CPU data
        self.memory_manager.get_cpu_ptr(self.identifier, True)
    
    def __del__(self):
        """Clean up when the object is destroyed"""
        try:
            self.memory_manager.free(self.identifier)
        except:
            pass


# Example CUDA kernel execution from Python
def execute_gpu_kernel(hybrid_array, kernel_name, block_size=256):
    """
    Execute a CUDA kernel on the data
    
    Args:
        hybrid_array: HybridArray instance
        kernel_name: Name of the kernel function in the shared library
        block_size: CUDA block size
    """
    # Get library handle from the memory manager
    lib = hybrid_array.memory_manager.lib
    
    # Get kernel function
    kernel_func = getattr(lib, kernel_name, None)
    if not kernel_func:
        raise ValueError(f"Kernel function '{kernel_name}' not found in the library")
    
    # Set function signature
    kernel_func.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel_func.restype = None
    
    # Ensure data is on GPU
    hybrid_array.to_gpu()
    
    # Get GPU pointer
    gpu_ptr = hybrid_array.get_gpu_ptr()
    
    # Call the kernel
    array_size = np.prod(hybrid_array.shape)
    kernel_func(gpu_ptr, array_size)
    
    # Mark data as needing sync back to CPU
    hybrid_array.location = MemoryLocation.GPU


# GUI Application for Memory Management
class HybridMemoryManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hybrid CPU-GPU Memory Manager")
        self.geometry("1200x800")
        
        # Initialize memory manager with 1GB GPU and 4GB CPU memory
        self.memory_manager = HybridMemoryManager(1024*1024*1024, 4*1024*1024*1024)
        
        # Set up the UI
        self.create_widgets()
        
        # Test arrays for benchmarking
        self.test_arrays = {}
        
        # For monitoring memory usage
        self.monitoring = False
        self.memory_history = {
            'cpu_used': [],
            'gpu_used': []
        }
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        tab_control = ttk.Notebook(main_frame)
        
        # Memory management tab
        memory_tab = ttk.Frame(tab_control)
        self.setup_memory_tab(memory_tab)
        tab_control.add(memory_tab, text="Memory Management")
        
        # Benchmark tab
        benchmark_tab = ttk.Frame(tab_control)
        self.setup_benchmark_tab(benchmark_tab)
        tab_control.add(benchmark_tab, text="Benchmarks")
        
        # Monitoring tab
        monitoring_tab = ttk.Frame(tab_control)
        self.setup_monitoring_tab(monitoring_tab)
        tab_control.add(monitoring_tab, text="Monitoring")
        
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # Update memory stats
        self.update_memory_stats()
    
    def setup_memory_tab(self, parent):
        # Left panel for controls
        control_frame = ttk.LabelFrame(parent, text="Memory Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create array frame
        create_frame = ttk.Frame(control_frame)
        create_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(create_frame, text="Array Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.array_name_var = tk.StringVar(value="test_array_1")
        ttk.Entry(create_frame, textvariable=self.array_name_var).grid(row=0, column=1, pady=2)
        
        ttk.Label(create_frame, text="Shape:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.array_shape_var = tk.StringVar(value="1000, 1000")
        ttk.Entry(create_frame, textvariable=self.array_shape_var).grid(row=1, column=1, pady=2)
        
        ttk.Label(create_frame, text="Data Type:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.array_dtype_var = tk.StringVar(value="float32")
        dtype_combo = ttk.Combobox(create_frame, textvariable=self.array_dtype_var)
        dtype_combo['values'] = ('float32', 'float64', 'int32', 'int64')
        dtype_combo.grid(row=2, column=1, pady=2)
        
        ttk.Label(create_frame, text="Initial Location:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.array_location_var = tk.StringVar(value="CPU")
        location_combo = ttk.Combobox(create_frame, textvariable=self.array_location_var)
        location_combo['values'] = ('CPU', 'GPU', 'BOTH')
        location_combo.grid(row=3, column=1, pady=2)
        
        ttk.Button(create_frame, text="Create Array", command=self.create_array).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Operations frame
        ops_frame = ttk.LabelFrame(control_frame, text="Array Operations")
        ops_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ops_frame, text="Select Array:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.selected_array_var = tk.StringVar()
        self.array_combo = ttk.Combobox(ops_frame, textvariable=self.selected_array_var)
        self.array_combo.grid(row=0, column=1, pady=2)
        
        # Buttons for operations
        op_buttons_frame = ttk.Frame(ops_frame)
        op_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(op_buttons_frame, text="To GPU", command=self.transfer_to_gpu).pack(side=tk.LEFT, padx=5)
        ttk.Button(op_buttons_frame, text="From GPU", command=self.transfer_from_gpu).pack(side=tk.LEFT, padx=5)
        ttk.Button(op_buttons_frame, text="Fill Random", command=self.fill_random).pack(side=tk.LEFT, padx=5)
        ttk.Button(op_buttons_frame, text="Free Memory", command=self.free_array).pack(side=tk.LEFT, padx=5)
        
        # Policy selection
        policy_frame = ttk.Frame(control_frame)
        policy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(policy_frame, text="Eviction Policy:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.policy_var = tk.StringVar(value="LRU")
        policy_combo = ttk.Combobox(policy_frame, textvariable=self.policy_var)
        policy_combo['values'] = ('LRU', 'LFU', 'Advanced')
        policy_combo.grid(row=0, column=1, pady=2)
        
        ttk.Button(policy_frame, text="Set Policy", command=self.set_policy).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Right panel for array info
        info_frame = ttk.LabelFrame(parent, text="Array Information")
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a Treeview for array listing
        columns = ('name', 'shape', 'size', 'dtype', 'location')
        self.array_tree = ttk.Treeview(info_frame, columns=columns, show='headings')
        
        # Define headings
        self.array_tree.heading('name', text='Name')
        self.array_tree.heading('shape', text='Shape')
        self.array_tree.heading('size', text='Size (MB)')
        self.array_tree.heading('dtype', text='Data Type')
        self.array_tree.heading('location', text='Location')
        
        # Define columns
        self.array_tree.column('name', width=100)
        self.array_tree.column('shape', width=100)
        self.array_tree.column('size', width=80)
        self.array_tree.column('dtype', width=80)
        self.array_tree.column('location', width=80)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.array_tree.yview)
        self.array_tree.configure(yscroll=scrollbar.set)
        
        # Pack the Treeview and scrollbar
        self.array_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status section
        status_frame = ttk.LabelFrame(info_frame, text="Memory Status")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.cpu_usage_var = tk.StringVar()
        self.gpu_usage_var = tk.StringVar()
        
        ttk.Label(status_frame, text="CPU Usage:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(status_frame, textvariable=self.cpu_usage_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(status_frame, text="GPU Usage:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(status_frame, textvariable=self.gpu_usage_var).grid(row=1, column=1, sticky=tk.W, pady=2)
    
    def setup_benchmark_tab(self, parent):
        # Left panel for benchmark controls
        control_frame = ttk.LabelFrame(parent, text="Benchmark Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Benchmark parameters
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Array Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bench_size_var = tk.StringVar(value="1000, 1000")
        ttk.Entry(params_frame, textvariable=self.bench_size_var).grid(row=0, column=1, pady=2)
        
        ttk.Label(params_frame, text="Number of Arrays:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.bench_num_arrays_var = tk.StringVar(value="5")
        ttk.Entry(params_frame, textvariable=self.bench_num_arrays_var).grid(row=1, column=1, pady=2)
        
        ttk.Label(params_frame, text="Iterations:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.bench_iterations_var = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.bench_iterations_var).grid(row=2, column=1, pady=2)
        
        # Benchmark buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Run Transfer Benchmark", 
                  command=self.run_transfer_benchmark).pack(side=tk.LEFT, padx=5, pady=5)
                  
        ttk.Button(button_frame, text="Run Allocation Benchmark", 
                  command=self.run_allocation_benchmark).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Benchmark Results")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text area for results
        self.results_text = tk.Text(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_monitoring_tab(self, parent):
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.monitor_button = ttk.Button(control_frame, text="Start Monitoring", command=self.toggle_monitoring)
        self.monitor_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Clear History", command=self.clear_monitoring).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Update Interval (ms):").pack(side=tk.LEFT, padx=5)
        self.update_interval_var = tk.StringVar(value="1000")
        ttk.Entry(control_frame, textvariable=self.update_interval_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Graph frame
        graph_frame = ttk.LabelFrame(parent, text="Memory Usage Over Time")
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self.update_plot()
    
    def create_array(self):
        """Create a new array in the memory manager"""
        try:
            # Parse inputs
            name = self.array_name_var.get()
            shape_str = self.array_shape_var.get()
            dtype_str = self.array_dtype_var.get()
            location_str = self.array_location_var.get()
            
            # Convert shape string to tuple
            shape = tuple(int(dim.strip()) for dim in shape_str.split(','))
            
            # Convert dtype string to numpy dtype
            dtype = getattr(np, dtype_str)
            
            # Convert location string to enum
            location = getattr(MemoryLocation, location_str)
            
            # Create the array
            array = HybridArray(self.memory_manager, name, shape, dtype, location)
            
            # Store the array
            self.test_arrays[name] = array
            
            # Update the UI
            self.update_array_list()
            self.status_label.config(text=f"Created array '{name}' with shape {shape}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create array: {e}")
    
    def transfer_to_gpu(self):
        """Transfer selected array to GPU"""
        try:
            name = self.selected_array_var.get()
            if name in self.test_arrays:
                start_time = time.time()
                self.test_arrays[name].to_gpu()
                elapsed = time.time() - start_time
                
                self.status_label.config(text=f"Transferred '{name}' to GPU in {elapsed:.4f} seconds")
                self.update_array_list()
            else:
                messagebox.showwarning("Warning", f"Array '{name}' not found")
        except Exception as e:
            messagebox.showerror("Error", f"Transfer failed: {e}")
    
    def transfer_from_gpu(self):
        """Transfer selected array from GPU to CPU"""
        try:
            name = self.selected_array_var.get()
            if name in self.test_arrays:
                start_time = time.time()
                self.test_arrays[name].from_gpu()
                elapsed = time.time() - start_time
                
                self.status_label.config(text=f"Transferred '{name}' from GPU in {elapsed:.4f} seconds")
                self.update_array_list()
            else:
                messagebox.showwarning("Warning", f"Array '{name}' not found")
        except Exception as e:
            messagebox.showerror("Error", f"Transfer failed: {e}")
    
    def fill_random(self):
        """Fill selected array with random data"""
        try:
            name = self.selected_array_var.get()
            if name in self.test_arrays:
                array = self.test_arrays[name]
                # Generate random data
                array.array[:] = np.random.random(array.shape).astype(array.dtype)
                self.status_label.config(text=f"Filled '{name}' with random data")
                self.update_array_list()
            else:
                messagebox.showwarning("Warning", f"Array '{name}' not found")
        except Exception as e:
            messagebox.showerror("Error", f"Fill operation failed: {e}")
    
    def free_array(self):
        """Free the selected array"""
        try:
            name = self.selected_array_var.get()
            if name in self.test_arrays:
                # Delete the array (destructor will free memory)
                del self.test_arrays[name]
                self.status_label.config(text=f"Freed array '{name}'")
                self.update_array_list()
            else:
                messagebox.showwarning("Warning", f"Array '{name}' not found")
        except Exception as e:
            messagebox.showerror("Error", f"Free operation failed: {e}")
    
    def set_policy(self):
        """Set the eviction policy"""
        try:
            policy_str = self.policy_var.get()
            policy_map = {"LRU": 0, "LFU": 1, "Advanced": 2}
            
            if policy_str in policy_map:
                self.memory_manager.set_eviction_policy(policy_map[policy_str])
                self.status_label.config(text=f"Set eviction policy to {policy_str}")
            else:
                messagebox.showwarning("Warning", "Invalid policy selection")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set policy: {e}")
    
    def update_array_list(self):
        """Update the array listing in the UI"""
        # Clear existing items
        for item in self.array_tree.get_children():
            self.array_tree.delete(item)
        
        # Add arrays to the list
        for name, array in self.test_arrays.items():
            # Calculate size in MB
            size_mb = np.prod(array.shape) * np.dtype(array.dtype).itemsize / (1024*1024)
            
            # Add to tree
            self.array_tree.insert('', 'end', values=(
                name,
                str(array.shape),
                f"{size_mb:.2f}",
                str(array.dtype),
                array.location.name
            ))
        
        # Update combobox values
        self.array_combo['values'] = list(self.test_arrays.keys())
        
        # Update memory stats
        self.update_memory_stats()
    
    def update_memory_stats(self):
        """Update memory usage statistics in the UI"""
        try:
            cpu_used, gpu_used, cpu_total, gpu_total = self.memory_manager.get_memory_stats()
            
            # Convert to MB for display
            cpu_used_mb = cpu_used / (1024*1024)
            gpu_used_mb = gpu_used / (1024*1024)
            cpu_total_mb = cpu_total / (1024*1024)
            gpu_total_mb = gpu_total / (1024*1024)
            
            # Update UI
            self.cpu_usage_var.set(f"{cpu_used_mb:.2f} MB / {cpu_total_mb:.2f} MB ({cpu_used*100/cpu_total:.1f}%)")
            self.gpu_usage_var.set(f"{gpu_used_mb:.2f} MB / {gpu_total_mb:.2f} MB ({gpu_used*100/gpu_total:.1f}%)")
            
            # Update memory history if monitoring
            if self.monitoring:
                self.memory_history['cpu_used'].append(cpu_used_mb)
                self.memory_history['gpu_used'].append(gpu_used_mb)
                self.update_plot()
        except Exception as e:
            print(f"Error updating memory stats: {e}")
    
    def run_transfer_benchmark(self):
        """Run benchmark for transfer operations"""
        try:
            # Get benchmark parameters
            shape_str = self.bench_size_var.get()
            shape = tuple(int(dim.strip()) for dim in shape_str.split(','))
            num_arrays = int(self.bench_num_arrays_var.get())
            iterations = int(self.bench_iterations_var.get())
            
            # Clear results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Running transfer benchmark with {shape} arrays...\n\n")
            
            # Create test arrays
            arrays = []
            for i in range(num_arrays):
                name = f"bench_array_{i}"
                arrays.append(HybridArray(self.memory_manager, name, shape, np.float32, MemoryLocation.CPU))
                # Fill with random data
                arrays[i].array[:] = np.random.random(shape).astype(np.float32)
            
            # CPU to GPU transfer timing
            self.results_text.insert(tk.END, "CPU to GPU Transfer Times:\n")
            cpu_to_gpu_times = []
            
            for i in range(iterations):
                start_time = time.time()
                for array in arrays:
                    array.to_gpu()
                elapsed = time.time() - start_time
                cpu_to_gpu_times.append(elapsed)
                self.results_text.insert(tk.END, f"  Iteration {i+1}: {elapsed:.4f} seconds\n")
                self.results_text.see(tk.END)
                self.update()
            
            avg_cpu_to_gpu = sum(cpu_to_gpu_times) / len(cpu_to_gpu_times)
            self.results_text.insert(tk.END, f"\nAverage CPU to GPU: {avg_cpu_to_gpu:.4f} seconds\n\n")
            
            # GPU to CPU transfer timing
            self.results_text.insert(tk.END, "GPU to CPU Transfer Times:\n")
            gpu_to_cpu_times = []
            
            for i in range(iterations):
                start_time = time.time()
                for array in arrays:
                    array.from_gpu()
                elapsed = time.time() - start_time
                gpu_to_cpu_times.append(elapsed)
                self.results_text.insert(tk.END, f"  Iteration {i+1}: {elapsed:.4f} seconds\n")
                self.results_text.see(tk.END)
                self.update()
            
            avg_gpu_to_cpu = sum(gpu_to_cpu_times) / len(gpu_to_cpu_times)
            self.results_text.insert(tk.END, f"\nAverage GPU to CPU: {avg_gpu_to_cpu:.4f} seconds\n\n")
            
            # Calculate bandwidth
            total_bytes = np.prod(shape) * np.dtype(np.float32).itemsize * num_arrays
            total_mb = total_bytes / (1024*1024)
            
            cpu_to_gpu_bandwidth = total_mb / avg_cpu_to_gpu
            gpu_to_cpu_bandwidth = total_mb / avg_gpu_to_cpu
            
            self.results_text.insert(tk.END, f"Data size per transfer: {total_mb:.2f} MB\n")
            self.results_text.insert(tk.END, f"CPU to GPU bandwidth: {cpu_to_gpu_bandwidth:.2f} MB/s\n")
            self.results_text.insert(tk.END, f"GPU to CPU bandwidth: {gpu_to_cpu_bandwidth:.2f} MB/s\n")
            
            # Clean up
            for array in arrays:
                self.memory_manager.free(array.identifier)
            
            self.update_array_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Benchmark failed: {e}")
    
    def run_allocation_benchmark(self):
        """Run benchmark for allocation/deallocation operations"""
        try:
            # Get benchmark parameters
            shape_str = self.bench_size_var.get()
            shape = tuple(int(dim.strip()) for dim in shape_str.split(','))
            num_arrays = int(self.bench_num_arrays_var.get())
            iterations = int(self.bench_iterations_var.get())
            
            # Clear results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Running allocation benchmark...\n\n")
            
            # CPU allocation timing
            self.results_text.insert(tk.END, "CPU Allocation Times:\n")
            cpu_alloc_times = []
            
            for i in range(iterations):
                arrays = []
                start_time = time.time()
                for j in range(num_arrays):
                    name = f"bench_alloc_{j}_{i}"
                    arrays.append(HybridArray(self.memory_manager, name, shape, np.float32, MemoryLocation.CPU))
                elapsed = time.time() - start_time
                cpu_alloc_times.append(elapsed)
                self.results_text.insert(tk.END, f"  Iteration {i+1}: {elapsed:.4f} seconds\n")
                self.results_text.see(tk.END)
                self.update()
                
                # Clean up
                for array in arrays:
                    self.memory_manager.free(array.identifier)
            
            avg_cpu_alloc = sum(cpu_alloc_times) / len(cpu_alloc_times)
            self.results_text.insert(tk.END, f"\nAverage CPU allocation: {avg_cpu_alloc:.4f} seconds\n\n")
            
            # GPU allocation timing
            self.results_text.insert(tk.END, "GPU Allocation Times:\n")
            gpu_alloc_times = []
            
            for i in range(iterations):
                arrays = []
                start_time = time.time()
                for j in range(num_arrays):
                    name = f"bench_alloc_{j}_{i}"
                    arrays.append(HybridArray(self.memory_manager, name, shape, np.float32, MemoryLocation.GPU))
                elapsed = time.time() - start_time
                gpu_alloc_times.append(elapsed)
                self.results_text.insert(tk.END, f"  Iteration {i+1}: {elapsed:.4f} seconds\n")
                self.results_text.see(tk.END)
                self.update()
                
                # Clean up
                for array in arrays:
                    self.memory_manager.free(array.identifier)
            
            avg_gpu_alloc = sum(gpu_alloc_times) / len(gpu_alloc_times)
            self.results_text.insert(tk.END, f"\nAverage GPU allocation: {avg_gpu_alloc:.4f} seconds\n\n")
            
            # Deallocation timing
            self.results_text.insert(tk.END, "Deallocation Times:\n")
            dealloc_times = []
            
            for i in range(iterations):
                arrays = []
                # Create arrays first
                for j in range(num_arrays):
                    name = f"bench_dealloc_{j}_{i}"
                    arrays.append(HybridArray(self.memory_manager, name, shape, np.float32, MemoryLocation.BOTH))
                
                # Time deallocation
                start_time = time.time()
                for array in arrays:
                    self.memory_manager.free(array.identifier)
                elapsed = time.time() - start_time
                dealloc_times.append(elapsed)
                self.results_text.insert(tk.END, f"  Iteration {i+1}: {elapsed:.4f} seconds\n")
                self.results_text.see(tk.END)
                self.update()
            
            avg_dealloc = sum(dealloc_times) / len(dealloc_times)
            self.results_text.insert(tk.END, f"\nAverage deallocation: {avg_dealloc:.4f} seconds\n\n")
            
            # Calculate rates
            total_bytes = np.prod(shape) * np.dtype(np.float32).itemsize * num_arrays
            total_mb = total_bytes / (1024*1024)
            
            cpu_alloc_rate = total_mb / avg_cpu_alloc
            gpu_alloc_rate = total_mb / avg_gpu_alloc
            dealloc_rate = total_mb / avg_dealloc
            
            self.results_text.insert(tk.END, f"Data size per operation: {total_mb:.2f} MB\n")
            self.results_text.insert(tk.END, f"CPU allocation rate: {cpu_alloc_rate:.2f} MB/s\n")
            self.results_text.insert(tk.END, f"GPU allocation rate: {gpu_alloc_rate:.2f} MB/s\n")
            self.results_text.insert(tk.END, f"Deallocation rate: {dealloc_rate:.2f} MB/s\n")
            
            self.update_array_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Benchmark failed: {e}")
    
    def toggle_monitoring(self):
        """Start or stop memory usage monitoring"""
        self.monitoring = not self.monitoring
        
        if self.monitoring:
            self.monitor_button.config(text="Stop Monitoring")
            self.schedule_monitoring_update()
        else:
            self.monitor_button.config(text="Start Monitoring")
    
    def schedule_monitoring_update(self):
        """Schedule the next monitoring update"""
        if not self.monitoring:
            return
        
        try:
            interval = int(self.update_interval_var.get())
            self.update_memory_stats()
            self.after(interval, self.schedule_monitoring_update)
        except ValueError:
            messagebox.showwarning("Warning", "Invalid update interval")
            self.monitoring = False
            self.monitor_button.config(text="Start Monitoring")
    
    def clear_monitoring(self):
        """Clear monitoring history"""
        self.memory_history = {
            'cpu_used': [],
            'gpu_used': []
        }
        self.update_plot()
    
    def update_plot(self):
        """Update the memory usage plot"""
        self.ax.clear()
        
        x = list(range(len(self.memory_history['cpu_used'])))
        
        if x:  # Only plot if we have data
            self.ax.plot(x, self.memory_history['cpu_used'], 'b-', label='CPU Usage (MB)')
            self.ax.plot(x, self.memory_history['gpu_used'], 'r-', label='GPU Usage (MB)')
            self.ax.set_xlabel('Sample')
            self.ax.set_ylabel('Memory Usage (MB)')
            self.ax.legend()
            self.ax.grid(True)
        
        self.canvas.draw()


# Example CUDA operations
def simulate_cuda_ops(memory_manager):
    """
    A function to demonstrate CUDA operations with the memory manager.
    In a real implementation, these would call actual CUDA kernels.
    """
    # Create two arrays
    array1 = HybridArray(memory_manager, "array1", (1000, 1000), np.float32, MemoryLocation.CPU)
    array2 = HybridArray(memory_manager, "array2", (1000, 1000), np.float32, MemoryLocation.CPU)
    
    # Initialize with data
    array1.array[:] = np.random.random((1000, 1000)).astype(np.float32)
    array2.array[:] = np.random.random((1000, 1000)).astype(np.float32)
    
    # Transfer to GPU
    array1.to_gpu()
    array2.to_gpu()
    
    # In a real implementation, execute_gpu_kernel would call an actual CUDA kernel
    # execute_gpu_kernel(array1, "MatrixMultiplyKernel")
    
    # Instead, we'll simulate a computation result
    result = HybridArray(memory_manager, "result", (1000, 1000), np.float32, MemoryLocation.GPU)
    
    # Simulate computation on GPU
    # In a real implementation, this would be done by a CUDA kernel
    # For demonstration, let's get the pointers and reference we'd pass to such a kernel
    array1_ptr = array1.get_gpu_ptr()
    array2_ptr = array2.get_gpu_ptr()
    result_ptr = result.get_gpu_ptr()
    
    print(f"CUDA operation with pointers: {array1_ptr}, {array2_ptr} -> {result_ptr}")
    
    # Transfer result back to CPU for verification
    result.from_gpu()
    
    # Generate some fake result data for demonstration
    result.array[:] = np.matmul(array1.array, array2.array)
    
    print(f"Result sum: {np.sum(result.array)}")
    
    # Clean up
    memory_manager.free("array1")
    memory_manager.free("array2")
    memory_manager.free("result")


# Example usage
def main():
    """Main function demonstrating usage in a script"""
    print("Hybrid Memory Manager Python Interface")
    print("======================================")
    
    # Create the memory manager
    manager = HybridMemoryManager()
    
    try:
        # Simulate some CUDA operations
        print("\nSimulating CUDA operations...")
        simulate_cuda_ops(manager)
        
        # Run a simple benchmark
        print("\nRunning a simple benchmark...")
        shape = (5000, 5000)
        
        # Create a large array
        print(f"Creating a {shape} float32 array...")
        array = HybridArray(manager, "benchmark", shape, np.float32, MemoryLocation.CPU)
        
        # Initialize with random data
        print("Filling with random data...")
        array.array[:] = np.random.random(shape).astype(np.float32)
        
        # Measure CPU to GPU transfer time
        print("Transferring to GPU...")
        start = time.time()
        array.to_gpu()
        elapsed = time.time() - start
        print(f"CPU to GPU transfer time: {elapsed:.4f} seconds")
        
        # Measure GPU to CPU transfer time
        print("Transferring back to CPU...")
        start = time.time()
        array.from_gpu()
        elapsed = time.time() - start
        print(f"GPU to CPU transfer time: {elapsed:.4f} seconds")
        
        # Clean up
        manager.free("benchmark")
        
        print("\nBenchmark complete!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nLaunching GUI application...")
    app = HybridMemoryManagerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
