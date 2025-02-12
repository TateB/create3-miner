use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
use rustacuda::launch;
use std::ffi::CString;
use std::cell::Cell;
use std::sync::Arc;

pub struct GpuVanitySearch {
    _context: Arc<Context>, // Underscore prefix indicates intentionally unused but necessary field
    module: Module,
    stream: Stream,
    iterations: Cell<u64>,
    time_taken: Cell<f64>,
    last_print: Cell<std::time::Instant>,
}

impl GpuVanitySearch {
    pub fn new() -> Option<Self> {
        match rustacuda::init(CudaFlags::empty()) {
            Ok(_) => (),
            Err(e) => {
                println!("Failed to initialize CUDA: {}", e);
                return None;
            }
        }

        let device = match Device::get_device(0) {
            Ok(device) => device,
            Err(e) => {
                println!("Failed to get CUDA device: {}", e);
                return None;
            }
        };

        println!("Found CUDA device: {}", device.name().unwrap());

        let context = match Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device) {
            Ok(ctx) => Arc::new(ctx),
            Err(e) => {
                println!("Failed to create CUDA context: {}", e);
                return None;
            }
        };

        // Load the CUDA module (compiled PTX)
        let ptx = include_str!("../shader/Keccak256.ptx");
        let ptx = CString::new(ptx).unwrap();
        let module = match Module::load_from_string(&ptx) {
            Ok(module) => module,
            Err(e) => {
                println!("Failed to load CUDA module: {}", e);
                return None;
            }
        };

        let stream = match Stream::new(StreamFlags::NON_BLOCKING, None) {
            Ok(stream) => stream,
            Err(e) => {
                println!("Failed to create CUDA stream: {}", e);
                return None;
            }
        };

        println!("Successfully initialized CUDA GPU support");

        Some(Self {
            _context: context,
            module,
            stream,
            iterations: Cell::new(0),
            time_taken: Cell::new(0.0),
            last_print: Cell::new(std::time::Instant::now()),
        })
    }

    pub fn search_with_threads(
        &self,
        deployer: &[u8],
        prefix: &str,
        namespace: &str,
        initial_salt: &[u8],
        thread_count: u32,
        block_count: u32,
    ) -> Option<Vec<u8>> {
        // Convert deployer address to bytes
        let deployer_bytes = if deployer.len() == 20 {
            deployer.to_vec()
        } else {
            // Assume it's a hex string with 0x prefix
            let hex_str = std::str::from_utf8(deployer)
                .expect("Invalid deployer address")
                .trim_start_matches("0x");
            hex::decode(hex_str).expect("Invalid hex in deployer address")
        };

        // Convert prefix string to individual hex values (0-15)
        let prefix_bytes: Vec<u8> = prefix
            .chars()
            .map(|c| c.to_digit(16).unwrap() as u8)
            .collect();

        // Allocate device memory
        let mut deployer_buf = DeviceBuffer::from_slice(&deployer_bytes).unwrap();
        let mut prefix_buf = DeviceBuffer::from_slice(&prefix_bytes).unwrap();
        let prefix_len = prefix_bytes.len() as u32;
        let mut prefix_len_buf = DeviceBuffer::from_slice(&[prefix_len]).unwrap();
        
        let ns_bytes = namespace.as_bytes();
        let mut ns_buf = DeviceBuffer::from_slice(ns_bytes).unwrap();
        let ns_len = ns_bytes.len() as u32;
        let mut ns_len_buf = DeviceBuffer::from_slice(&[ns_len]).unwrap();
        
        let mut initial_salt_buf = DeviceBuffer::from_slice(initial_salt).unwrap();
        let mut result_salt = vec![0u8; 32];
        let mut result_salt_buf = DeviceBuffer::from_slice(&result_salt).unwrap();
        let mut found = vec![0i32; 1];
        let mut found_buf = DeviceBuffer::from_slice(&found).unwrap();


        let function_name = CString::new("vanity_search").unwrap();
        let function = self.module.get_function(&function_name).unwrap();
        
        let start_time = std::time::Instant::now();

        unsafe {
            let stream = &self.stream;
            launch!(
                function<<<(block_count, 1, 1), (thread_count, 1, 1), 0, stream>>>(
                    deployer_buf.as_device_ptr(),
                    prefix_buf.as_device_ptr(),
                    prefix_len_buf.as_device_ptr(),
                    ns_buf.as_device_ptr(),
                    ns_len_buf.as_device_ptr(),
                    initial_salt_buf.as_device_ptr(),
                    result_salt_buf.as_device_ptr(),
                    found_buf.as_device_ptr()
                )
            ).unwrap();
        }

        self.stream.synchronize().unwrap();

        let end_time = std::time::Instant::now();
        let duration = end_time.duration_since(start_time);
        self.time_taken.set(self.time_taken.get() + duration.as_secs_f64());
        self.iterations.set(self.iterations.get() + (thread_count * block_count) as u64);
        
        // Check if we found a result
        found_buf.copy_to(&mut found).unwrap();

        if found[0] != 0 || self.last_print.get().elapsed() > std::time::Duration::from_secs(1) {
            let iterations_per_second = (self.iterations.get() as f64) / self.time_taken.get();
            println!(
                "CUDA: Iterations per second: {}",
                (iterations_per_second as u64)
                    .to_string()
                    .chars()
                    .rev()
                    .collect::<Vec<_>>()
                    .chunks(3)
                    .map(|chunk| chunk.iter().collect::<String>())
                    .collect::<Vec<_>>()
                    .join(",")
                    .chars()
                    .rev()
                    .collect::<String>()
            );
            println!("CUDA: Total time taken: {:?}s", self.time_taken.get());
            self.last_print.set(std::time::Instant::now());
        }

        if found[0] != 0 {
            result_salt_buf.copy_to(&mut result_salt).unwrap();
            Some(result_salt)
        } else {
            None
        }
    }
}

impl Drop for GpuVanitySearch {
    fn drop(&mut self) {
        // First synchronize the stream
        if let Err(e) = self.stream.synchronize() {
            println!("Warning: Failed to synchronize CUDA stream: {}", e);
        }

        // Create a new scope to ensure all CUDA resources are dropped before the context
        {
            // Drop the stream first
            let _ = std::mem::replace(&mut self.stream, unsafe { std::mem::zeroed() });
            
            // Drop the module
            let _ = std::mem::replace(&mut self.module, unsafe { std::mem::zeroed() });
        }

        // The context will be dropped last automatically
    }
} 