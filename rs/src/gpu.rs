use hex;
use metal::*;
use objc::rc::autoreleasepool;

fn is_debug() -> bool {
    std::env::var("DEBUG").is_ok()
}

fn dbg_println(args: std::fmt::Arguments) {
    if is_debug() {
        println!("{}", args);
    }
}

macro_rules! dbg_println {
    () => {
        dbg_println(format_args!(""))
    };
    ($($arg:tt)*) => {{
        dbg_println(format_args!($($arg)*))
    }};
}

pub struct GpuVanitySearch {
    device: Device,
    pipeline_state: metal::ComputePipelineState,
    command_queue: metal::CommandQueue,
    iterations: std::cell::Cell<u64>,
    time_taken: std::cell::Cell<f64>,
}

impl GpuVanitySearch {
    pub fn new() -> Option<Self> {
        autoreleasepool(|| {
            let device = Device::system_default()?;
            println!("Found Metal device: {}", device.name());

            let shader_src = include_str!("shader/Keccak256.metal");
            let library =
                match device.new_library_with_source(shader_src, &metal::CompileOptions::new()) {
                    Ok(lib) => lib,
                    Err(e) => {
                        println!("Failed to create Metal library: {}", e);
                        return None;
                    }
                };

            let function = match library.get_function("vanity_search", None) {
                Ok(f) => f,
                Err(e) => {
                    println!("Failed to get Metal function: {}", e);
                    return None;
                }
            };

            let pipeline_state = match device.new_compute_pipeline_state_with_function(&function) {
                Ok(p) => p,
                Err(e) => {
                    println!("Failed to create Metal pipeline: {}", e);
                    return None;
                }
            };

            let command_queue = device.new_command_queue();
            println!("GPU: Command queue created");

            println!("Successfully initialized Metal GPU support");

            Some(Self {
                device,
                pipeline_state,
                command_queue,
                iterations: std::cell::Cell::new(0),
                time_taken: std::cell::Cell::new(0.0),
            })
        })
    }

    pub fn search(
        &self,
        deployer: &[u8],
        prefix: &str,
        namespace: &str,
        initial_salt: &[u8],
    ) -> Option<&[u8]> {
        autoreleasepool(|| {
            let capture_scope =
                CaptureManager::shared().new_capture_scope_with_device(&self.device);
            let capture_descriptor = CaptureDescriptor::new();
            capture_descriptor.set_capture_scope(&capture_scope);
            capture_descriptor.set_output_url(std::path::Path::new(
                "/Users/tate/Development/playgrounds/safe-vanity/framecapture.gputrace",
            ));
            capture_descriptor.set_destination(MTLCaptureDestination::GpuTraceDocument);
            if std::env::var("METAL_CAPTURE_ENABLED").is_ok() {
                dbg_println!("Metal capture enabled.");
                CaptureManager::shared()
                    .start_capture(&capture_descriptor)
                    .expect("Failed to start capture");
                capture_scope.begin_scope();
            }
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

            dbg_println!("GPU: Searching for prefix bytes: {:?}", prefix_bytes);
            dbg_println!("GPU: Debug address bytes: {:02x?}", initial_salt);
            dbg_println!("GPU: Prefix length: {}", prefix_bytes.len());
            dbg_println!("GPU: Deployer bytes: {:?}", deployer_bytes);
            dbg_println!("GPU: Initial salt: {:?}", initial_salt);

            // Create buffers
            let deployer_buffer = self.device.new_buffer_with_data(
                deployer_bytes.as_ptr() as *const _,
                deployer_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            dbg_println!("GPU: Deployer buffer created");

            let prefix_buffer = self.device.new_buffer_with_data(
                prefix_bytes.as_ptr() as *const _,
                prefix_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            dbg_println!("GPU: Prefix buffer created");

            // Note: prefix_len is now the number of bytes, not characters
            let prefix_len = prefix_bytes.len() as u32;
            let prefix_len_buffer = self.device.new_buffer_with_data(
                &prefix_len as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            dbg_println!("GPU: Prefix length buffer created");

            // Create namespace buffer with at least 1 byte
            let namespace_buffer = if namespace.is_empty() {
                self.device
                    .new_buffer(1, MTLResourceOptions::StorageModeShared)
            } else {
                self.device.new_buffer_with_data(
                    namespace.as_ptr() as *const _,
                    namespace.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };

            dbg_println!("GPU: Namespace buffer created");

            let namespace_len = namespace.len() as u32;
            let namespace_len_buffer = self.device.new_buffer_with_data(
                &namespace_len as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            dbg_println!("GPU: Namespace length buffer created");

            let initial_salt_buffer = self.device.new_buffer_with_data(
                initial_salt.as_ptr() as *const _,
                initial_salt.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            dbg_println!("GPU: Initial salt buffer created");

            let result_buffer = self.device.new_buffer(
                std::mem::size_of::<u64>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            dbg_println!("GPU: Result buffer created");

            let found_buffer = self.device.new_buffer(
                std::mem::size_of::<bool>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            dbg_println!("GPU: Found buffer created");

            // Initialize found buffer to false
            unsafe {
                *(found_buffer.contents() as *mut bool) = false;
            }

            dbg_println!("GPU: Found buffer initialized to false");

            // Create command buffer
            let command_buffer = self.command_queue.new_command_buffer();

            dbg_println!("GPU: Command buffer created");

            // Create compute encoder and take ownership
            let compute_encoder = Some(command_buffer.new_compute_command_encoder());
            let encoder = compute_encoder.as_ref().unwrap();

            dbg_println!("GPU: Compute encoder created");

            // Set pipeline state and buffers
            encoder.set_compute_pipeline_state(&self.pipeline_state);

            dbg_println!("GPU: Pipeline state set");
            encoder.set_buffer(0, Some(&deployer_buffer), 0);
            dbg_println!("GPU: Deployer buffer set");
            encoder.set_buffer(1, Some(&prefix_buffer), 0);
            dbg_println!("GPU: Prefix buffer set");
            encoder.set_buffer(2, Some(&prefix_len_buffer), 0);
            dbg_println!("GPU: Prefix length buffer set");
            encoder.set_buffer(3, Some(&namespace_buffer), 0);
            dbg_println!("GPU: Namespace buffer set");
            encoder.set_buffer(4, Some(&namespace_len_buffer), 0);
            dbg_println!("GPU: Namespace length buffer set");
            encoder.set_buffer(5, Some(&initial_salt_buffer), 0);
            dbg_println!("GPU: Initial salt buffer set");
            encoder.set_buffer(6, Some(&result_buffer), 0);
            dbg_println!("GPU: Result buffer set");
            encoder.set_buffer(7, Some(&found_buffer), 0);
            dbg_println!("GPU: Found buffer set");
            dbg_println!("GPU: Buffers set");

            let thread_count = 64;
            let threadgroup_count = 65536;

            // Configure thread groups
            let threads_per_threadgroup = metal::MTLSize::new(thread_count, 1, 1);
            let threadgroups = metal::MTLSize::new(threadgroup_count, 1, 1); // Much larger search space

            dbg_println!("GPU: Thread groups configured");
            dbg_println!(
                "GPU: Max threads per threadgroup: {:?}",
                &self.pipeline_state.max_total_threads_per_threadgroup()
            );
            dbg_println!(
                "GPU: Thread execution width: {:?}",
                &self.pipeline_state.thread_execution_width()
            );

            // Dispatch work
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);

            dbg_println!("GPU: Dispatching work");

            // End encoding and commit
            encoder.end_encoding();
            let start_time = std::time::Instant::now();
            command_buffer.commit();

            dbg_println!("GPU: Waiting for command buffer to complete");

            command_buffer.wait_until_completed();

            let end_time = std::time::Instant::now();
            let duration = end_time.duration_since(start_time);
            self.time_taken
                .set(self.time_taken.get() + duration.as_secs_f64());
            dbg_println!("GPU: Command buffer completed in {:?}", duration);

            let status = command_buffer.status();
            dbg_println!("GPU: Command buffer status: {:?}", status);

            let error = command_buffer.error();
            dbg_println!("GPU: Command buffer error: {:?}", error);

            if std::env::var("METAL_CAPTURE_ENABLED").is_ok() {
                capture_scope.end_scope();
            }

            // Get result and debug info
            unsafe {
                let found = *(found_buffer.contents() as *const bool);
                let salt = std::slice::from_raw_parts(result_buffer.contents() as *const u8, 32);
                self.iterations
                    .set(self.iterations.get() + thread_count * threadgroup_count);
                let iterations_per_second = (self.iterations.get() as f64) / self.time_taken.get();
                println!(
                    "GPU: Iterations per second: {}",
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
                println!("GPU: Total time taken: {:?}s", self.time_taken.get());

                if is_debug() {
                    dbg_println!("GPU: Getting result and debug info");
                    dbg_println!("GPU: Salt value: {:02x?}", salt);
                    dbg_println!("GPU: Found: {}", found);
                    dbg_println!(
                        "GPU: Total iterations: {}",
                        self.iterations
                            .get()
                            .to_string()
                            .chars()
                            .rev()
                            .collect::<Vec<_>>()
                            .chunks(3)
                            .map(|chunk| chunk.iter().collect::<String>())
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
                if found {
                    Some(salt)
                } else {
                    None
                }
            }
        })
    }
}
