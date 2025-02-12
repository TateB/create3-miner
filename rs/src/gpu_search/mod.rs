#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "metal")]
mod metal_search;

#[derive(Debug)]
pub enum GpuBackend {
    #[cfg(feature = "metal")]
    Metal,
    Cuda,
}

pub struct GpuVanitySearch {
    backend: GpuBackend,
    #[cfg(feature = "metal")]
    metal_search: Option<metal_search::GpuVanitySearch>,
    cuda_search: Option<cuda::GpuVanitySearch>,
}

impl GpuVanitySearch {
    pub fn new() -> Option<Self> {
        // Try CUDA first, then fall back to Metal
        if let Some(cuda_search) = cuda::GpuVanitySearch::new() {
            println!("Using CUDA GPU backend");
            return Some(Self {
                backend: GpuBackend::Cuda,
                #[cfg(feature = "metal")]
                metal_search: None,
                cuda_search: Some(cuda_search),
            });
        } 
        #[cfg(feature = "metal")]
        {
            if let Some(metal_search) = metal_search::GpuVanitySearch::new() {
                println!("Using Metal GPU backend");
                return Some(Self {
                    backend: GpuBackend::Metal,
                    metal_search: Some(metal_search),
                    cuda_search: None,
                });
            }
        }
        println!("No GPU backend available");
        None
    }

    pub fn search(
        &self,
        deployer: &[u8],
        prefix: &str,
        namespace: &str,
        initial_salt: &[u8],
    ) -> Option<Vec<u8>> {
        match self.backend {
            GpuBackend::Cuda => {
                self.cuda_search
                    .as_ref()
                    .and_then(|cs| cs.search(deployer, prefix, namespace, initial_salt))
            }
            #[cfg(feature = "metal")]
            GpuBackend::Metal => {
                self.metal_search
                    .as_ref()
                    .and_then(|ms| ms.search(deployer, prefix, namespace, initial_salt))
                    .map(|salt| salt.to_vec())
            }
        }
    }
} 