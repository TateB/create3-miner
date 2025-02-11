mod constants;
mod create3;

use alloy_primitives::{keccak256, Address, B256};
use anyhow::Result;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::task;

use crate::create3::compute_create3_address;

fn create_random_salt() -> B256 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();
    let random = rand::random::<u64>().to_string();
    let combined = format!("{}{}", now, random);
    let hash = keccak256(combined.as_bytes()).to_vec();
    B256::from_slice(&hash)
}

struct Stats {
    attempts: Arc<AtomicU64>,
    last_print: Arc<Mutex<Instant>>,
}

impl Stats {
    fn new() -> Self {
        Self {
            attempts: Arc::new(AtomicU64::new(0)),
            last_print: Arc::new(Mutex::new(Instant::now())),
        }
    }

    fn increment(&self) {
        self.attempts.fetch_add(1, Ordering::Relaxed);

        let mut last_print = self.last_print.lock().unwrap();
        if last_print.elapsed() >= Duration::from_secs(1) {
            let attempts = self.attempts.swap(0, Ordering::Relaxed);
            println!("Attempts per second: {}", attempts);
            *last_print = Instant::now();
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        println!(
            "Usage: {} <deployer_address> <prefix> [namespace] [num_threads]",
            args[0]
        );
        std::process::exit(1);
    }

    let deployer: Address = args[1].parse().expect("Invalid deployer address");
    let prefix = args[2].to_lowercase();
    let namespace = args.get(3).map(|s| s.as_str()).unwrap_or("");
    let num_threads = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| num_cpus::get());

    println!(
        "Starting {} threads to search for vanity address",
        num_threads
    );
    println!("Deployer: {}", deployer);
    println!("Prefix: {}", prefix);
    println!("Namespace: {}", namespace);

    let stats = Arc::new(Stats::new());
    let running = Arc::new(AtomicU64::new(1));
    let mut handles = vec![];

    for _ in 0..num_threads {
        let stats = Arc::clone(&stats);
        let running = Arc::clone(&running);
        let deployer = deployer;
        let prefix = prefix.clone();
        let namespace = namespace.to_string();

        let handle = task::spawn(async move {
            while running.load(Ordering::Relaxed) == 1 {
                let salt = create_random_salt();
                let address = compute_create3_address(deployer, salt, Some(&namespace))
                    .expect("Failed to compute address");

                stats.increment();

                if format!("{:x}", address).starts_with(&prefix) {
                    println!("\nFound matching address!");
                    println!("Address: {}", address);
                    println!("Salt: {}", salt);
                    running.store(0, Ordering::Relaxed);
                    break;
                }
            }
        });

        handles.push(handle);
    }

    // Wait for any thread to find a result
    futures::future::join_all(handles).await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_random_salt_generation() {
        // Test that random salts are unique
        let mut salts = HashSet::new();
        for _ in 0..100 {
            let salt = create_random_salt();
            assert!(!salts.contains(&salt), "Generated duplicate salt");
            salts.insert(salt);
        }
    }

    #[test]
    fn test_stats_tracking() {
        let stats = Stats::new();

        // Test initial state
        assert_eq!(stats.attempts.load(Ordering::Relaxed), 0);

        // Test increment
        stats.increment();
        assert_eq!(stats.attempts.load(Ordering::Relaxed), 1);

        // Test multiple increments
        for _ in 0..10 {
            stats.increment();
        }
        assert_eq!(stats.attempts.load(Ordering::Relaxed), 11);

        // Test reset after elapsed time
        std::thread::sleep(Duration::from_secs(1));
        stats.increment();
        assert_eq!(stats.attempts.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_vanity_address_search() {
        // Test with a simple prefix that should be found quickly
        let deployer = "0x343431e9CEb7C19cC8d3eA0EE231bfF82B584910"
            .parse()
            .expect("Invalid test deployer address");
        let prefix = "0";
        let namespace = "";
        let num_threads = 2;

        let stats = Arc::new(Stats::new());
        let running = Arc::new(AtomicU64::new(1));
        let mut handles = vec![];

        for _ in 0..num_threads {
            let stats = Arc::clone(&stats);
            let running = Arc::clone(&running);
            let prefix = prefix.to_string();

            let handle = task::spawn(async move {
                let mut found_salt = None;
                while running.load(Ordering::Relaxed) == 1 {
                    let salt = create_random_salt();
                    let address = compute_create3_address(deployer, salt, Some(namespace))
                        .expect("Failed to compute address");

                    stats.increment();

                    if format!("{:x}", address).starts_with(prefix.as_str()) {
                        found_salt = Some(salt);
                        running.store(0, Ordering::Relaxed);
                        break;
                    }
                }
                found_salt
            });

            handles.push(handle);
        }

        let results = futures::future::join_all(handles).await;
        let found_salt = results.into_iter().find_map(|r| r.unwrap());

        assert!(found_salt.is_some(), "Failed to find matching address");

        // Verify the found salt produces an address with the correct prefix
        if let Some(salt) = found_salt {
            let address = compute_create3_address(deployer, salt, Some(namespace))
                .expect("Failed to compute address");
            assert!(
                format!("{:x}", address).starts_with(prefix),
                "Found address does not match prefix"
            );
        }
    }
}
