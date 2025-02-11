use alloy_primitives::Address;

pub const CREATE3_FACTORY: &str = "0x004ee012d77c5d0e67d861041d11824f51b590fb";
pub const FACTORY_BYTECODE: &str = "0x67363d3d37363d34f03d5260086018f3";

// Test constants
pub const TEST_DEPLOYER: &str = "0x343431e9CEb7C19cC8d3eA0EE231bfF82B584910";
pub const TEST_SALT: &str = "0x7cc6b9a2afa05a889a0394c767107d001d86bf77bea0141c11c296d3a8f72dac";

// Helper function to get CREATE3_FACTORY as Address
pub fn create3_factory() -> Address {
    CREATE3_FACTORY.parse().expect("Invalid factory address")
}

// Helper function to get TEST_DEPLOYER as Address
pub fn test_deployer() -> Address {
    TEST_DEPLOYER
        .parse()
        .expect("Invalid test deployer address")
}
