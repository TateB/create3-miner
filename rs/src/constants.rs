use alloy_primitives::Address;

pub const CREATE3_FACTORY: &str = "0x004ee012d77c5d0e67d861041d11824f51b590fb";
pub const FACTORY_BYTECODE: &str = "0x67363d3d37363d34f03d5260086018f3";

// Helper function to get CREATE3_FACTORY as Address
pub fn create3_factory() -> Address {
    CREATE3_FACTORY.parse().expect("Invalid factory address")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const TEST_NAMESPACE: &str = "L2ReverseRegistrar v1.0.0";
    pub(crate) const TEST_DEPLOYER: &str = "0x343431e9CEb7C19cC8d3eA0EE231bfF82B584910";
    pub(crate) const TEST_SALT: &str =
        "0x7cc6b9a2afa05a889a0394c767107d001d86bf77bea0141c11c296d3a8f72dac";

    // Helper function to get TEST_DEPLOYER as Address
    pub(crate) fn test_deployer() -> Address {
        TEST_DEPLOYER
            .parse()
            .expect("Invalid test deployer address")
    }
}
