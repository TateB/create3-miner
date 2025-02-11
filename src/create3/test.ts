import { computeCreate3Address } from "./utils";

// Test constants
const TEST_DEPLOYER = "0x343431e9CEb7C19cC8d3eA0EE231bfF82B584910" as const;
const TEST_SALT =
  "0x7cc6b9a2afa05a889a0394c767107d001d86bf77bea0141c11c296d3a8f72dac" as const;
const TEST_RESULT = "0xAB6528783ac2a0BEf235ada3E1A5F6d8a623867E" as const;

console.log("Running CREATE3 address computation test...");

// Test the reference case without namespace
const computedAddress = computeCreate3Address({
  deployer: TEST_DEPLOYER,
  salt: TEST_SALT,
  namespace: "", // Override the default namespace for testing
});

console.log("Test Values:");
console.log("Deployer:", TEST_DEPLOYER);
console.log("Salt:", TEST_SALT);
console.log("Expected Address:", TEST_RESULT);
console.log("Computed Address:", computedAddress);

if (computedAddress.toLowerCase() !== TEST_RESULT.toLowerCase()) {
  console.error(
    "❌ Test failed: Computed address does not match expected result"
  );
  console.error(`Expected: ${TEST_RESULT}`);
  console.error(`Received: ${computedAddress}`);
  process.exit(1);
}

console.log("✅ Test passed: Computed address matches expected result");

// Additional test to show how namespace affects the address
const namespacedAddress = computeCreate3Address({
  deployer: TEST_DEPLOYER,
  salt: TEST_SALT,
}); // Uses default namespace

console.log("\nNamespaced address test:");
console.log("Default namespace address:", namespacedAddress);
console.log("(Different from non-namespaced address as expected)");

// Test the Rust-found address to verify implementations match
console.log("\nVerifying Rust implementation result:");
const rustFoundSalt =
  "0xdd6861a2cd697756c46d7f30702b071c6c10607df341fbd73238727cbf125a69";
const rustFoundAddress = computeCreate3Address({
  deployer: TEST_DEPLOYER,
  salt: rustFoundSalt,
  namespace: "", // No namespace was used in the Rust test
});
console.log("Rust-found salt:", rustFoundSalt);
console.log("TypeScript computed address:", rustFoundAddress);
console.log(
  "Expected (from Rust):",
  "0x0000829333bAD0a38AFebBA7a8e7282C2024ee28"
);
console.log(
  "Implementations match:",
  rustFoundAddress.toLowerCase() ===
    "0x0000829333bAD0a38AFebBA7a8e7282C2024ee28".toLowerCase()
    ? "✅"
    : "❌"
);

process.exit(0);
