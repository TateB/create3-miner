import {
  type Address,
  type Hex,
  concatHex,
  encodePacked,
  keccak256,
  stringToHex,
} from "viem";
import {
  CREATE3_FACTORY,
  FACTORY_BYTECODE,
  PREFIX,
  SALT_NAMESPACE,
  SUFFIX,
} from "./constants";

export const createSeed = (): Hex =>
  keccak256(
    `0x${Math.random().toString(16).slice(2)}${Date.now().toString(16)}`
  );

let failuresSinceLastCheckpoint = 0;
let lastCheckpoint = performance.now();

export const compute = async ({
  ref,
  deployer,
  seed,
  namespace,
}: {
  ref: { running: boolean };
  deployer: Address;
  seed: Hex;
  namespace: string;
}): Promise<{ address: Address; vanillaSalt: Hex; namespacedSalt: Hex }> => {
  if (!ref.running)
    return { address: "0x", vanillaSalt: "0x", namespacedSalt: "0x" };

  const address = computeCreate3Address({
    deployer,
    salt: seed,
  });

  if (
    address.toLowerCase().startsWith(`0x${PREFIX.toLowerCase()}`) &&
    address.toLowerCase().endsWith(SUFFIX.toLowerCase())
  ) {
    return {
      address,
      vanillaSalt: seed,
      namespacedSalt: createArbitraryNamespacedSalt({
        namespace,
        salt: seed,
      }),
    };
  }

  failuresSinceLastCheckpoint++;
  if (lastCheckpoint < performance.now() - 100) {
    postMessage({
      type: "failure",
      data: { amount: failuresSinceLastCheckpoint },
    });
    failuresSinceLastCheckpoint = 0;
    lastCheckpoint = performance.now();
  }

  await new Promise((resolve) => process.nextTick(resolve));

  return compute({
    ref,
    deployer,
    seed: createSeed(),
    namespace,
  });
};

export const computeCreate2Address = ({
  deployer,
  salt,
  bytecode,
}: {
  deployer: Address;
  salt: Hex;
  bytecode: Hex;
}): Address => {
  // Compute CREATE2 address according to EIP-1014
  // keccak256(0xff ++ deployer ++ salt ++ keccak256(bytecode))[12:]
  const initCodeHash = keccak256(bytecode);
  const create2Input = encodePacked(
    ["bytes1", "address", "bytes32", "bytes32"],
    ["0xff", deployer, salt, initCodeHash]
  );
  return `0x${keccak256(create2Input).slice(26)}` as Address;
};

export const createArbitraryNamespacedSalt = ({
  namespace,
  salt,
}: {
  namespace: string;
  salt: Hex;
}) => keccak256(concatHex([salt, stringToHex(namespace)]));

export const computeCreate3Address = ({
  deployer,
  salt,
  namespace = SALT_NAMESPACE,
}: {
  deployer: Address;
  salt: Hex;
  namespace?: string;
}): Address => {
  // First hash the salt with the deployer address and namespace if provided
  const arbitraryNamespacedSalt = namespace
    ? createArbitraryNamespacedSalt({ namespace, salt })
    : salt;
  const namespacedSalt = keccak256(
    encodePacked(["address", "bytes32"], [deployer, arbitraryNamespacedSalt])
  );

  // Then compute the proxy address using CREATE2, but using the CREATE3_FACTORY as deployer
  const proxyAddress = computeCreate2Address({
    deployer: CREATE3_FACTORY,
    salt: namespacedSalt,
    bytecode: FACTORY_BYTECODE,
  });

  // Finally compute the contract address that will be deployed by the proxy
  // This follows the RLP encoding rules for contract addresses created by CREATE
  // prefix ++ address ++ nonce, where:
  // prefix = 0xd6 (0xc0 + 0x16), where 0x16 is length of: 0x94 ++ address ++ 0x01
  // 0x94 = 0x80 + 0x14 (0x14 is the length of an address)
  const rlpEncodedData = encodePacked(
    ["bytes1", "bytes1", "address", "bytes1"],
    ["0xd6", "0x94", proxyAddress, "0x01"]
  );

  return `0x${keccak256(rlpEncodedData).slice(26)}` as Address;
};
