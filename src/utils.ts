import {
  concat,
  encodePacked,
  getContractAddress,
  hexToBigInt,
  keccak256,
  pad,
  toHex,
  zeroAddress,
  type Address,
  type Hex,
} from "viem";
import {
  FALLBACK_HANDLER,
  PREFIX,
  PROXY_FACTORY_L2,
  SUFFIX,
  ZERO,
} from "./constants";

export const generateArgInitialisers = ({
  owners,
  threshold,
}: {
  owners: Address[] | readonly Address[];
  threshold: number;
}) =>
  "b63e800d" + //Function signature
  "100".padStart(64, "0") + // Version
  threshold.toString().padStart(64, "0") + // Threshold
  zeroAddress.substring(2).padStart(64, "0") + // Address zero, TO
  pad(toHex(0x120 + 0x20 * owners.length))
    .substring(2)
    .padStart(64, "0") + // Data length
  FALLBACK_HANDLER.substring(2).padStart(64, "0") +
  zeroAddress.substring(2).padStart(64, "0") + // paymentToken
  ZERO.padStart(64, "0") + // payment
  zeroAddress.substring(2).padStart(64, "0") + // paymentReceiver
  owners.length.toString().padStart(64, "0") + // owners.length
  owners.map((owner): string => owner.substring(2).padStart(64, "0")).join("") + // owners
  ZERO.padStart(64, "0"); // data.length

export const createSeed = () =>
  hexToBigInt(
    keccak256(concat([toHex("poggers"), toHex(Math.random().toString())]))
  );

let failuresSinceLastCheckpoint = 0;
let lastCheckpoint = performance.now();

export const compute = async ({
  ref,
  argInitialisers,
  bytecode,
  seed,
}: {
  ref: { running: boolean };
  argInitialisers: string;
  bytecode: Hex;
  seed: bigint;
}): Promise<{ address: Address; salt: bigint }> => {
  if (!ref.running) return { address: "0x", salt: 0n };

  const salt = keccak256(
    encodePacked(
      ["bytes", "uint256"],
      [keccak256(`0x${argInitialisers}`), seed]
    )
  );
  const addrCreate2 = getContractAddress({
    bytecode,
    from: PROXY_FACTORY_L2,
    opcode: "CREATE2",
    salt,
  });

  if (addrCreate2.startsWith(`0x${PREFIX}`) && addrCreate2.endsWith(SUFFIX))
    return { address: addrCreate2, salt: seed };

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
    argInitialisers,
    bytecode,
    seed: createSeed(),
  });
};
