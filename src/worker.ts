import { encodePacked, hexToBigInt } from "viem";
import {
  GNOSIS_SAFE_PROXY_CREATION_CODE,
  OWNER_ADDRESSES,
  SINGLETON_L2,
  THRESHOLD,
} from "./constants";
import type { WorkerIncomingMessage } from "./types";
import { compute, createSeed, generateArgInitialisers } from "./utils";

declare var self: Worker;

let ref = { running: false };
let name = 0;

const argInitialisers = generateArgInitialisers({
  owners: OWNER_ADDRESSES,
  threshold: THRESHOLD,
});
const singletonFactory = hexToBigInt(SINGLETON_L2);
const bytecode = encodePacked(
  ["bytes", "uint256"],
  [GNOSIS_SAFE_PROXY_CREATION_CODE, singletonFactory]
);

self.onmessage = async (event: MessageEvent<WorkerIncomingMessage>) => {
  const msg = event.data;

  if (msg.type === "assign") {
    name = msg.data;
    return;
  }

  if (msg.data === "start") {
    console.log(`worker id: ${name} started`);
    ref.running = true;
    const { address, salt } = await compute({
      ref,
      argInitialisers,
      bytecode,
      seed: createSeed(),
    });
    if (address !== "0x") {
      postMessage({
        type: "success",
        data: { address, salt, argInitialisers },
      });
    }
  } else if (msg.data === "stop") {
    console.log(`worker id: ${name} stopped`);
    ref.running = false;
  }
};

export default {} as unknown as Blob;
