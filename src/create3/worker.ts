import type { WorkerIncomingMessage } from "../types";
import { DEPLOYER_ADDRESS, SALT_NAMESPACE } from "./constants";
import { compute, createSeed } from "./utils";

declare var self: Worker;

let ref = { running: false };
let name = 0;

self.onmessage = async (event: MessageEvent<WorkerIncomingMessage>) => {
  const msg = event.data;

  if (msg.type === "assign") {
    name = msg.data;
    return;
  }

  if (msg.data === "start") {
    console.log(`worker id: ${name} started`);
    ref.running = true;
    const result = await compute({
      ref,
      deployer: DEPLOYER_ADDRESS,
      seed: createSeed(),
      namespace: SALT_NAMESPACE,
    });

    if (result.address !== "0x") {
      postMessage({
        type: "success",
        data: result,
      });
    }
  } else if (msg.data === "stop") {
    console.log(`worker id: ${name} stopped`);
    ref.running = false;
  }
};

export default {} as unknown as Blob;
