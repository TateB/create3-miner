import os from "os";
import type { WorkerOutgoingMessage } from "./types";

const workerCount = Math.floor(os.cpus().length);

console.log(`starting ${workerCount} workers`);

const workers = Array.from(
  { length: workerCount },
  (_) => new Worker(new URL("./worker.ts", import.meta.url))
);

let failuresSinceLastCheckpoint = 0;
let lastCheckpoint = performance.now();

const listener = (event: MessageEvent<WorkerOutgoingMessage>) => {
  if (event.data.type === "success") {
    console.log(event.data.data);
    return process.exit(0);
  }

  failuresSinceLastCheckpoint += event.data.data.amount;
  if (lastCheckpoint > performance.now() - 1000) return;

  console.log(`addresses per second: ${failuresSinceLastCheckpoint}`);
  failuresSinceLastCheckpoint = 0;
  lastCheckpoint = performance.now();
};

for (let i = 0; i < workerCount; i++) {
  const worker = workers[i];
  worker.onmessage = listener;
  worker.postMessage({ type: "assign", data: i });
  worker.postMessage({ type: "status", data: "start" });
}

console.log("all workers started");
