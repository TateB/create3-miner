import type { Address } from "viem";

export type WorkerIncomingMessage =
  | {
      type: "status";
      data: "start" | "stop";
    }
  | {
      type: "assign";
      data: number;
    };

export type WorkerOutgoingMessage =
  | {
      type: "failure";
      data: {
        amount: number;
      };
    }
  | {
      type: "success";
      data: {
        address: Address;
        salt: bigint;
      };
    };
