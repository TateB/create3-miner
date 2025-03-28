export const safeProxyFactoryAbi = [
  {
    anonymous: false,
    inputs: [
      {
        indexed: false,
        internalType: "contract GnosisSafeProxy",
        name: "proxy",
        type: "address",
      },
      {
        indexed: false,
        internalType: "address",
        name: "singleton",
        type: "address",
      },
    ],
    name: "ProxyCreation",
    type: "event",
  },
  {
    inputs: [
      { internalType: "address", name: "_singleton", type: "address" },
      { internalType: "bytes", name: "initializer", type: "bytes" },
      { internalType: "uint256", name: "saltNonce", type: "uint256" },
    ],
    name: "calculateCreateProxyWithNonceAddress",
    outputs: [
      {
        internalType: "contract GnosisSafeProxy",
        name: "proxy",
        type: "address",
      },
    ],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { internalType: "address", name: "singleton", type: "address" },
      { internalType: "bytes", name: "data", type: "bytes" },
    ],
    name: "createProxy",
    outputs: [
      {
        internalType: "contract GnosisSafeProxy",
        name: "proxy",
        type: "address",
      },
    ],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { internalType: "address", name: "_singleton", type: "address" },
      { internalType: "bytes", name: "initializer", type: "bytes" },
      { internalType: "uint256", name: "saltNonce", type: "uint256" },
      {
        internalType: "contract IProxyCreationCallback",
        name: "callback",
        type: "address",
      },
    ],
    name: "createProxyWithCallback",
    outputs: [
      {
        internalType: "contract GnosisSafeProxy",
        name: "proxy",
        type: "address",
      },
    ],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      { internalType: "address", name: "_singleton", type: "address" },
      { internalType: "bytes", name: "initializer", type: "bytes" },
      { internalType: "uint256", name: "saltNonce", type: "uint256" },
    ],
    name: "createProxyWithNonce",
    outputs: [
      {
        internalType: "contract GnosisSafeProxy",
        name: "proxy",
        type: "address",
      },
    ],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [],
    name: "proxyCreationCode",
    outputs: [{ internalType: "bytes", name: "", type: "bytes" }],
    stateMutability: "pure",
    type: "function",
  },
  {
    inputs: [],
    name: "proxyRuntimeCode",
    outputs: [{ internalType: "bytes", name: "", type: "bytes" }],
    stateMutability: "pure",
    type: "function",
  },
] as const;
