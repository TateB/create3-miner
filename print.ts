import { hexToBytes, getCreate2Address } from 'viem'

const ddpAddress = '0x13b0D85CcB8bf860b6b79AF3029fCA081AE9beF2'
const contractBytecode =
  '0x7f69476edfc91a3154b906b27ffc9db0856c3b7327c897efe8253aa4e7644074'

const ddpAddressBytes = hexToBytes(ddpAddress)
const contractBytecodeBytes = hexToBytes(contractBytecode)

console.log('DDP Address bytes:')
for (let i = 0; i < ddpAddressBytes.length; i += 8) {
  const row = Array.from(ddpAddressBytes)
    .slice(i, i + 8)
    .map((b) => '0x' + b.toString(16).padStart(2, '0'))
    .join(', ')
  console.log(row)
}

console.log('\nContract bytecode bytes:')
for (let i = 0; i < contractBytecodeBytes.length; i += 8) {
  const row = Array.from(contractBytecodeBytes)
    .slice(i, i + 8)
    .map((b) => '0x' + b.toString(16).padStart(2, '0'))
    .join(', ')
  console.log(row)
}

const create2Address = getCreate2Address({
  bytecodeHash: contractBytecode,
  from: ddpAddress,
  salt: '0xb0e9fcbc27b7000484e91b2ec674970cb0c378aa2d871d6bf99806aeaa7c9ead',
})

console.log('Create2 Address:', create2Address)
