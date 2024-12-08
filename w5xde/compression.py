import torch

class FastCompression:
    """Fast compression optimized for gradient data using LZ4-like algorithm."""
    
    def __init__(self, window_size=64*1024):
        self.window_size = window_size
        self.hash_table = {}
    
    def _hash(self, data, pos):
        """4-byte rolling hash."""
        if pos + 4 > len(data):
            return 0
        return (data[pos] | 
                (data[pos + 1] << 8) | 
                (data[pos + 2] << 16) | 
                (data[pos + 3] << 24))
    
    def compress(self, data: bytes) -> bytes:
        """Fast compression optimized for numerical data."""
        if len(data) < 4:
            return data
        
        compressed = bytearray()
        pos = 0
        self.hash_table.clear()
        
        while pos < len(data):
            # Look for matches
            best_len = 3  # Minimum match length
            best_offset = 0
            
            if pos + best_len <= len(data):
                hash_val = self._hash(data, pos)
                if hash_val in self.hash_table:
                    prev_pos = self.hash_table[hash_val]
                    offset = pos - prev_pos
                    
                    if offset <= self.window_size:
                        # Find match length
                        match_len = 0
                        while (pos + match_len < len(data) and 
                               prev_pos + match_len < pos and 
                               data[pos + match_len] == data[prev_pos + match_len] and
                               match_len < 255):
                            match_len += 1
                        
                        if match_len >= best_len:
                            best_len = match_len
                            best_offset = offset
            
            # Store position in hash table
            if pos + 4 <= len(data):
                self.hash_table[hash_val] = pos
            
            # Output token
            if best_offset:
                # Match found
                token = (0x80 | (best_len - 3))  # Set high bit for match
                compressed.append(token)
                # Store offset in 2 bytes, little endian
                compressed.extend(best_offset.to_bytes(2, 'little'))
                pos += best_len
            else:
                # Literal
                compressed.append(data[pos])
                pos += 1
        
        return bytes(compressed)
    
    def decompress(self, data: bytes) -> bytes:
        """Fast decompression."""
        if len(data) < 4:
            return data
        
        decompressed = bytearray()
        pos = 0
        
        while pos < len(data):
            token = data[pos]
            pos += 1
            
            if token & 0x80:  # Match
                length = (token & 0x7F) + 3
                offset = int.from_bytes(data[pos:pos+2], 'little')
                pos += 2
                
                start = len(decompressed) - offset
                for i in range(length):
                    decompressed.append(decompressed[start + i])
            else:  # Literal
                decompressed.append(token)
        
        return bytes(decompressed)

class GradientCompression:
    def __init__(self, bits=8, scale_method='dynamic'):
        self.bits = bits
        self.scale_method = scale_method
        self.max_val = 2**(bits-1) - 1
        self.min_val = -(2**(bits-1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.scale_factor = float(self.max_val)
        self.buffers = {}
        
        self.compressor = FastCompression()
    
    def _get_buffer(self, shape, grad_id):
        key = (grad_id, shape)
        if key not in self.buffers:
            self.buffers[key] = {
                'error': torch.zeros(shape, dtype=torch.float32, device=self.device),
                'quantized': torch.zeros(shape, dtype=torch.int8, device=self.device),
                'temp': torch.zeros(shape, dtype=torch.float32, device=self.device)
            }
        return self.buffers[key]
    
    def compress(self, gradients, grad_id=None):
        """Fast compression with minimal memory operations."""
        compressed_grads = []
        scales = []
        
        for i, grad in enumerate(gradients):
            if grad is None:
                compressed_grads.append(None)
                scales.append(None)
                continue
            
            buffer = self._get_buffer(grad.shape, f"{grad_id}_{i}")
            
            with torch.no_grad():
                torch.add(grad.to(self.device), buffer['error'], out=buffer['temp'])
                if self.scale_method == 'dynamic':
                    scale = buffer['temp'].abs().max().clamp_(min=1e-8)
                else:
                    scale = 1.0
                
                # Quantize in-place
                buffer['temp'].div_(scale).mul_(self.scale_factor)
                buffer['temp'].round_().clamp_(self.min_val, self.max_val)
                buffer['quantized'].copy_(buffer['temp'])
                
                # Update error feedback
                buffer['temp'].mul_(scale / self.scale_factor)
                torch.sub(grad.to(self.device), buffer['temp'], out=buffer['error'])
                
                # Apply fast compression to quantized data
                quantized_data = buffer['quantized'].cpu()
                compressed_data = self.compressor.compress(quantized_data.numpy().tobytes())
                
                # Store compressed form
                compressed_grads.append(compressed_data)
                scales.append(scale.item())
        
        return compressed_grads, scales
    
    def decompress(self, data: bytes) -> bytes:
        """Safer decompression with bounds checks."""
        if len(data) < 4:
            return data

        decompressed = bytearray()
        pos = 0

        while pos < len(data):
            token = data[pos]
            pos += 1

            if token & 0x80:  # Match
                length = (token & 0x7F) + 3
                if pos + 2 > len(data):
                    raise ValueError("Malformed compressed data: insufficient bytes for offset")
                offset = int.from_bytes(data[pos:pos + 2], 'little')
                pos += 2

                start = len(decompressed) - offset
                if start < 0 or start + length > len(decompressed):
                    raise ValueError(f"Invalid offset or length during decompression (offset={offset}, length={length})")
                
                for i in range(length):
                    decompressed.append(decompressed[start + i])
            else:  # Literal
                if pos > len(data):
                    raise ValueError("Malformed compressed data: unexpected end of literals")
                decompressed.append(token)

        return bytes(decompressed)