"""`payload_codec` に関する提出用処理。"""
from __future__ import annotations

import heapq

# Compact alphabet for huff91/base91 payloads.
# Use almost all ASCII control+printable bytes except '\n' and '\r'
# (line separators in generated C++ source).
_BASE91_EXTRA_CODES = tuple(c for c in range(1, 128) if c not in (10, 13))
_BASE91_ALPHABET = "".join(chr(c) for c in _BASE91_EXTRA_CODES)
_BASE91_ENC = _BASE91_ALPHABET
_BASE91_DEC = {ch: i for i, ch in enumerate(_BASE91_ALPHABET)}
_BASE91_RADIX = len(_BASE91_ALPHABET)
_BASE91_EXTRA = _BASE91_RADIX * _BASE91_RADIX - 8192

# Base122 illegal set (+ '?' to avoid trigraph conversion in C++ source).
_BASE122_ILLEGALS = (0, 10, 13, 34, 38, 63, 92)
_BASE122_ILLEGAL_TO_INDEX = {v: i for i, v in enumerate(_BASE122_ILLEGALS)}
_BASE122_SHORTENED = 0b111

PAYLOAD_CODEC_BASE91 = 91
PAYLOAD_CODEC_BASE122 = 122
PAYLOAD_CODEC_HUFF122 = 1122
PAYLOAD_CODEC_HUFF91 = 1091


def _encode_base91(data: bytes) -> str:
    """内部ヘルパー: `encode_base91` を実行する。

    Args:
        data (bytes): data の値。

    Returns:
        str: 計算結果。
    """
    b = 0
    n = 0
    out: list[str] = []
    for x in data:
        b |= x << n
        n += 8
        if n > 13:
            v = b & 8191
            if v < _BASE91_EXTRA:
                v = b & 16383
                b >>= 14
                n -= 14
            else:
                b >>= 13
                n -= 13
            out.append(_BASE91_ENC[v % _BASE91_RADIX])
            out.append(_BASE91_ENC[v // _BASE91_RADIX])
    if n:
        out.append(_BASE91_ENC[b % _BASE91_RADIX])
        if n > 7 or b >= _BASE91_RADIX:
            out.append(_BASE91_ENC[b // _BASE91_RADIX])
    return "".join(out)


def _decode_base91(text: str) -> bytes:
    """内部ヘルパー: `decode_base91` を実行する。

    Args:
        text (str): text の値。

    Returns:
        bytes: 計算結果。
    """
    v = -1
    b = 0
    n = 0
    out = bytearray()
    for ch in text:
        d = _BASE91_DEC.get(ch, -1)
        if d < 0:
            continue
        if v < 0:
            v = d
            continue
        v += d * _BASE91_RADIX
        b |= v << n
        n += 14 if (v & 8191) < _BASE91_EXTRA else 13
        while n > 7:
            out.append(b & 0xFF)
            b >>= 8
            n -= 8
        v = -1
    if v >= 0:
        out.append((b | (v << n)) & 0xFF)
    return bytes(out)


def _encode_base122(data: bytes) -> str:
    """内部ヘルパー: `encode_base122` を実行する。

    Args:
        data (bytes): data の値。

    Returns:
        str: 計算結果。
    """
    cur_index = 0
    cur_bit = 0
    n = len(data)
    out = bytearray()

    def get7() -> int | None:
        """`get7` を実行する。

        Returns:
            int | None: 計算結果。
        """
        nonlocal cur_index, cur_bit
        if cur_index >= n:
            return None
        first_byte = data[cur_index]
        first_part = ((0b11111110 >> cur_bit) & first_byte) << cur_bit
        first_part >>= 1
        cur_bit += 7
        if cur_bit < 8:
            return int(first_part)
        cur_bit -= 8
        cur_index += 1
        if cur_index >= n:
            return int(first_part)
        second_byte = data[cur_index]
        second_part = ((0xFF00 >> cur_bit) & second_byte) & 0xFF
        second_part >>= 8 - cur_bit
        return int(first_part | second_part)

    while True:
        bits = get7()
        if bits is None:
            break
        illegal_idx = _BASE122_ILLEGAL_TO_INDEX.get(bits, -1)
        if illegal_idx != -1:
            next_bits = get7()
            b1 = 0b11000010
            b2 = 0b10000000
            if next_bits is None:
                b1 |= (_BASE122_SHORTENED & 0b111) << 2
                next_bits = bits
            else:
                b1 |= (illegal_idx & 0b111) << 2
            b1 |= 1 if (next_bits & 0b01000000) else 0
            b2 |= next_bits & 0b00111111
            out.append(b1)
            out.append(b2)
        else:
            out.append(bits)
    return out.decode("utf-8")


def _decode_base122(text: str) -> bytes:
    """内部ヘルパー: `decode_base122` を実行する。

    Args:
        text (str): text の値。

    Returns:
        bytes: 計算結果。
    """
    illegals = _BASE122_ILLEGALS
    shortened = _BASE122_SHORTENED
    encoded = text.encode("utf-8")
    out = bytearray()
    cur_byte = 0
    bit_of_byte = 0

    def push7(seven: int) -> None:
        """`push7` を実行する。

        Args:
            seven (int): seven の値。
        """
        nonlocal cur_byte, bit_of_byte
        byte = (seven & 0x7F) << 1
        cur_byte |= (byte >> bit_of_byte) & 0xFF
        bit_of_byte += 7
        if bit_of_byte >= 8:
            out.append(cur_byte)
            bit_of_byte -= 8
            cur_byte = (byte << (7 - bit_of_byte)) & 0xFF

    i = 0
    n = len(encoded)
    while i < n:
        c0 = encoded[i]
        i += 1
        c = int(c0)
        if c > 127:
            if i >= n:
                break
            c1 = encoded[i]
            i += 1
            if (c1 & 0xC0) != 0x80:
                continue
            c = int(((c0 & 0x1F) << 6) | (c1 & 0x3F))
        if c > 127:
            illegal_index = (c >> 8) & 0b111
            if illegal_index != shortened:
                if illegal_index >= len(illegals):
                    continue
                push7(illegals[illegal_index])
            push7(c & 0x7F)
        else:
            push7(c)
    return bytes(out)


def _huff_code_lengths(data: bytes) -> list[int]:
    """内部ヘルパー: `huff_code_lengths` を実行する。

    Args:
        data (bytes): data の値。

    Returns:
        list[int]: 計算結果。
    """
    lengths = [0] * 256
    if not data:
        return lengths
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    heap: list[tuple[int, int, int]] = []
    order = 0
    for sym, w in enumerate(freq):
        if w > 0:
            heapq.heappush(heap, (w, order, sym))
            order += 1
    if len(heap) == 1:
        lengths[heap[0][2]] = 1
        return lengths
    nodes: list[tuple[int, int]] = []
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        node_id = 256 + len(nodes)
        nodes.append((n1, n2))
        heapq.heappush(heap, (w1 + w2, order, node_id))
        order += 1
    root = heap[0][2]
    stack: list[tuple[int, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node < 256:
            lengths[node] = max(1, depth)
            continue
        left, right = nodes[node - 256]
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))
    return lengths


def _huff_canonical_codes(lengths: list[int]) -> list[tuple[int, int]]:
    """内部ヘルパー: `huff_canonical_codes` を実行する。

    Args:
        lengths (list[int]): lengths の値。

    Returns:
        list[tuple[int, int]]: 計算結果。
    """
    syms = sorted((ln, sym) for sym, ln in enumerate(lengths) if ln > 0)
    codes: list[tuple[int, int]] = [(0, 0)] * 256
    code = 0
    prev_len = 0
    for ln, sym in syms:
        if ln > 63:
            raise RuntimeError("[ENC] huffman code length exceeded 63 bits")
        code <<= ln - prev_len
        codes[sym] = (code, ln)
        code += 1
        prev_len = ln
    return codes


def _huff122_compress(data: bytes) -> bytes:
    """内部ヘルパー: `huff122_compress` を実行する。

    Args:
        data (bytes): data の値。

    Returns:
        bytes: 計算結果。
    """
    lengths = _huff_code_lengths(data)
    codes = _huff_canonical_codes(lengths)
    out = bytearray(lengths)
    cur = 0
    nbits = 0
    for b in data:
        code, ln = codes[b]
        for k in range(ln - 1, -1, -1):
            cur = (cur << 1) | ((code >> k) & 1)
            nbits += 1
            if nbits >= 8:
                out.append(cur & 0xFF)
                cur = 0
                nbits = 0
    if nbits > 0:
        out.append((cur << (8 - nbits)) & 0xFF)
    return bytes(out)


def _huff122_decompress(blob: bytes, *, expected_size: int) -> bytes:
    """内部ヘルパー: `huff122_decompress` を実行する。

    Args:
        blob (bytes): blob の値。
        expected_size (int): expected_size の値。

    Returns:
        bytes: 計算結果。
    """
    if expected_size == 0:
        return b""
    if len(blob) < 256:
        raise RuntimeError("[ENC] huff122 payload too short")
    lengths = list(blob[:256])
    syms = sorted((ln, sym) for sym, ln in enumerate(lengths) if ln > 0)
    if not syms:
        raise RuntimeError("[ENC] huff122 has no symbols")

    left = [-1]
    right = [-1]
    value = [-1]
    code = 0
    prev_len = 0
    for ln, sym in syms:
        code <<= ln - prev_len
        node = 0
        for k in range(ln - 1, -1, -1):
            bit = (code >> k) & 1
            nxt = right[node] if bit else left[node]
            if nxt < 0:
                left.append(-1)
                right.append(-1)
                value.append(-1)
                nxt = len(value) - 1
                if bit:
                    right[node] = nxt
                else:
                    left[node] = nxt
            node = nxt
        value[node] = sym
        code += 1
        prev_len = ln

    out = bytearray()
    node = 0
    for by in blob[256:]:
        for k in range(7, -1, -1):
            bit = (by >> k) & 1
            node = right[node] if bit else left[node]
            if node < 0:
                raise RuntimeError("[ENC] huff122 broken bitstream")
            sym = value[node]
            if sym >= 0:
                out.append(sym)
                if len(out) == expected_size:
                    return bytes(out)
                node = 0
    raise RuntimeError("[ENC] huff122 unexpected eof")


def encode_model_payload(blob: bytes, *, encoding: str) -> tuple[str, int]:
    """`encode_model_payload` を実行する。

    Args:
        blob (bytes): blob の値。
        encoding (str): encoding の値。

    Returns:
        tuple[str, int]: 計算結果。
    """
    enc = str(encoding).strip().lower()
    if enc == "base91":
        payload = _encode_base91(blob)
        if _decode_base91(payload) != blob:
            raise RuntimeError("[ENC] base91 roundtrip failed")
        return payload, PAYLOAD_CODEC_BASE91
    if enc == "base122":
        payload = _encode_base122(blob)
        if _decode_base122(payload) != blob:
            raise RuntimeError("[ENC] base122 roundtrip failed")
        return payload, PAYLOAD_CODEC_BASE122
    if enc == "huff122":
        packed = _huff122_compress(blob)
        payload = _encode_base122(packed)
        if _huff122_decompress(_decode_base122(payload), expected_size=len(blob)) != blob:
            raise RuntimeError("[ENC] huff122 roundtrip failed")
        return payload, PAYLOAD_CODEC_HUFF122
    if enc == "huff91":
        packed = _huff122_compress(blob)
        payload = _encode_base91(packed)
        if _huff122_decompress(_decode_base91(payload), expected_size=len(blob)) != blob:
            raise RuntimeError("[ENC] huff91 roundtrip failed")
        return payload, PAYLOAD_CODEC_HUFF91
    raise ValueError(f"unknown payload encoding: {encoding!r}")
