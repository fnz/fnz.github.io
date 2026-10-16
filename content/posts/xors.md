---
title: "XORs, swizzling, and reductions"
date: 2026-10-11
slug: "xors"
draft: false
---

One way to think about XOR is as a bit flip: XORing a bit with \(0\) is a no-op, while XORing it with \(1\) flips it. The same idea extends to bitwise XOR: to compute \(x \oplus y\), flip each bit of \(x\) wherever the corresponding bit of \(y\) is \(1\).

The rule is straightforward, but I never had a good intuition for what bitwise XOR does to a number. That changed when I visualized what XOR does to a sequence.

## XORs and sequences

Consider the sequence from \(0\) to \(15\). Apply each mask independently to the original sequence, using masks with exactly one set bit:

{{< illustration
  src="/images/xors/xor_sequence.png?v=962695d7"
  alt="The sequence zero through fifteen, followed by the permutations produced by XOR masks one, two, four, and eight"
  width="2400"
  height="882"
  caption="Masks with one set bit swap adjacent blocks of one, two, four, or eight elements."
  wide="true"
>}}

XOR with \(1_2\) flips the lowest bit, so each even number trades places with the odd number next to it. XOR with \(10_2\) flips the next bit, swapping adjacent blocks of two. The masks \(100_2\) and \(1000_2\) swap blocks of four and eight. Applying the same mask twice restores the original order: \((x \oplus m) \oplus m = x\).

What happens when a mask has more than one set bit? The result is the same as applying the corresponding single-bit masks in succession. The order does not matter: XOR is associative and commutative.

For the sequence \(0,1,\ldots,2^n-1\), each \(n\)-bit mask produces a distinct permutation, built from swaps of adjacent blocks whose sizes are powers of two. There are \(2^n\) possible masks, giving \(2^n\) distinct XOR permutations.

A mask with all \(n\) bits set reverses the sequence by successively swapping adjacent blocks of increasing size. We can reach the same conclusion algebraically: the original number and its bitwise complement add up to \(2^n-1\), so \(x \oplus (2^n-1) = 2^n-1-x\).

We can apply these permutations to indices, too. XOR can then select which worker to exchange values with, or where to store an element in memory.

## Reductions

Suppose four workers, numbered \(0\) through \(3\), hold values \(a,b,c,d\). We want to compute their sum and leave the result in every worker. This is an all-reduce.

At each stage, worker \(i\) combines its current value with the previous-stage value from worker \(i\oplus m\), where \(m\) is the XOR mask. The illustration shows two stages, using masks \(1_2\) and \(10_2\). Multiple letters inside a tile represent the sum of those values:

{{< illustration
  src="/images/xors/butterfly_reduction.png?v=38d265a7"
  alt="Four workers start with a, b, c, d; XOR-selected exchanges and additions give every worker the sum a+b+c+d"
  width="2400"
  height="1000"
  wide="true"
>}}

The first stage pairs workers \(0\) and \(1\), and workers \(2\) and \(3\). The first pair now holds \(a+b\); the second holds \(c+d\). The second stage pairs workers \(0\) and \(2\), and workers \(1\) and \(3\), leaving all four with \(a+b+c+d\).

At each stage, the XOR mask pairs workers whose partial sums contain disjoint sets of inputs. Combining those sums doubles the number of original values included in each worker's sum without counting anything twice. With this butterfly pattern, a power-of-two group of \(N\) workers needs \(\log_2 N\) stages, using masks \(1,2,4,\ldots,N/2\).

## Swizzling

Swizzling changes the mapping from logical array indices to physical memory locations. An XOR swizzle can avoid bank conflicts in GPU shared memory. We’ll use NVIDIA’s CUDA programming model as a concrete example.

For the scalar 32-bit accesses considered here, shared memory has 32 banks, each serving 4-byte words. A word at index \(q\) maps to bank \(q \bmod 32\). Threads in a warp can access different banks in parallel. When multiple threads access different words in the same bank, however, those accesses must be serialized, reducing throughput. This is called a bank conflict.

Suppose we have a \(32\times32\) matrix of 32-bit floating-point values. The matrix occupies 4 KiB of shared memory. We want the 32 threads in a warp to read either a whole row or a whole column, with each thread reading one element.

Assume we store the matrix in row-major order, using zero-based indices. The element at row \(r\), column \(c\) has word offset \(32r+c\). Taking this offset modulo \(32\) keeps only its lowest five bits, which come entirely from \(c\).

When reading a row, \(r\) is fixed and \(c\) runs from \(0\) to \(31\), so every thread accesses a different bank. When reading a column, \(c\) is fixed and \(r\) varies, but \((32r+c)\bmod 32=c\). Every thread therefore accesses a different word in the same bank.

XOR lets us mix the row index into those bits: store each element in column \(c\oplus r\) instead. For a fixed logical column \(c\), the bank indices \(c\oplus r\) cover all 32 banks exactly once as \(r\) runs from \(0\) to \(31\). Row accesses remain conflict-free because XOR with a fixed row index simply permutes the columns.

The following \(4\times4\) example illustrates the idea. Each entry labels an element by its original column index. The bank-conflict argument above applies to the \(32\times32\) layout:

{{< illustration
  src="/images/xors/xor_swizzle.png?v=38b7578c"
  alt="A four-by-four matrix with each row permuted by XORing the column index with the row index"
  width="2400"
  height="765"
  wide="true"
>}}

The swizzled word offset for logical element \((r,c)\) is \(q_{\mathrm{swizzled}}(r,c)=32r+(c\oplus r)\). Both reads and writes use this mapping; the stored values themselves are unchanged.  Practical GPU layouts choose which address bits to XOR according to the element size and access pattern.
