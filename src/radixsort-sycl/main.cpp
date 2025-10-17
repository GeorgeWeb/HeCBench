/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <chrono>
#include "RadixSort.h"
#include "Scan.h"

void makeRandomUintVector(uint32_t *a, uint32_t numElements, uint32_t keybits);
bool verifySortUint(uint32_t *keysSorted,
    uint32_t *valuesSorted,
    uint32_t *keysUnsorted,
    uint32_t len);

int main(int argc, const char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int numIterations = atoi(argv[1]);

  const uint32_t numElements = 128*128*128*2; //1048576;
  const int keybits = 32; // bit size of uint32_t
  const int batchSize = 1; // only support a batch size of 1

  // Check power-of-two factorization before the scan operations start
  uint32_t arrayLength = numElements/2/CTA_SIZE*16;
  uint32_t log2L;
  uint32_t factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  //Check supported size range
  assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE));
  assert(arrayLength > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE);

  //Check total batch size limit
  assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

  const size_t elem_size = numElements * sizeof(uint32_t);

  // Alloc and init some data on the host, then alloc and init GPU buffer
  uint32_t* h_keys       = (uint32_t*)malloc(elem_size);
  uint32_t* h_keysSorted = (uint32_t*)malloc(elem_size);
  makeRandomUintVector(h_keys, numElements, keybits);

  uint32_t numBlocks = ((numElements % (CTA_SIZE * 4)) == 0) ?
    (numElements / (CTA_SIZE * 4)) : (numElements / (CTA_SIZE * 4) + 1);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uint32_t *d_keys, *d_tempKeys, *d_counters,
               *d_countersSum, *d_blockOffsets, *d_buffer;

  d_keys = sycl::malloc_device<uint32_t>(numElements, q);
  q.memcpy(d_keys, h_keys, elem_size);
  d_tempKeys = sycl::malloc_device<uint32_t>(numElements, q);
  d_counters = sycl::malloc_device<uint32_t>(WARP_SIZE*numBlocks, q);
  d_countersSum = sycl::malloc_device<uint32_t>(WARP_SIZE*numBlocks, q);
  d_blockOffsets = sycl::malloc_device<uint32_t>(WARP_SIZE*numBlocks, q);
  d_buffer = sycl::malloc_device<uint32_t>(arrayLength/MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE, q);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; i++)
  {
    radixSortKeys(q, d_keys, d_tempKeys, d_counters, d_blockOffsets, d_countersSum,
                  d_buffer, numElements, keybits, batchSize);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of radixsort: %f (s)\n", (time * 1e-9f) / numIterations);
  fflush(stdout);

  // copy sorted keys to CPU
  q.memcpy(h_keysSorted, d_keys, elem_size).wait();

  // Check results
  bool passed = true;
  passed &= verifySortUint(h_keysSorted, NULL, h_keys, numElements);

  free(h_keys);
  free(h_keysSorted);
  sycl::free(d_keys, q);
  sycl::free(d_tempKeys, q);
  sycl::free(d_counters, q);
  sycl::free(d_countersSum, q);
  sycl::free(d_blockOffsets, q);
  sycl::free(d_buffer, q);

  // finish
  if (passed)
    printf("PASS\n");
  else
    printf("FAIL\n");

  return 0;
}

void makeRandomUintVector(uint32_t *a, uint32_t numElements, uint32_t keybits)
{
  // Fill up with some random data
  int keyshiftmask = 0;
  if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
  int keymask = 0xffff;
  if (keybits < 16) keymask = (1 << keybits) - 1;

  srand(95123);
  for(uint32_t i=0; i < numElements; ++i)
  {
    a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask);
  }
}

// assumes the values were initially indices into the array, for simplicity of
// checking correct order of values
bool verifySortUint(uint32_t *keysSorted,
    uint32_t *valuesSorted,
    uint32_t *keysUnsorted,
    uint32_t len)
{
  bool passed = true;
  for(uint32_t i=0; i<len-1; ++i)
  {
    if( (keysSorted[i])>(keysSorted[i+1]) )
    {
      printf("Unordered key[%d]: %d > key[%d]: %d\n", i, keysSorted[i], i+1, keysSorted[i+1]);
      passed = false;
      break;
    }
  }

  if (valuesSorted)
  {
    for(uint32_t i=0; i<len; ++i)
    {
      if( keysUnsorted[valuesSorted[i]] != keysSorted[i] )
      {
        printf("Incorrectly sorted value[%u] (%u): %u != %u\n",
            i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
        passed = false;
        break;
      }
    }
  }

  return passed;
}
