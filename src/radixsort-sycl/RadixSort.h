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
#ifndef _RADIXSORT_H_
#define _RADIXSORT_H_

#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include "Scan.h"

static const uint32_t WARP_SIZE = 32;
static const uint32_t bitStep = 4;
static const uint32_t CTA_SIZE = 128;

void radixSortKeys(sycl::queue &q,
                   uint32_t *d_keys,
                   uint32_t *d_tempKeys,
                   uint32_t *d_counters,
                   uint32_t *d_blockOffsets,
                   uint32_t *d_countersSum,
                   uint32_t *d_buffer,
                   const uint32_t numElements,
                   const uint32_t keyBits,
                   const uint32_t batchSize);

#endif
