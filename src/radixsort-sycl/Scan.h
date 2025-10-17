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
#ifndef _SCAN_H_
#define _SCAN_H_

#include <sycl/sycl.hpp>

using uint4 = sycl::uint4;
using uint2 = sycl::uint2;

#define MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 1024
#define MAX_LOCAL_GROUP_SIZE 256
static const int WORKGROUP_SIZE = 256;
static const uint32_t   MAX_BATCH_ELEMENTS = 64 * 1048576;
static const uint32_t MIN_SHORT_ARRAY_SIZE = 4;
static const uint32_t MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
static const uint32_t MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
static const uint32_t MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

uint32_t factorRadix2(uint32_t& log2L, uint32_t L);

void scanExclusiveLarge(
    sycl::queue &q,
    uint32_t *d_Dst,
    uint32_t *d_Src,
    uint32_t *d_Buf,
    const uint32_t batchSize,
    const uint32_t arrayLength,
    const uint32_t numElements);
#endif
