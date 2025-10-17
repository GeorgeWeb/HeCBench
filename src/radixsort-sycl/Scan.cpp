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

#include "Scan.h"
#include "Scan_kernels.cpp"


void scanExclusiveLocal1(
    sycl::queue &q,
    uint32_t *d_Dst,
    uint32_t *d_Src,
    const uint32_t n,
    const uint32_t size)
{
  size_t localWorkSize = WORKGROUP_SIZE;
  size_t globalWorkSize = (n * size) / 4;

  sycl::range<1> gws (globalWorkSize);
  sycl::range<1> lws (localWorkSize);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 1> l_Data(sycl::range<1>(2 * WORKGROUP_SIZE), cgh);
    cgh.parallel_for<class scan_exclusive_local1>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      scanExclusiveLocal1K(item, d_Dst, d_Src, l_Data.get_pointer(), size);
    });
  });
}

void scanExclusiveLocal2(
    sycl::queue &q,
    uint32_t *d_Buf,
    uint32_t *d_Dst,
    uint32_t *d_Src,
    const uint32_t n,
    const uint32_t size)
{
  const uint32_t elements = n * size;
  size_t localWorkSize = WORKGROUP_SIZE;
  size_t globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);
  sycl::range<1> gws (globalWorkSize);
  sycl::range<1> lws (localWorkSize);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 1> l_Data(sycl::range<1>(2 * WORKGROUP_SIZE), cgh);
    cgh.parallel_for<class scan_exclusive_local2>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      scanExclusiveLocal2K(item, d_Buf, d_Dst, d_Src,
                           l_Data.get_pointer(), elements, size);
    });
  });
}

void uniformUpdate(
    sycl::queue &q,
    uint32_t *d_Dst,
    uint32_t *d_Buf,
    const uint32_t n)
{
  sycl::range<1> gws (n * WORKGROUP_SIZE);
  sycl::range<1> lws (WORKGROUP_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 0> buf(cgh);
    cgh.parallel_for<class uniform_update>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      uniformUpdateK(item, d_Dst, d_Buf, buf);
    });
  });
}

// main exclusive scan routine
void scanExclusiveLarge(
    sycl::queue &q,
    uint32_t *d_Dst,
    uint32_t *d_Src,
    uint32_t *d_Buf,
    const uint32_t batchSize,
    const uint32_t arrayLength,
    const uint32_t numElements)
{

  scanExclusiveLocal1(
      q,
      d_Dst,
      d_Src,
      (batchSize * arrayLength) / (4 * WORKGROUP_SIZE),
      4 * WORKGROUP_SIZE);

  scanExclusiveLocal2(
      q,
      d_Buf,
      d_Dst,
      d_Src,
      batchSize,
      arrayLength / (4 * WORKGROUP_SIZE));

  uniformUpdate(
      q,
      d_Dst,
      d_Buf,
      (batchSize * arrayLength) / (4 * WORKGROUP_SIZE));
}
