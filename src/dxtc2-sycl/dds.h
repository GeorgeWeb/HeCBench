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

#ifndef DDS_H
#define DDS_H

#if !defined(MAKEFOURCC)
#    define MAKEFOURCC(ch0, ch1, ch2, ch3) \
        ( (unsigned int)(ch0)        | ((unsigned int)(ch1) << 8) | \
        ( (unsigned int)(ch2) << 16) | ((unsigned int)(ch3) << 24) )
#endif

typedef unsigned int uint32_t;
typedef unsigned short ushort;

struct DDSPixelFormat {
    uint32_t size;
    uint32_t flags;
    uint32_t fourcc;
    uint32_t bitcount;
    uint32_t rmask;
    uint32_t gmask;
    uint32_t bmask;
    uint32_t amask;
};

struct DDSCaps {
    uint32_t caps1;
    uint32_t caps2;
    uint32_t caps3;
    uint32_t caps4;
};

/// DDS file header.
struct DDSHeader {
    uint32_t fourcc;
    uint32_t size;
    uint32_t flags;
    uint32_t height;
    uint32_t width;
    uint32_t pitch;
    uint32_t depth;
    uint32_t mipmapcount;
    uint32_t reserved[11];
    DDSPixelFormat pf;
    DDSCaps caps;
    uint32_t notused;
};

static const uint32_t FOURCC_DDS = MAKEFOURCC('D', 'D', 'S', ' ');
static const uint32_t FOURCC_DXT1 = MAKEFOURCC('D', 'X', 'T', '1');
static const uint32_t DDSD_WIDTH = 0x00000004U;
static const uint32_t DDSD_HEIGHT = 0x00000002U;
static const uint32_t DDSD_CAPS = 0x00000001U;
static const uint32_t DDSD_PIXELFORMAT = 0x00001000U;
static const uint32_t DDSCAPS_TEXTURE = 0x00001000U;
static const uint32_t DDPF_FOURCC = 0x00000004U;
static const uint32_t DDSD_LINEARSIZE = 0x00080000U;


#endif // DDS_H
