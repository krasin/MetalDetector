//
//  Layers.metal
//  MetalDetector
//
//  Created by Ivan Krasin on 9/30/15.
//  Copyright © 2015 Ivan Krasin. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void cropAndRotate(texture2d<float, access::sample> in [[texture(0)]],
                          texture2d<float, access::write> out [[texture(1)]],
                          sampler sample [[sampler(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    // TODO: pass width / height ratio as a parameter.
    float k = 0.75;
    float2 coord = {
        0.5 + k * (float(gid.x) - 128) / 256.0,
        float(gid.y) / 256.0,
    };
    const float4 color = in.sample(sample, coord);
    out.write(color, gid);
}

// Takes 352x288 32BGRA image, crops 224x224 from the center, extracts color channels,
// and puts float values into the output 3d texture.
kernel void crop352x288to224(texture2d<float, access::read> in [[texture(0)]],
                             texture2d_array<float, access::write> out [[texture(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint2 coord = { gid.x + 64, gid.y + 32 };
    const float4 bgra = in.read(coord);
    out.write(bgra.b, gid, 0);
    out.write(bgra.g, gid, 1);
    out.write(bgra.r, gid, 2);
    // TODO: make sure the color is in the valid range.
}

// Takes 32BGRA image, crops 224x224 from the center, extracts color channels,
// subtracts ImageNet mean, and puts float values into the output 3d texture.
// It also flips the image vertically.
kernel void preprocess(texture2d<float, access::read> in [[texture(0)]],
                       texture2d_array<float, access::write> out [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint dx = (in.get_width() - 224) / 2;
    uint dy = (in.get_height() - 224) / 2;
    uint2 coord = { gid.x + dx, gid.y + dy };

    // Flipping vertically. This is need in case of loading an image from resources.
    // Not so sure about the camera images.
    coord.y = in.get_height()-coord.y-1;

    const float4 bgra = in.read(coord);
    const float b = bgra[0];
    const float g = bgra[1];
    const float r = bgra[2];
    out.write(b * 255 - 104.007, gid, 0);
    out.write(g * 255 - 116.66947, gid, 1);
    out.write(r * 255 - 122.6751, gid, 2);
}


kernel void float2BGRA(texture2d_array<float, access::read> in [[texture(0)]],
                       texture2d<float, access::write> out [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const float b = in.read(gid, 0).r;
    const float g = in.read(gid, 1).r;
    const float r = in.read(gid, 2).r;
    out.write(float4((r+122.679)/255, (g+116.669)/255, (b+104.001)/255, 255), gid);
}

kernel void computeL1(texture2d_array<float, access::read> in [[texture(0)]],
                      device float* res [[buffer(0)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= in.get_height()) { return; }
    float sum = 0.0;
    for (uint i = 0; i < in.get_array_size(); i++) {
        for (uint j = 0; j < in.get_width(); j++) {
            sum += abs(in.read(uint2(j, gid), i).r);
        }
    }
    res[gid] = sum;
}

kernel void computeL2(texture2d_array<float, access::read> in [[texture(0)]],
                      device float* res [[buffer(0)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= in.get_height()) { return; }
    float sum = 0.0;
    for (uint i = 0; i < in.get_array_size(); i++) {
        for (uint j = 0; j < in.get_width(); j++) {
            float r = in.read(uint2(j, gid), i).r;
            sum += r*r;
        }
    }
    res[gid] = sum;
}

kernel void computeMax(texture2d_array<float, access::read> in [[texture(0)]],
                       device float* res [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= in.get_height()) { return; }
    float val = in.read(uint2(0, gid), 0).r;
    for (uint i = 0; i < in.get_array_size(); i++) {
        for (uint j = 0; j < in.get_width(); j++) {
            float r = in.read(uint2(j, gid), i).r;
            if (r > val) {
                val = r;
            }
        }
    }
    res[gid] = val;
}

// Takes a sample 8x8 from the first slice.
kernel void sample8x8(texture2d_array<float, access::read> in [[texture(0)]],
                      device float* res [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    res[gid.y*8+gid.x] = in.read(gid, 0).r;
}

kernel void loss3_classifier_0(texture2d_array<float, access::read> in [[texture(0)]],
                               texture2d_array<float, access::write> out [[texture(1)]],
                               device float* weights [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= 1000) { return; }
    float sum = 0.0;
    // Skip weights for the previous filters
    uint i = gid.x * in.get_array_size() * in.get_height() * in.get_height();
    for (uint c = 0; c < in.get_array_size(); c++) {
        for (uint y = 0; y < in.get_height(); y++) {
            for (uint x = 0; x < in.get_width(); x++) {
                float v = in.read(uint2(x, y), c)[0];
                sum += weights[i] * v;
                i++;
            }
        }
    }
    out.write(sum, uint2(0,0), gid.x);
}

kernel void prob_0(texture2d_array<float, access::read> in [[texture(0)]],
                   texture2d_array<float, access::write> out [[texture(1)]],
                   device float* weights [[buffer(0)]],
                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= 1) { return; }
    float maxv = in.read(uint2(0,0), 0)[0];
    for (uint x = 0; x < 1000; x++) {
        float v = in.read(uint2(0, 0), x)[0];
        maxv = max(maxv, v);
    }

    float sum = 0.0;
    for (uint x = 0; x < 1000; x++) {
        float v = in.read(uint2(0, 0), x)[0];
        sum += exp(v - maxv);
    }

    for (uint x = 0; x < 1000; x++) {
        float v = in.read(uint2(0, 0), x)[0];
        float tmp = exp(v - maxv);
        float res = tmp / sum;
        out.write(res, uint2(0, 0), x);
    }
}

// Converts a texture 1x1xarray_length into a buffer.
kernel void array1x1_to_buffer_0(texture2d_array<float, access::read> in [[texture(0)]],
                                 device float* out [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= in.get_array_size()) {
        return;
    }
    float v = in.read(uint2(0, 0), gid.x)[0];
    out[gid.x] = v;
}
