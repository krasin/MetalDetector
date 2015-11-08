#include <metal_stdlib>
using namespace metal;

static void max_pool(texture2d_array<half, access::read> in,
                     texture2d_array<half, access::write> out,
                     uint2 gid,
                     int kernelH, int kernelW, int strideH, int strideW,
                     int padH, int padW) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;

    for (uint f = 0; f < in.get_array_size(); f++) {
        half v = in.read(uint2(x, y), f)[0];
        for (int yy = y-kernelH/2; yy <= y+kernelH/2; yy++) {
            if (yy < 0 || yy >= int(in.get_height())) {
                continue;
            }
            for (int xx = x-kernelW/2; xx <= x+kernelW/2; xx++) {
                if (xx < 0 || xx >= int(in.get_width())) {
                    continue;
                }
                half cur = in.read(uint2(xx, yy), f)[0];
                if (cur > v) {
                    v = cur;
                }
            }
        }
        out.write(v, gid, f);
    }
}

static void ave_pool(texture2d_array<half, access::read> in,
                     texture2d_array<half, access::write> out,
                     uint2 gid,
                     int kernelH, int kernelW, int strideH, int strideW,
                     int padH, int padW) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;

    for (uint f = 0; f < in.get_array_size(); f++) {
        float sum = 0.0;
        for (int yy = y-kernelH/2; yy <= y+kernelH/2; yy++) {
            if (yy < 0 || yy >= int(in.get_height())) {
                continue;
            }
            for (int xx = x-kernelW/2; xx <= x+kernelW/2; xx++) {
                if (xx < 0 || xx >= int(in.get_width())) {
                    continue;
                }
                sum += in.read(uint2(xx, yy), f)[0];
            }
        }
        out.write(sum/(kernelH*kernelW), gid, f);
    }
}

static void cross_channel_lrn(texture2d_array<half, access::read> in,
                              texture2d_array<half, access::write> out,
                              uint2 gid,
                              int local_size, float alpha, float beta) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }

    float sum = 0.0;

    // Compute the first half
    for (int c = 0; c < local_size / 2; c++) {
        float v = in.read(gid, c)[0];
        sum += v*v;
    }
    for (int c = 0; c < int(in.get_array_size()); c++) {
        float tail = 0.0;
        if (c > local_size / 2) {
            tail = in.read(gid, c - local_size/2 - 1)[0];
        }
        float head = 0.0;
        if (c < int(in.get_array_size()) - local_size/2) {
            head = in.read(gid, c + local_size/2)[0];
        }
        sum += head*head - tail*tail;
        float p = 1 + alpha/local_size*sum;
        float q = pow(p, beta);
        out.write(in.read(gid, c)[0] / q, gid, c);
    }
}

constant half conv1_7x7_s2_bias[] = {
    -2.527231, -0.738781, -1.447727, 0.723762, 1.790998, 0.088202, -1.789945, -2.435894,
    2.625920, -1.464736, -0.989690, -2.396273, -0.979302, -2.431748, -1.652557, 0.226866,
    4.435109, -0.253463, 4.067864, -1.533034, 5.112535, -2.037816, -1.394271, -3.665097,
    -0.435853, -2.881038, -3.139135, -4.150452, -3.222371, -3.112326, 0.185826, 2.008326,
    4.501478, 3.169470, 4.940134, -0.298538, -0.541925, 2.691462, -1.105637, -2.295830,
    -0.615527, -0.474120, 2.186058, -1.491552, -1.611037, -4.244909, -2.643700, -3.266358,
    -5.803696, -6.166785, -4.219829, -3.748438, -4.459106, -3.622702, -2.058036, 1.455683,
    3.402357, -2.447811, -2.605467, 0.235679, -2.594265, -5.335870, -5.118693, -7.981039,
};

kernel void conv1_7x7_s2_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 7;
    const int kernelW = 7;
    const int strideH = 2;
    const int strideW = 2;
    const int padH = 3;
    const int padW = 3;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = conv1_7x7_s2_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 3; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 224 && x+dx >= 0 && x+dx < 224) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void pool1_3x3_s2_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 2;
    const int strideW = 2;
    const int padH = 0;
    const int padW = 0;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

kernel void pool1_norm1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int local_size = 5;
    const float alpha = 0.000100;
    const float beta = 0.750000;
    cross_channel_lrn(in, out, gid, local_size, alpha, beta);
}

constant half conv2_3x3_reduce_bias[] = {
    0.381635, 0.039805, 0.733464, 0.151748, 1.596660, 0.623407, -0.158631, -0.390446,
    -1.091076, 0.285065, -1.447569, 0.861561, 1.942077, 0.643233, 0.367726, 0.201104,
    -0.224591, 0.110250, 0.940798, -0.017305, -0.120072, 0.075464, 0.526009, 0.499644,
    -2.174395, -0.119149, 0.039089, -1.408327, 0.727392, 2.257186, 0.562781, 0.259066,
    0.095934, 0.737730, 0.423583, -0.154759, 0.013040, 1.326806, -0.782034, 0.868504,
    1.070002, 0.933033, 0.653132, -0.163463, 0.807962, 0.057831, 0.169202, -0.119385,
    -0.069811, -0.494880, -0.380963, -0.710609, 0.069571, -0.003370, -0.184156, -0.017192,
    1.247302, -0.093687, 1.122774, 0.817257, -0.532370, 0.661138, 0.271513, -0.147721,
};

kernel void conv2_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = conv2_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 64; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 56 && x+dx >= 0 && x+dx < 56) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half conv2_3x3_bias[] = {
    -0.018045, 0.859649, -0.243074, -0.078558, 0.066982, 0.013044, -0.444394, 0.026513,
    -0.023854, -0.098568, -0.208399, 0.202242, -0.236095, 0.083836, -0.158423, -0.130466,
    -0.141710, 0.102807, -0.044140, 0.397561, -0.100035, -0.022219, -0.086748, 0.066620,
    -0.504363, -0.032682, 0.049474, -0.015798, -0.086592, 0.152469, -0.045321, 0.242998,
    0.277625, -0.026484, 0.205391, 0.413144, -0.304399, 0.380759, 0.366718, 0.512499,
    0.172767, 0.103286, -0.031361, -0.040604, 0.050591, -0.058977, -0.166147, 0.357889,
    0.092035, -0.436973, -0.248229, 0.032827, 0.347654, -0.177136, -0.047772, 0.682986,
    -0.016202, 0.018334, -0.428193, -0.293579, 0.596653, -0.103259, 0.282682, 0.525740,
    -0.067584, -0.394834, 0.122873, -0.095895, 0.098696, 0.343101, -0.237528, 0.038658,
    -0.085134, 0.433906, -0.134415, 0.470490, 0.128177, -0.274744, 0.297731, -0.144459,
    -0.366618, -0.128433, 0.100089, 0.660534, -0.083600, -0.066656, -0.105381, 0.187281,
    -0.038112, 0.254171, 0.043598, 0.300458, 0.124110, 0.242896, 0.133399, 0.039427,
    -0.018298, -0.242644, 0.828521, -0.084042, 1.625556, -0.052083, 0.062697, 0.070897,
    -0.379805, -0.062163, -0.032850, 0.354569, 0.159876, -0.165125, -0.056362, 0.281309,
    -0.069830, 0.633957, -0.211270, 0.011622, 0.068714, -0.099356, 0.325811, 0.387915,
    0.207147, -0.097830, 0.164864, -0.183474, 0.068332, -0.040898, -0.260564, -0.048733,
    -0.217698, -0.172005, 0.095328, -0.011094, 0.605625, -0.089793, 0.346167, 0.062285,
    0.020376, 0.170477, -0.161986, -0.034228, 0.503234, -0.149493, 0.340676, 0.083997,
    -0.050570, -0.653233, -0.044648, -0.011313, 0.273607, -0.258398, 0.810286, -0.255558,
    0.234701, 0.323857, 0.077906, -0.038161, -0.173450, 0.022598, 0.153025, 0.087075,
    -0.172730, 0.151664, -0.031301, 0.043224, 0.364073, -0.083543, -0.090865, -0.052129,
    0.190257, -0.552814, -0.004167, -0.299732, 0.038459, -0.019732, -0.082415, 0.001550,
    0.040179, -0.167832, 0.246141, 0.016906, 0.041944, 0.039024, 0.116925, -0.390617,
    0.092427, -0.061722, -0.040117, -0.067746, 0.844944, -0.097982, 0.354183, 0.897184,
};

kernel void conv2_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[192];
    for (int f = 0; f < 192; f++) {
        sum[f] = conv2_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 64; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 56 && x+dx >= 0 && x+dx < 56) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 192; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 192;
            }
        }
    }

    for (int f = 0; f < 192; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void conv2_norm2_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int local_size = 5;
    const float alpha = 0.000100;
    const float beta = 0.750000;
    cross_channel_lrn(in, out, gid, local_size, alpha, beta);
}

kernel void pool2_3x3_s2_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 2;
    const int strideW = 2;
    const int padH = 0;
    const int padW = 0;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_3a_1x1_bias[] = {
    -0.026487, -0.292247, -0.165532, -0.350475, 0.676674, 0.345550, 0.779559, -0.375620,
    0.506046, 0.194911, 0.533021, -0.423778, -0.227247, 0.135422, 0.690028, 0.349100,
    -0.179175, 0.322045, 1.032794, 0.270825, -0.120131, -0.317952, -0.091555, -0.196342,
    0.754887, 1.272587, -0.018109, -0.327786, 0.012645, 0.094291, 0.711310, -0.339721,
    -1.017326, -0.556699, -0.064581, -0.045442, -0.160866, -0.259176, -0.131964, 0.176338,
    -0.114521, -0.650283, -0.660740, -0.803953, 0.861309, 0.179753, 0.595724, -0.435663,
    0.569224, 0.232103, 0.543166, 0.383420, -0.095557, -0.581560, 0.830209, 0.608088,
    0.099112, 0.002090, 0.417921, 0.147452, 0.276963, 0.399921, 0.228397, -0.245762,
};

kernel void inception_3a_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_3a_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 192; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3a_3x3_reduce_bias[] = {
    0.444450, 0.742302, 0.742348, 0.928892, -0.133459, -0.300912, -0.274805, -0.221462,
    -0.523578, 0.400588, -0.331309, -0.217450, -0.153935, 1.031904, 0.191419, 0.114952,
    -0.252943, -0.434137, 1.142241, 0.323789, 0.046747, 0.588310, -0.032665, -0.300166,
    -0.132584, -0.101325, -0.411793, 0.122021, -0.037180, 0.778134, -0.093180, -0.228789,
    -0.134148, 0.372300, -0.544382, 0.147568, -0.467647, 0.924606, -0.410599, -0.077996,
    -0.493179, -0.261328, -0.050229, 0.187024, -0.145724, -0.123196, 0.210992, 0.075320,
    -0.506418, 0.275727, 1.388808, 0.287038, -0.185410, 0.257893, -0.356466, 0.535687,
    0.010977, 0.581898, -0.019528, -0.072174, 1.414213, -0.523291, 0.044709, 0.385924,
    -0.148533, 0.100180, -0.211680, 0.594924, 0.064475, -0.006752, -0.394408, -0.291680,
    -0.071066, 1.173087, -0.263585, -0.219881, 0.337821, -0.144830, -0.303959, -0.077917,
    -0.237823, -0.049124, 0.619378, -0.391171, 0.872612, 0.087517, -0.103412, -0.082671,
    1.512146, -0.217145, 0.331512, -0.200594, -0.344177, -0.154807, -0.034828, -0.122621,
};

kernel void inception_3a_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[96];
    for (int f = 0; f < 96; f++) {
        sum[f] = inception_3a_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 192; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 96; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 96;
            }
        }
    }

    for (int f = 0; f < 96; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3a_3x3_bias[] = {
    -0.130083, -0.181063, 0.354590, 0.483169, -0.008969, 0.049078, 0.226737, 0.018692,
    -0.064437, 0.061680, 0.437045, -0.117031, 0.541062, 0.488556, 0.034759, 0.091008,
    0.195861, 0.219708, 0.316996, 0.111495, -0.046773, 0.063844, 0.330574, 0.193963,
    -0.101596, 0.166687, 0.017817, 0.117393, 0.030503, 0.239627, -0.008497, 0.400187,
    0.202275, 0.083260, 0.467872, 0.099779, 0.120530, 0.770865, 0.099319, 0.239269,
    -0.304378, -0.071569, -0.032486, -0.010558, 0.351959, 0.026115, 0.124910, 0.523985,
    0.536310, 0.058426, 0.134180, 0.463738, -0.025643, 0.162935, 0.019075, -0.040809,
    0.610603, -0.062858, 0.491919, 0.263671, 0.020908, -0.186498, 0.119408, 0.002056,
    0.005989, 0.152866, -0.030024, -0.075548, 0.461988, 0.135461, 0.030356, 0.085785,
    0.444532, 0.165792, 0.036233, 0.030018, 0.154906, -0.064101, 0.812634, -0.241834,
    0.241522, 0.307966, 0.236250, -0.225907, 0.051468, -0.033101, 0.232262, -0.314043,
    0.276810, 0.065183, -0.043691, 0.192829, -0.158639, -0.109177, 0.239890, 0.288664,
    0.107085, 0.205482, 0.425432, -0.034356, 0.222365, 0.569657, 0.125989, -0.030176,
    -0.079203, -0.116299, 0.751974, 0.171964, 0.104998, 0.083926, 0.001650, 0.251051,
    -0.464477, -0.116633, 0.035062, 0.430229, 0.078416, 0.242093, 0.152806, 0.475617,
    0.466768, 0.048544, 0.037765, 0.063393, -0.073878, -0.064708, -0.132981, -0.058425,
};

kernel void inception_3a_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_3a_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 96; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3a_5x5_reduce_bias[] = {
    0.328687, 0.159931, 0.715946, 0.061775, 0.845152, 0.449255, 0.407990, -0.142434,
    -0.005802, 1.212135, 0.429293, 0.189353, 0.143700, 0.608470, -0.139362, 0.213858,
};

kernel void inception_3a_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[16];
    for (int f = 0; f < 16; f++) {
        sum[f] = inception_3a_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 192; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 16; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 16;
            }
        }
    }

    for (int f = 0; f < 16; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3a_5x5_bias[] = {
    -0.099902, 0.132938, 0.136594, 0.134427, -0.287408, 0.537675, 0.057207, 0.222657,
    0.140582, -0.194733, -0.043158, 0.051582, -0.017562, -0.132438, 0.168174, -0.061807,
    0.200534, -0.141866, 0.149240, 0.029323, 0.547436, -0.029844, -0.053806, 0.350738,
    0.056696, 0.240434, -0.097851, 0.449390, 0.324670, -0.057889, 0.040809, 0.015824,
};

kernel void inception_3a_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[32];
    for (int f = 0; f < 32; f++) {
        sum[f] = inception_3a_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 16; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 32; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 32;
            }
        }
    }

    for (int f = 0; f < 32; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_3a_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_3a_pool_proj_bias[] = {
    0.489764, 0.443646, 0.747081, 0.742486, 0.294234, 0.462667, 0.452358, 0.754135,
    0.386173, 0.334248, 0.418359, 0.221777, 0.400554, 0.484855, 0.233882, 0.570823,
    0.456639, 0.313370, 0.368941, 0.508402, 0.472884, 0.440949, 0.497927, -0.092461,
    0.460996, 0.541627, 0.382924, 0.211163, 0.423665, 0.153273, 0.537126, 0.420518,
};

kernel void inception_3a_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[32];
    for (int f = 0; f < 32; f++) {
        sum[f] = inception_3a_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 192; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 32; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 32;
            }
        }
    }

    for (int f = 0; f < 32; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3b_1x1_bias[] = {
    0.517202, 0.193344, 0.784453, 0.319167, 0.453732, 0.335644, 0.178081, 0.236958,
    0.627706, -0.005765, 1.278699, 0.187317, 0.288272, -0.243884, -0.289435, 0.180679,
    0.371554, 0.390884, 0.225552, 0.980130, 0.777318, 0.246522, -0.206257, 0.734026,
    0.056641, -0.043843, 0.416048, 0.566722, 0.505036, 0.332219, 0.561527, -0.058659,
    0.439009, 0.018233, -0.060554, 0.114484, 0.066639, -0.085187, 0.682754, -0.170939,
    0.842984, 0.210480, 0.269055, 0.062849, 0.871779, 0.193191, 0.084931, 0.281544,
    -0.019280, -0.189493, -0.006848, 0.167934, 0.783327, 0.835426, 0.181710, 0.503638,
    -0.197890, 0.517899, 0.346322, 1.673057, 0.154697, 0.433978, 0.053449, 0.533498,
    0.101809, 0.401641, 0.139758, 0.229351, 0.509939, 0.039810, -0.433698, 0.128053,
    0.452310, 0.706880, -0.074171, 0.251628, 0.292691, 0.308470, -0.044505, 0.350364,
    -0.105974, 0.114738, 0.321358, 0.086395, -0.079365, 0.611093, 0.274140, 1.629840,
    -0.383016, -0.225822, -0.071273, -0.064651, -0.434405, 0.358937, 0.293724, -0.029000,
    0.223522, -0.397158, 0.829026, 0.338199, 0.499590, 0.098880, 0.381819, 0.694202,
    0.542432, 0.244378, 0.143056, 0.956080, 0.180302, 0.144639, 0.145388, 0.260161,
    0.149008, -0.106471, -0.313397, 0.855570, -0.085109, 0.780079, 0.658532, 0.897885,
    0.186714, 0.080251, 0.942741, -0.052523, 0.417881, 0.354804, -0.169631, 0.069935,
};

kernel void inception_3b_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_3b_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 256; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3b_3x3_reduce_bias[] = {
    0.484428, 0.623330, -0.226280, 0.883183, -0.525385, 0.475895, 0.296849, 1.310619,
    -0.014882, 1.123278, 0.280144, 0.810785, 0.506641, 0.316120, 0.527448, 0.630965,
    -0.029390, 0.292370, 0.439780, 0.782508, 0.782908, 0.265169, 0.613843, 0.808608,
    1.445138, -0.533400, 0.513659, 0.456619, 0.145234, 0.546545, 0.412531, 0.417108,
    0.519355, 0.751498, 0.295150, 0.230289, 0.479291, 0.074302, 0.598333, 0.620732,
    0.540873, 0.331526, 0.012146, 0.651930, 0.566192, 0.331811, 0.509891, 0.430682,
    0.572926, 0.624842, 0.629257, 0.528074, -0.065331, 0.444473, 0.308141, 0.429412,
    1.019109, 0.478225, -0.648597, 0.241801, 0.041926, 0.568298, 0.285303, 1.162663,
    0.470205, 0.954633, 0.518302, 0.657462, 0.303637, -0.482006, 0.270016, 0.547468,
    0.011682, 0.450142, 0.355767, 0.771841, 0.220933, -0.029698, 0.306299, 0.355412,
    0.512841, 0.784008, 0.417809, -0.412646, 0.431704, 0.826796, 0.400976, 0.347710,
    0.140952, 0.693478, 0.544461, 0.650577, 0.697430, 0.664652, 0.578560, 0.751319,
    0.028246, 0.109533, 0.330246, 0.194400, 0.392565, 0.203250, 0.615823, 0.322200,
    0.318039, 0.349593, 0.127769, 0.525453, 0.335054, 0.755499, -0.044888, 0.412917,
    0.524356, -1.030278, 0.420263, 0.468836, 0.459429, 0.021657, 0.334782, 0.358241,
    -0.566681, 0.752849, -0.116033, -0.042913, 0.277491, 0.290845, 0.608392, 0.475003,
};

kernel void inception_3b_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_3b_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 256; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3b_3x3_bias[] = {
    0.223271, 0.170274, 0.325534, 0.001825, 0.230001, 0.094742, 0.231181, 0.484654,
    0.485094, 0.342579, 0.030072, -0.064400, 0.487218, 0.377388, 0.414962, 0.352970,
    -0.034906, 0.150254, 0.194524, 0.220880, 0.252389, 0.234761, 0.472692, 0.512878,
    0.067815, 0.064452, 0.369551, 0.334699, 0.220259, -0.144634, 0.304351, 0.330189,
    0.419256, 0.449978, 0.184854, 0.196994, 0.194138, 0.621898, 0.225909, 0.224113,
    0.172876, 0.334963, -0.031079, 0.284043, -0.006084, 0.079230, 0.346489, 0.437436,
    0.109467, 0.254232, 0.366738, 0.232141, 0.173080, 0.102024, 0.215183, 0.239195,
    0.286946, 0.400702, 0.554964, 0.220729, 0.331863, 0.426871, 0.035317, -0.169737,
    0.110280, 0.172322, 0.360288, 0.078471, 0.252913, 0.307922, 0.172354, 0.421479,
    0.408021, 0.132124, 0.274130, 0.028840, 0.089843, 0.365539, 0.070814, 0.042280,
    0.893758, 0.657523, 0.153164, 0.102353, 0.130411, 0.288360, 0.260319, 0.254302,
    0.187138, 0.021466, 0.330957, 0.267095, 0.337280, 0.404229, 0.382653, 0.015911,
    0.336236, 0.422807, 0.121191, 0.229236, 0.251299, 0.026165, 0.320606, 0.185604,
    0.282489, 0.412270, 0.150048, 0.321208, 0.018701, 0.423233, 0.396006, 0.460339,
    0.254752, 0.183425, 0.344852, 0.332707, 0.007536, 0.308788, 0.219634, 0.328697,
    0.107757, 0.155108, 0.394945, 0.070490, 0.070158, -0.017366, 0.111204, 0.366049,
    0.392387, 0.374073, 0.147854, 0.287120, -0.075448, 0.050361, 0.195134, 0.207426,
    0.140100, 0.353945, 0.366047, 0.302626, 0.429597, 0.198319, 0.991953, 0.543049,
    0.398266, 0.282143, 0.325003, 0.048087, 0.143016, 0.472201, 0.280440, 0.178184,
    0.130522, -0.025452, 0.262824, 0.649900, 0.588346, 0.250423, 0.316493, 0.027442,
    0.299795, 0.247707, 0.300967, -0.016030, 0.254838, 0.257278, 0.095177, 0.155683,
    0.019717, 0.109280, 0.370105, 0.279694, 0.274693, 0.376678, 0.148286, 0.166077,
    0.085558, 0.231749, 0.300759, 0.336878, 0.052956, 0.327219, 0.170257, -0.083968,
    0.211352, 0.439408, 0.268901, 0.362824, 0.036533, 0.206510, -0.002426, 0.369396,
};

kernel void inception_3b_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[192];
    for (int f = 0; f < 192; f++) {
        sum[f] = inception_3b_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 128; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 192; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 192;
            }
        }
    }

    for (int f = 0; f < 192; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3b_5x5_reduce_bias[] = {
    0.832936, 0.574529, 1.794232, 1.178153, 1.785562, 0.624555, 0.710381, 1.030853,
    0.145824, 0.577602, 0.543413, 0.442512, 0.021956, 0.408795, 0.129725, 0.592336,
    0.544084, 0.434475, 0.717093, 0.156934, 0.298283, 0.077281, 0.643191, 0.848562,
    0.799337, 0.735961, 0.245328, 0.947288, 0.722371, 0.259402, 0.085483, 0.256312,
};

kernel void inception_3b_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[32];
    for (int f = 0; f < 32; f++) {
        sum[f] = inception_3b_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 256; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 32; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 32;
            }
        }
    }

    for (int f = 0; f < 32; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_3b_5x5_bias[] = {
    0.310971, 0.069839, 0.323002, 0.105434, -0.001610, 0.363373, -0.102033, 0.644733,
    0.275540, -0.133308, 0.055490, -0.063250, 0.081637, -0.253112, 0.662786, 0.177877,
    0.000039, 0.111174, 0.230510, 0.040236, 0.218652, 0.019930, 0.213827, 0.225016,
    0.852450, 0.282444, 0.051016, 0.004536, 0.703043, 0.177598, 0.185346, 0.353390,
    0.197616, 0.749142, 0.005935, 0.289945, 0.082758, 0.263648, -0.097097, 0.572294,
    0.160732, -0.198985, 1.123239, 0.368178, 0.018092, 0.251019, 0.309360, 0.214136,
    -0.171548, -0.002130, 0.263994, 0.204378, 0.281472, 0.084481, 0.166836, 0.157864,
    0.162763, 0.090805, 0.254414, -0.011451, 0.261088, 0.194847, 0.140406, -0.031353,
    0.072302, 0.214331, -0.051398, 0.596024, 0.216573, 0.195637, 0.034467, 0.097652,
    0.438216, -0.099781, 0.208300, 0.195783, -0.140913, 0.758381, 0.063438, 0.665457,
    0.759231, 0.152735, 0.049546, 0.068227, 0.090516, 0.180551, -0.002562, 0.113001,
    0.048684, -0.013200, 0.180205, 0.383765, 0.351673, -0.138268, -0.006922, 0.237328,
};

kernel void inception_3b_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[96];
    for (int f = 0; f < 96; f++) {
        sum[f] = inception_3b_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 32; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 96; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 96;
            }
        }
    }

    for (int f = 0; f < 96; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_3b_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_3b_pool_proj_bias[] = {
    0.276613, 0.372262, 0.137782, 0.256473, 0.330985, 0.446550, 0.225437, 0.428935,
    0.148198, 0.177414, 0.255645, 0.329028, 0.187252, 0.111059, 0.159187, 0.244891,
    0.161223, 0.099017, 0.224051, 0.225699, -0.060474, 0.131604, 0.300311, 0.422964,
    0.211121, 0.467511, 0.063219, 0.524457, 0.239408, 0.911071, 0.353067, 0.305860,
    0.253985, 0.358665, 0.482895, 0.351763, 0.174324, 0.052196, 0.130628, 0.294740,
    0.079137, 0.162622, 0.330155, 0.483323, 0.133000, 0.148438, 0.299794, 0.171247,
    0.126320, 0.190610, 0.177909, 0.157635, 0.075089, 0.131430, 0.267873, 0.274738,
    0.327808, 0.122926, 0.083257, 0.502988, 0.157476, 0.271984, 0.237958, 0.294487,
};

kernel void inception_3b_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_3b_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 256; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 28 && x+dx >= 0 && x+dx < 28) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void pool3_3x3_s2_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 2;
    const int strideW = 2;
    const int padH = 0;
    const int padW = 0;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_4a_1x1_bias[] = {
    -0.096732, 0.522425, -0.217666, -0.392352, -0.148984, 0.636029, 0.678981, -0.057715,
    -0.071809, -0.319611, -0.073878, -0.303381, -0.346494, 0.418386, 0.012918, -0.273674,
    0.605584, 0.113191, -0.803179, 0.005296, -0.403941, 0.431988, -0.617834, -0.068119,
    0.820109, -0.100621, 0.178585, 0.033484, 0.508756, 0.003924, -0.696776, -0.202471,
    -0.383181, 0.096378, 0.573906, -0.452863, -0.012400, -0.165083, 0.193962, -0.271268,
    0.104685, -0.208633, 0.290655, -0.094228, -0.164015, 0.385706, 0.260306, -0.056447,
    0.156726, -0.313045, 0.344351, -0.342482, -0.465934, -0.021368, 0.358284, 0.029610,
    0.302923, -0.305318, -0.311820, -0.292920, 0.628808, 0.483363, -0.486410, -0.116283,
    -0.655261, 0.109193, 0.271967, -0.270821, -0.139007, 0.012577, 0.271995, -0.210138,
    0.287200, 0.225075, -0.365330, -0.324632, -0.437788, 0.380665, 0.384747, -0.117667,
    -0.286743, 0.213610, 0.161654, -0.350869, -0.023213, 0.652640, 0.170851, 0.305383,
    -0.354215, 0.414852, -0.032156, -0.461619, -0.082803, -0.185523, -0.102393, -0.004084,
    -0.080093, -0.457432, 0.634138, 0.019884, -0.121452, -0.407730, 0.047012, 0.645831,
    -0.244859, 0.020827, 0.619551, 0.298953, -0.204877, -0.136437, -0.210939, -0.255995,
    -0.391340, -0.255843, 0.003248, 0.804268, -0.196461, 0.112949, -0.204358, 0.215710,
    0.450044, 0.244538, -0.175991, -0.504282, -0.408794, 0.168824, 0.551947, 0.661609,
    0.517616, 0.240200, -0.283354, -0.275529, -0.236575, 0.728244, 0.185469, -0.760195,
    0.282321, -0.474874, 0.491931, 0.211603, 0.008978, 0.037728, -0.304584, 0.444465,
    -0.390732, -0.342995, -0.053346, -0.191781, 0.176060, -0.151312, 0.203581, 0.240706,
    -0.024006, 0.148958, 0.042695, 0.094698, -0.104491, -0.482442, 0.038821, -0.413785,
    -0.045698, 1.098266, -0.232251, -0.208882, -0.531735, 0.248134, 0.437724, -0.057790,
    -0.243967, -0.354804, -0.221634, 0.007190, -0.369981, 0.110081, 1.100466, 0.072209,
    -0.136629, -0.057743, -0.129228, 0.326956, -0.206367, -0.212608, -0.635389, -0.220111,
    -0.157289, 0.765137, -0.112530, 0.102027, -0.025701, 0.310938, 0.223109, 0.543163,
};

kernel void inception_4a_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[192];
    for (int f = 0; f < 192; f++) {
        sum[f] = inception_4a_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 480; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 192; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 192;
            }
        }
    }

    for (int f = 0; f < 192; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4a_3x3_reduce_bias[] = {
    0.405404, 0.971131, 0.560198, 0.256676, -0.510361, -0.794153, 0.490313, -1.529760,
    -0.263374, 0.026761, 1.568448, 0.047147, -0.580085, 3.428455, 0.847212, 0.767717,
    0.515283, -0.432662, 0.310727, 0.473030, 0.366429, 0.165733, 1.277436, 0.889163,
    0.064899, 0.064811, -0.125794, 0.380364, 0.221697, 0.074829, 0.043056, -0.025990,
    0.240852, -0.079771, 0.407935, 1.545148, -0.367350, -0.219142, 0.125910, 0.654829,
    0.417810, 0.125894, 1.174077, 0.305221, 0.170240, 0.306077, -0.137344, 0.202840,
    0.016284, 0.512643, 0.790540, 2.797552, 0.088335, 0.954809, 0.193598, -0.040657,
    1.619518, 0.479987, 0.344677, 0.457596, 0.876863, 0.977815, 0.429060, -0.170005,
    0.414977, 0.362674, 1.759708, 1.282248, 0.180126, 0.078880, 0.067900, -0.217827,
    0.661758, 0.512325, 0.167506, 0.408711, 0.227239, 0.404894, -0.250885, -0.092492,
    1.133505, 0.927847, 0.282207, 0.915894, 0.435151, 0.623501, 0.526074, 0.238675,
    0.231384, -0.098127, 1.106672, 0.450837, -0.093429, 0.235329, 0.329361, 0.732357,
};

kernel void inception_4a_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[96];
    for (int f = 0; f < 96; f++) {
        sum[f] = inception_4a_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 480; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 96; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 96;
            }
        }
    }

    for (int f = 0; f < 96; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4a_3x3_bias[] = {
    0.416427, 0.324993, 0.133941, 0.319330, 0.183009, 0.725896, 0.434757, 0.152795,
    0.276879, 0.379512, -0.439358, 0.443460, 0.119089, 0.224874, 0.435086, 0.394378,
    0.264176, 0.696628, 0.374076, 0.141310, 0.272906, 0.202382, 0.361352, 0.215527,
    0.241483, 0.121554, 0.437376, 0.041112, 0.583172, 0.634915, 0.187317, 0.419912,
    0.095989, 0.196853, 0.101164, 0.238065, 0.271984, 0.341503, 0.198215, 0.279185,
    0.548332, 0.278195, 0.156250, 0.499398, 0.065736, 0.695368, 0.229853, 0.167674,
    0.211695, 0.392684, 0.309421, 0.474313, 0.197313, 0.398875, 0.285900, 0.359329,
    0.506995, 0.067729, 0.194068, 0.288061, 0.444048, 0.502489, 0.312988, 0.287463,
    0.338430, 0.262145, 0.389433, 0.365353, 0.381749, 0.087078, 0.374379, 0.188193,
    0.020989, 0.250111, 0.298294, 0.170979, 0.414556, 0.205838, 0.404277, 0.346683,
    0.104009, 0.678043, 0.744088, 0.267508, 0.404330, 0.395717, 0.327188, 0.179112,
    0.141308, 0.049377, 0.247993, 0.388643, 0.317789, 0.180996, 0.542695, 0.390954,
    0.231497, 0.291899, 0.229497, 0.271718, 0.294001, 0.331067, 0.135896, 0.267950,
    0.563556, 0.196045, 0.259265, 0.026513, 0.252445, 0.239416, 0.505293, 0.171447,
    0.025047, 0.159370, 0.087364, 0.344940, 0.064021, 0.321111, 0.217778, 0.172454,
    0.104152, 0.408933, 0.319444, 0.146042, -0.602021, 0.591972, 0.147340, 0.296988,
    0.403825, -0.226270, 0.247927, 0.252202, 0.169962, 0.194158, 0.448534, 0.192638,
    0.030839, 0.245185, 0.286128, 0.108052, 0.027983, 0.075956, 0.587682, 0.329789,
    0.750594, 0.360208, 0.445752, 0.255049, 0.273881, 0.239663, 0.462938, 0.131272,
    0.237467, 0.141123, 0.615932, 0.078796, -0.148889, 0.346178, 0.377603, 0.445312,
    0.082124, 0.224534, 0.563746, 0.345386, 0.211067, 0.190723, -0.000200, 0.237469,
    0.062995, 0.009635, 0.197137, 0.217595, 0.012417, 0.395860, 0.396174, 0.453212,
    0.022138, 0.283536, 0.185639, 0.334704, 0.068755, 0.284133, 0.435210, 0.260906,
    0.348436, 0.629151, 0.255827, 0.417281, 0.296457, 0.144532, 0.387587, 0.215197,
    0.343114, 0.193766, 0.554422, 0.438995, 0.261566, 0.493430, 0.205752, 0.429411,
    0.095368, 0.387375, 0.320423, 0.149914, 0.325786, 0.553327, 0.466260, 0.355189,
};

kernel void inception_4a_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[208];
    for (int f = 0; f < 208; f++) {
        sum[f] = inception_4a_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 96; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 208; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 208;
            }
        }
    }

    for (int f = 0; f < 208; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4a_5x5_reduce_bias[] = {
    1.942451, 0.664249, 2.198740, 0.467953, 0.708753, 2.169502, 0.775304, 0.781113,
    0.217476, 0.764576, 1.205428, 1.136084, 0.561078, 0.684667, 0.402248, 1.042604,
};

kernel void inception_4a_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[16];
    for (int f = 0; f < 16; f++) {
        sum[f] = inception_4a_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 480; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 16; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 16;
            }
        }
    }

    for (int f = 0; f < 16; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4a_5x5_bias[] = {
    0.093207, 0.271874, 0.682746, 0.536251, 0.165826, 0.431015, 0.402202, 0.391268,
    0.866706, 0.547557, 0.301560, 0.295370, 0.538365, 0.342327, 0.130225, 0.260505,
    0.284573, 0.700281, -0.015808, 0.308223, 0.550344, 0.356198, 0.646071, 1.183410,
    0.559368, 0.274459, -0.038232, 0.321191, 0.415616, 0.108708, 0.459646, 0.151152,
    0.281951, 0.343231, 0.329826, 0.842050, 0.255449, 0.662117, 0.349543, 0.490488,
    0.242759, 0.365260, 0.306641, 0.593387, 0.615900, 0.264461, 0.355654, 0.152631,
};

kernel void inception_4a_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[48];
    for (int f = 0; f < 48; f++) {
        sum[f] = inception_4a_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 16; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 48; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 48;
            }
        }
    }

    for (int f = 0; f < 48; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_4a_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_4a_pool_proj_bias[] = {
    0.178076, 0.447315, 0.014591, 0.097954, 0.759264, 0.619243, 0.655725, 0.390604,
    0.191496, 0.319661, 0.442270, 0.297723, 0.352767, 0.549787, 0.824009, 0.499427,
    0.478684, 0.272364, 0.269049, 0.436549, 0.384885, 0.180703, 0.100216, 0.349327,
    0.147806, 0.195987, 0.692255, 0.159624, 0.290529, 0.454224, 0.486683, 0.298079,
    0.209399, 0.279510, 0.286252, 0.153607, 0.719790, 0.740491, 0.510688, 0.372466,
    0.719319, 0.145392, 0.110574, 0.438921, 0.421559, 0.162292, 0.383198, 0.811379,
    0.179335, 0.273580, 0.451180, 0.200116, 0.325868, 0.179819, 0.254223, -0.095421,
    0.612177, 0.435756, 0.322302, 0.471047, 0.241324, 0.047487, 0.294849, 0.455967,
};

kernel void inception_4a_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_4a_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 480; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4b_1x1_bias[] = {
    -0.102160, 0.095605, 0.851112, 1.148678, 0.479168, 0.143876, 0.479829, -0.039933,
    1.527872, 2.665528, -0.377209, 0.825424, 0.036142, 0.413453, -0.274239, 0.303719,
    0.823797, 1.184425, 1.434679, 0.481040, 0.996481, 0.878751, 2.189200, -0.478463,
    1.377912, 0.725561, 1.837699, -0.133791, 1.306535, -0.151691, 1.779348, 0.013123,
    0.105745, -0.280271, 1.580653, 0.778487, 2.142767, 0.207891, 1.638904, 0.833302,
    0.178311, 1.366296, 0.746168, 0.233707, 0.804246, 1.017430, 1.357319, 0.366007,
    0.753578, -0.521148, -0.341678, 1.600122, 0.575877, 1.472119, 1.137577, 1.054114,
    0.179482, 1.517427, 0.861934, 1.121493, 0.360183, -0.182119, 0.130405, 0.771538,
    1.102561, 2.009426, 0.897346, 1.398131, -0.415237, 0.193028, 1.508687, 1.028912,
    1.390372, -1.183248, -0.521401, -0.177962, -0.345734, 0.059063, 2.477219, 0.850990,
    0.145522, 1.384443, 0.507793, 0.115127, -0.373204, 0.679673, 1.709573, 1.349866,
    0.593889, 0.825300, 0.045799, -0.715085, 0.922380, 1.780406, 0.608753, 1.007209,
    1.889052, 0.119224, 1.452273, 1.330054, -0.245513, 0.956836, 1.052452, -0.314633,
    -0.882852, 0.748499, 0.757258, -0.355956, 0.800419, 1.035481, -0.541667, 0.397278,
    0.589840, -0.773705, 0.151419, -0.583837, 1.224232, 0.421002, 0.424119, -0.183036,
    0.640336, 0.722084, 1.076569, 0.647169, 0.471616, -0.096468, -0.281795, 0.962629,
    1.344637, 1.108392, 0.852666, 0.028270, 0.500989, 0.149133, 0.264743, -0.508984,
    0.167883, 0.677400, -0.396021, 0.951626, 0.843163, -0.053556, -0.380497, -0.003069,
    0.907750, 0.145800, 1.034744, 0.328137, 1.648077, -0.075659, 0.985493, 0.256802,
    2.436814, 0.341296, 1.193733, 0.921636, 0.570421, 0.404675, 0.910326, 2.492461,
};

kernel void inception_4b_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[160];
    for (int f = 0; f < 160; f++) {
        sum[f] = inception_4b_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 160; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 160;
            }
        }
    }

    for (int f = 0; f < 160; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4b_3x3_reduce_bias[] = {
    1.334772, 0.000083, 1.416805, 0.389943, 0.660989, 0.670799, -0.243151, -0.461414,
    0.398226, -1.755023, 0.056562, 0.149180, 1.189510, -0.356174, 0.287293, 0.375755,
    0.101214, 0.582954, -0.076526, -0.000395, 1.065399, 0.610417, 1.316601, 0.584164,
    0.916147, 0.196367, -0.208310, -0.255333, -1.585272, 1.627663, 0.862194, 1.392759,
    -0.146506, 0.033421, 0.533258, 1.372188, 0.041098, 1.066136, -0.340962, -0.481890,
    -0.362929, 1.167907, -0.583001, 1.304231, -1.439183, 0.508762, 0.017326, 1.130388,
    -0.226699, -0.438468, 0.183484, 0.439333, -0.472428, -1.113280, 0.767938, 0.489744,
    0.096420, 0.462643, 0.784092, -0.055773, 0.385221, 1.256520, 0.721704, 0.779337,
    -0.850555, -0.398731, -0.429702, -0.147329, -0.208732, 0.840516, 0.226543, -0.225870,
    -2.034430, 1.233473, 1.047176, 0.188466, 1.312170, 0.753814, 0.222405, 1.155337,
    0.547965, 0.739453, -0.020735, 1.307605, 0.744603, 0.370065, 0.000124, 0.681730,
    2.644716, 0.977424, -1.511046, 0.611203, -0.188690, 0.426854, 1.485333, 1.352046,
    -1.096465, 0.064346, 1.173861, 0.822230, 0.179255, 0.959876, 0.429644, 0.449421,
    0.595503, -0.964095, 0.848090, 2.547104, 1.078282, 0.340987, 1.167141, -0.598332,
};

kernel void inception_4b_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[112];
    for (int f = 0; f < 112; f++) {
        sum[f] = inception_4b_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 112; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 112;
            }
        }
    }

    for (int f = 0; f < 112; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4b_3x3_bias[] = {
    0.136046, 0.671658, 0.837720, 0.385417, 0.263417, 0.481237, 0.737791, -1.940393,
    0.541737, 0.764130, -0.439639, 1.039367, 0.518134, 0.423545, 0.786343, 0.097540,
    0.554686, 1.210303, 0.035578, 0.393147, -0.438624, 0.510682, 0.461858, 0.177318,
    0.296381, 0.224760, 0.114799, 0.069258, 0.476835, 0.507525, 0.927182, 0.126703,
    0.531023, 0.216531, 0.732223, 0.382334, 0.644277, 0.908500, 0.198702, 1.350001,
    0.755161, 1.256081, 0.249613, 0.894528, 0.675978, 0.693410, 0.357188, 0.648308,
    -0.445798, 0.277619, 0.382462, 0.644604, 1.250144, 0.219809, 0.727401, 0.519146,
    0.011311, 0.278681, -0.140091, -0.470869, 0.745068, 1.485677, 0.506193, 0.740214,
    0.348894, -0.201684, 0.759330, 0.975191, 0.905444, 0.357378, 0.097168, 1.148649,
    0.300399, 0.171003, 0.407812, 0.416573, 0.293027, 1.788846, 0.258086, 0.524108,
    0.394513, -1.259779, 0.154127, 0.295929, -1.379815, 0.742868, -0.226090, 0.901581,
    0.824762, 0.934529, 1.238373, 0.322777, 0.470316, 0.767120, 0.162757, 0.380746,
    0.792924, 0.287028, -0.031553, 0.332892, 0.255812, 0.401937, 0.914238, 0.835980,
    -0.547550, 0.512782, 0.445190, -0.051283, -0.423916, -0.080860, 0.160642, 0.551850,
    0.299218, 0.629418, 0.173148, 0.608116, -3.470115, 0.899077, 0.648139, 0.453595,
    0.517313, 1.284661, 0.287666, 0.568200, 1.152938, 0.165665, 0.619300, 0.551064,
    1.315070, 1.107294, 0.392804, 0.191428, 0.799322, 0.311008, 1.215355, 0.141091,
    0.509628, 0.883030, -0.008543, 0.157298, -0.159415, 0.534418, 1.716992, 0.533298,
    1.122761, 0.711851, 0.123427, 0.450655, 0.869892, 0.479006, 1.144634, 0.138141,
    0.064673, -0.066886, 1.471283, 0.818909, 0.266253, -0.327833, 0.777184, 0.313474,
    0.491682, 0.353605, 0.517527, 0.276006, -0.060120, -0.112247, 0.531915, 0.686519,
    0.241879, 1.351261, 0.461015, 2.899051, 0.502698, 1.351804, 0.417818, 0.451851,
    0.097127, 0.580420, 1.553673, 0.771677, 0.581176, 1.039148, 0.450125, 0.589442,
    0.115807, 0.287083, 0.281564, 0.357465, 0.677953, -0.229044, 0.370320, 0.505448,
    0.428717, 0.610117, 0.494480, 0.765614, 0.315878, 0.603755, 0.580837, 1.506619,
    0.741418, 0.741612, 0.903239, 1.177304, 0.839904, 0.609260, 0.780828, 0.261573,
    0.398711, 0.490644, 0.515519, 0.058579, 0.151586, 0.076134, 0.613451, 0.494878,
    0.660042, 0.642728, 0.784868, 0.087330, 0.696137, -0.024935, 0.548973, 0.405289,
};

kernel void inception_4b_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[224];
    for (int f = 0; f < 224; f++) {
        sum[f] = inception_4b_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 112; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 224; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 224;
            }
        }
    }

    for (int f = 0; f < 224; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4b_5x5_reduce_bias[] = {
    1.871005, 0.898777, 4.316381, 7.077267, 2.909665, 2.285532, 1.927296, -0.809216,
    3.282066, 1.624092, -1.504506, 4.365613, 2.856409, 0.595743, 3.299976, 1.898857,
    2.588965, 0.575375, 1.385070, 1.990859, 2.421601, 0.078018, 0.929501, 0.195170,
};

kernel void inception_4b_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[24];
    for (int f = 0; f < 24; f++) {
        sum[f] = inception_4b_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 24; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 24;
            }
        }
    }

    for (int f = 0; f < 24; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4b_5x5_bias[] = {
    0.578404, 1.292307, 0.435831, 0.886690, 0.940671, 0.443458, 0.500853, 0.388765,
    0.540507, 0.074830, 2.933688, 0.599732, 0.641516, 1.701619, 1.897900, 0.539208,
    0.451707, 0.568782, 0.485158, 2.945876, 0.325996, 0.355953, 0.843269, 0.306687,
    0.390592, 0.531449, -0.799064, 0.224680, 0.482272, 2.038838, 0.212978, 1.072737,
    1.423964, 0.711048, 1.062335, 0.572496, 0.167074, 0.760059, 0.638885, 0.411867,
    1.214740, 0.197918, 1.449732, 0.239122, 0.788379, 0.809734, 0.846479, 0.674098,
    0.846059, 2.643107, 0.559670, 1.182625, 2.407684, 0.717179, 0.011873, -1.179580,
    0.406343, 0.231350, 0.185722, 0.722600, 0.478009, 0.614894, -0.136071, 1.314121,
};

kernel void inception_4b_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_4b_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 24; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_4b_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_4b_pool_proj_bias[] = {
    0.069124, 0.085376, 0.226558, 0.514309, 0.094247, 0.344292, -0.117405, 0.623216,
    0.424084, 0.271261, 0.134808, 0.143434, 0.634639, 0.346586, -0.139078, 0.448638,
    -0.001677, 0.330806, 0.426075, 0.284946, 0.221609, 0.847895, -0.065087, 0.326714,
    0.171078, 0.358036, 0.094076, 0.088557, 0.114752, 0.574392, 0.410845, 0.140357,
    0.452417, 0.529947, 0.466524, 1.251947, 0.313242, -0.235460, 0.199734, 0.082840,
    0.418177, 0.360931, 0.245471, 0.465070, 0.923323, 0.478781, -0.208671, -0.097927,
    0.172941, -0.022231, 0.234956, 0.338400, 1.279101, 1.225799, 0.332231, 0.424353,
    -0.258124, -0.167480, 0.581330, 0.164395, 0.113839, 0.547525, 0.155552, 0.240651,
};

kernel void inception_4b_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_4b_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4c_1x1_bias[] = {
    0.565275, 0.230773, -0.162081, 0.888391, 1.365102, 2.051698, 2.615906, 1.186683,
    1.850053, 1.008139, 0.644996, 2.707218, 0.927269, 2.063584, 1.829675, 0.075320,
    1.350179, 1.873962, 0.443941, 1.129002, 1.507170, 0.840973, 0.910471, 1.335924,
    0.866264, -0.060366, 0.210244, 2.726340, 0.782436, -0.176490, 2.038784, 0.140509,
    0.276051, 0.086465, 0.764419, 0.505502, 0.063243, 0.929007, 0.715022, 1.015763,
    1.200117, 0.996815, 2.242625, -0.179697, 1.852896, 1.395849, 0.635548, 1.370326,
    1.017221, 1.215556, 1.134199, 1.181988, 0.765125, 1.159135, 0.857140, 0.921754,
    -0.248233, 1.148142, 1.913760, -0.790684, -0.307521, 0.555475, 1.425424, 1.058713,
    1.281172, 3.052709, 0.893180, 0.733974, 0.617736, 0.624615, 0.479779, 0.449002,
    -0.483964, 0.847672, 1.563845, 1.276744, 0.925909, 0.272733, 2.374616, -0.254426,
    -0.415239, 1.282881, 2.834146, 0.458716, 1.150261, 1.624880, 1.024662, 0.731893,
    0.855287, 0.917634, 2.308857, -0.084553, 1.283963, 2.300787, 0.432381, -0.513604,
    1.124692, 0.645460, -0.008893, 0.653513, -0.012628, 1.210188, 0.620839, 1.495468,
    1.362787, 1.265362, 1.297214, 0.881734, 1.107360, 2.665394, 2.491129, 1.549735,
    -0.091216, 0.979634, 1.190411, 0.413484, 0.231955, 1.719903, 2.579929, 0.702090,
    0.225667, 1.456363, -0.073847, 0.947063, 0.759288, 0.450308, 0.410781, 1.691767,
};

kernel void inception_4c_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_4c_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4c_3x3_reduce_bias[] = {
    -1.511352, -0.344962, 1.179846, 0.374327, 1.100259, 0.790672, 2.030174, 2.649236,
    0.267407, 0.437297, 2.681054, 0.752997, 2.503152, -0.989969, 0.502429, 0.690552,
    -1.279658, 1.054774, 1.000561, -0.704312, -0.824641, -0.083922, 0.228074, 0.515467,
    -1.182747, 0.700318, 1.124908, -0.239616, 1.264799, 1.418039, 0.104274, 1.852803,
    -0.694446, -2.757429, 1.610132, -0.160702, -0.469654, 0.533899, -1.515811, 2.208556,
    2.082071, 0.643588, 0.754868, -0.032967, 0.474778, 1.474353, -0.797615, 1.692487,
    0.480366, 1.427852, -0.199175, 1.664835, 1.015266, -0.714808, 1.497359, 0.458793,
    0.728692, 0.932879, 0.923428, 1.743435, 0.616557, -0.430992, -1.489137, -1.150455,
    0.309329, -0.172624, -0.599678, -1.068419, -0.768881, -0.056953, 1.584143, 1.134116,
    1.073119, 1.600999, 2.648067, -2.233451, 0.806185, 3.839690, -0.470467, -5.481684,
    0.970659, 0.194860, -1.905462, 0.095249, 0.691441, 1.231754, 0.038260, -0.483140,
    0.943647, 1.227446, 0.372973, 1.180858, -0.009388, 2.066556, 2.770721, 1.326749,
    1.595512, 1.415594, -2.491801, -0.115755, 0.335910, 1.849463, 1.642340, 0.682807,
    1.862275, 1.179578, 1.204226, 0.144720, -0.444078, 2.139454, -3.174226, 2.429512,
    -0.687889, 1.302685, 0.437483, 0.576637, -2.237102, 0.817717, 0.859405, 0.228348,
    -0.692051, 0.941725, 0.212245, 0.875519, 0.155826, -4.597950, 1.555379, 2.237147,
};

kernel void inception_4c_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_4c_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4c_3x3_bias[] = {
    0.783225, 0.373694, 0.162489, 1.013702, 2.149110, 1.001713, 0.372050, 0.984167,
    0.318119, -0.985018, 0.739004, 0.965502, 1.809923, 0.639104, 0.532080, 0.726425,
    0.705341, 0.512766, 0.501687, 0.629955, 0.152790, 1.096720, 1.386472, 1.469833,
    0.038264, 0.191575, 0.894652, 0.764227, 0.263525, 0.063797, 0.376741, 1.534487,
    0.563464, 1.049852, 0.609601, 1.132667, 1.016009, 0.866332, 1.917132, 0.617960,
    1.763254, 0.526796, -2.393569, 0.917869, 0.065895, 0.536022, 0.622449, -0.060053,
    1.279768, 0.811158, 1.266693, 0.995575, 1.064851, 0.227613, 0.316119, 0.410652,
    -4.485679, 1.123585, 1.340498, 1.652575, 0.729833, 0.723977, 0.466733, 0.401619,
    0.657550, 0.395807, 0.363753, 1.118688, 0.140718, 0.139187, 0.974577, 1.014264,
    0.499027, 0.512202, 1.350246, 0.457757, 0.769284, -0.141068, 0.705231, 0.647551,
    0.374189, 0.983901, 0.760841, 0.390647, 0.081340, 0.883187, 0.464517, 0.597140,
    0.572504, 0.822291, 0.995452, -0.697419, 0.693596, 1.645233, 0.710276, -0.221970,
    1.015219, 1.089286, 0.462883, 0.885770, 1.606912, 1.588153, 0.711202, 0.398724,
    1.171455, 0.369820, 0.508929, 1.128808, 1.456469, 0.367196, 0.419223, 0.967878,
    0.746033, 1.375500, 1.028579, -0.098574, 1.855348, 1.224944, 1.048237, -0.165199,
    0.574023, 0.693227, 0.559466, 0.500709, -0.152627, 0.616015, 1.105691, 2.155587,
    0.362958, 0.384655, 0.933478, 0.583753, 0.879994, 0.364444, 0.649387, 0.501093,
    1.728658, 0.852210, 0.562047, 1.400952, 0.537840, 1.130679, 0.594505, 0.433427,
    -1.577044, 0.474738, 0.891418, 0.822989, 1.090733, 0.297942, 1.784131, 1.595734,
    1.604533, 0.907919, 0.733492, 0.201567, 2.037445, 1.022568, 1.164364, 0.375204,
    0.977552, -7.887056, 0.755856, 0.400987, 1.570005, 2.026537, 0.952978, 2.074304,
    0.195985, 1.020885, 0.462163, -0.195498, 0.192461, 0.859092, 1.551933, -1.893558,
    0.612908, 2.501345, 0.341863, 0.626534, 0.858652, 0.984428, 1.051280, 0.830261,
    0.958696, 1.224050, 0.473328, 0.965744, 0.511118, -0.117847, 0.570805, 1.129153,
    0.373402, 0.831287, 0.172844, 1.112953, 1.173262, 0.482985, 0.643658, 0.914704,
    0.351756, 0.184117, 1.866478, 0.489627, 1.501246, 0.684369, 0.891213, 0.789774,
    0.700582, 1.232425, 0.177350, 1.176263, 0.235083, 0.541679, 1.975141, 0.411598,
    0.604204, 1.224010, 1.824355, 0.597743, 1.211776, 1.128960, 1.157887, 0.588275,
    0.622984, -2.148252, 0.601376, -0.312957, 1.500535, 0.802222, 0.435823, 0.630489,
    0.741457, 0.741584, 0.109211, 0.511693, 1.106427, 0.573996, -0.886935, 0.849084,
    0.625758, 0.038248, 1.034437, 0.467186, -0.111844, 0.476869, 1.115567, 1.161331,
    0.472280, 0.571197, 0.517547, 1.234962, 0.812231, 1.037495, -0.495154, 0.566316,
};

kernel void inception_4c_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[256];
    for (int f = 0; f < 256; f++) {
        sum[f] = inception_4c_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 128; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 256; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 256;
            }
        }
    }

    for (int f = 0; f < 256; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4c_5x5_reduce_bias[] = {
    3.358047, 0.878611, 2.837822, 2.237918, 2.134300, 2.489475, 2.823614, 2.132928,
    0.293524, 2.268060, 1.150648, 1.576333, 1.380060, 1.705821, -2.066873, -0.142672,
    0.095792, -0.043944, 2.602054, 2.169264, 1.249534, -0.667040, 1.295227, 2.654631,
};

kernel void inception_4c_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[24];
    for (int f = 0; f < 24; f++) {
        sum[f] = inception_4c_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 24; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 24;
            }
        }
    }

    for (int f = 0; f < 24; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4c_5x5_bias[] = {
    0.009294, 0.221075, 1.341030, 1.143741, 0.747516, 0.509765, 1.226884, 1.363130,
    1.398190, 0.098966, 0.504094, 0.522832, 0.809891, 0.408136, 1.832848, 0.848089,
    0.067223, 0.023292, 1.643892, 1.528272, 0.307559, 0.342663, 0.789177, 1.265532,
    1.288187, 0.154558, 0.874321, 0.404473, 0.991359, 1.112591, 0.152725, 0.472532,
    0.534671, 1.128994, 0.376463, 0.651756, 1.542423, 0.564475, 1.960304, 0.057760,
    0.645733, 0.812898, 2.821546, 0.320979, 0.965379, 0.420010, 0.659628, -1.193263,
    0.771895, 0.218948, 0.161812, 0.228942, -0.450284, 0.321004, 0.457809, 0.654113,
    0.322887, 2.729447, 1.118458, 1.049294, 0.178339, 0.318640, 1.001378, -0.441179,
};

kernel void inception_4c_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_4c_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 24; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_4c_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_4c_pool_proj_bias[] = {
    0.124351, 0.570115, -0.129772, 0.747200, 0.593334, 1.033037, 0.068627, 0.933599,
    -0.043905, 0.108708, 0.264074, 1.262984, 0.133527, 0.455497, 1.205949, 0.777528,
    0.132040, -0.369035, 0.835913, 0.038845, 0.661533, 0.156035, -0.226808, -0.398321,
    0.601536, 0.147340, 0.382228, 0.525680, 0.774827, 0.916731, 0.636008, 0.749379,
    0.113363, -0.396757, 0.511620, 0.484174, 0.853237, 0.333344, 0.773573, 0.508271,
    0.594429, 0.130768, 0.315022, 0.714836, 0.487247, 1.397138, 0.110846, -0.017304,
    0.774361, 0.070327, -0.142001, -0.016024, 0.357628, 1.216631, 0.415600, 0.418583,
    -0.222136, 0.913460, 0.513615, 0.648182, 0.004991, 0.528856, 0.613127, -0.187071,
};

kernel void inception_4c_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_4c_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4d_1x1_bias[] = {
    0.518615, 1.520392, 1.487150, 0.410016, 1.206407, 0.458074, 2.403826, 3.695153,
    0.876310, 0.147931, -0.027527, 0.675580, 1.066317, 0.976443, 0.103835, 0.857177,
    1.448851, 1.618305, 0.553099, 0.999879, 0.646343, 1.075626, 1.681242, 0.097239,
    0.582430, 0.152747, 2.162432, 1.997862, 1.054141, -0.040956, 0.637600, 1.649223,
    1.741147, 0.592691, -0.205566, 0.354064, 0.879331, 0.440800, 1.757232, 2.716657,
    1.297327, 1.145736, 0.840290, 1.321013, 1.415701, 0.646793, 0.553142, 0.847396,
    1.465126, -0.054225, 0.693908, 0.397080, 1.876608, 0.702560, -0.296887, 0.132208,
    1.054543, 1.860796, 1.204085, 0.879561, 2.287120, 1.646534, -0.147203, 0.290614,
    -0.021343, -0.469134, 2.218339, 0.806990, 2.952414, 1.420996, 0.734965, 0.775905,
    0.052652, 2.203814, 1.766920, 1.427930, 0.851277, 0.269539, 0.109726, -0.468251,
    0.984059, 1.262834, 1.033795, 0.088328, 2.033408, 0.214372, 1.067520, 0.846587,
    1.116884, 1.809180, 1.555406, 0.817616, 1.076625, 0.136494, 1.129551, 1.075861,
    0.408773, -0.515530, 1.381190, 0.719803, 1.400894, 1.193462, 0.459289, 2.212081,
    0.879599, 0.537717, 1.461209, 1.120888, 0.268896, 1.473885, 1.659207, 1.482274,
};

kernel void inception_4d_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[112];
    for (int f = 0; f < 112; f++) {
        sum[f] = inception_4d_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 112; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 112;
            }
        }
    }

    for (int f = 0; f < 112; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4d_3x3_reduce_bias[] = {
    0.581487, -0.592227, 1.938594, -4.442925, -0.145627, -0.485794, -0.120007, -0.397332,
    2.424687, 1.705863, 1.393228, -0.329495, -0.482830, 0.396438, 1.679421, 0.248424,
    -0.673500, -0.231360, 1.003891, -0.330579, 2.346540, 0.066819, 1.298646, -0.185415,
    1.911955, -14.644096, 1.990733, 1.153879, -0.300359, -1.057296, -2.185122, 0.391367,
    -2.702030, 1.583762, 0.418296, 1.537145, 0.091991, -1.166775, 1.254342, -2.732508,
    -1.415234, 0.917260, -2.405902, -1.656484, 3.408365, 1.747427, 1.006687, -0.061724,
    0.289192, -1.898692, -0.318802, 3.019159, 2.028072, 1.894938, 0.055917, -0.405613,
    -1.809938, -0.157807, 1.456755, -0.208361, 0.026305, 0.496981, -11.274090, 1.350610,
    0.829082, -1.169261, -0.285698, -2.671441, -2.052904, -0.371374, 0.419036, 2.397906,
    0.965873, -0.010167, 1.321848, 2.712046, -1.695359, -0.684174, 0.286642, -0.489597,
    2.384901, 0.173471, 0.396143, 1.454831, 0.683847, 0.243898, 1.639499, 0.207629,
    0.420325, -0.459574, 1.877506, 4.747359, -0.316142, 1.469512, 0.011461, -1.486295,
    2.290003, 1.134811, -0.804923, 0.933588, -0.546595, 1.077691, 1.481270, -0.993504,
    0.238758, -1.180771, 1.138809, 0.101970, 0.565060, 0.593275, 0.169167, 0.442382,
    0.468880, 1.893411, -3.324589, 1.302678, 0.920311, 1.790152, 0.159438, 0.940579,
    0.023717, -0.059950, -1.024318, 0.268631, 1.957021, 1.651279, -1.923174, 1.836090,
    -0.284294, 0.420718, 1.081891, 1.221321, -1.135422, 0.454595, -1.754038, 1.383369,
    -0.043084, 0.296091, 1.993382, -0.257092, -0.520694, 1.414366, -0.493838, -2.277025,
};

kernel void inception_4d_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[144];
    for (int f = 0; f < 144; f++) {
        sum[f] = inception_4d_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 144; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 144;
            }
        }
    }

    for (int f = 0; f < 144; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4d_3x3_bias[] = {
    1.201623, 1.030571, 0.106885, 0.153105, 1.146796, 1.535835, 1.135963, 0.044721,
    0.556431, 0.326433, 1.272544, 0.803881, 0.493536, 0.194729, 0.187973, -0.417260,
    0.407726, 0.445410, -0.107539, 1.430106, -0.446617, -1.747883, 1.004587, 1.488311,
    -1.391021, 0.917169, -1.248440, -0.090351, 0.755867, 0.101867, -0.337715, 1.252331,
    0.942427, 1.620535, 1.026007, 2.842015, 0.288936, 0.515218, 0.426458, 0.338300,
    1.027206, 1.109813, 0.564453, 0.289956, 0.782177, 0.696588, 1.138403, 1.117987,
    1.736320, 1.027090, -0.064036, 0.126815, 1.250388, 0.690456, 0.781455, 1.847121,
    -0.087441, 1.717304, 0.504599, 0.936190, -0.566762, 0.327914, 0.566808, 1.273026,
    -0.284167, 0.075575, 0.183594, 1.168813, 0.748249, 0.324976, 0.484673, 0.435625,
    -12.172002, 0.366410, 0.650894, 0.413466, 1.243397, 0.687630, 0.307106, 0.905516,
    -0.666541, 1.299568, 0.510464, 0.960917, 1.177762, -0.576547, 1.323925, 0.645279,
    1.034886, 1.437938, 0.242704, 0.913410, 0.956313, 0.697688, 0.750115, 1.907959,
    0.519466, 0.882804, 1.105868, 0.959036, 0.350226, 2.341306, 1.108159, -0.080037,
    0.265160, 1.996563, -0.157504, 1.109778, 0.282625, 1.624356, 1.203221, 1.400852,
    -0.022640, -0.589328, 0.908367, 0.454422, 0.111233, 1.328455, -0.272165, 1.008468,
    0.496809, 0.501000, -0.017883, 1.502101, -0.000534, 2.184644, 0.090664, 1.024516,
    2.105845, 1.170754, 0.445027, -0.378618, 1.324734, 1.727996, 2.033142, -0.042120,
    0.781359, 0.438643, 1.522183, 1.655360, 0.945990, -0.305415, 0.510365, 0.053662,
    0.917860, 1.574314, 0.324710, 0.361416, 0.329516, 0.461022, 0.825003, 2.398542,
    0.916635, 1.890332, 1.243658, 0.315037, 0.518799, 1.702860, -0.941503, 0.343168,
    -3.036220, -0.061481, 0.655159, 0.239969, 1.332465, 1.726612, 1.208382, 0.839811,
    0.273041, 0.682528, 0.770481, 0.680584, 0.615497, 0.490705, 1.592443, 2.131993,
    0.972261, 1.648880, 1.044002, 0.192401, 0.864179, 0.648382, 1.271517, 0.488683,
    -0.702655, 0.370961, 0.942049, 0.825846, 0.652978, 0.508094, 0.554312, 1.741139,
    0.690628, 0.454290, 0.055854, 0.177036, 0.795444, 1.970325, 1.688310, 1.479837,
    0.262437, 0.298798, 0.394029, 1.470879, 1.295702, 0.223816, 0.285578, 2.369241,
    0.832690, 0.310490, 1.223386, 1.289177, 0.911839, 0.348456, 1.714668, 0.775385,
    -2.689760, 0.669125, 0.443368, 0.606103, 0.605555, 0.355926, 1.327676, -6.839345,
    1.467879, 0.891486, 0.639994, -0.106614, 1.010619, 0.278325, 0.151711, 0.071761,
    1.211139, 1.186564, 1.127609, 0.513191, 1.173454, 1.163873, 0.763972, -0.167537,
    0.601510, 1.600189, 0.134415, 0.531762, 0.772411, 0.159957, 0.208792, -4.193827,
    0.337273, -0.673962, -4.900822, 0.150378, 1.714704, 0.628515, 1.064368, 0.441126,
    0.399605, 0.486032, 1.177148, 1.004270, -2.830842, 0.113166, 0.488027, 0.261775,
    -0.237080, 0.864403, 0.822373, 0.345167, 0.325414, 0.534761, 0.316953, 0.633902,
    0.456023, 0.186679, 0.690372, 0.222806, 0.909711, 0.902289, 0.618901, 1.317922,
    0.952493, 0.080608, 0.628280, 1.043838, 0.732940, 0.263712, 0.134229, 0.539898,
};

kernel void inception_4d_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[288];
    for (int f = 0; f < 288; f++) {
        sum[f] = inception_4d_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 144; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 288; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 288;
            }
        }
    }

    for (int f = 0; f < 288; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4d_5x5_reduce_bias[] = {
    2.813022, 2.461241, -1.553315, 1.238744, 2.592844, 1.833823, 0.446580, 1.380617,
    2.354211, 1.331198, 3.029017, 1.055599, 3.057658, 0.235997, 0.891368, 2.384303,
    1.870693, 1.497485, 1.839311, 2.167912, -2.332108, -0.212662, 3.109605, 1.647632,
    3.991693, 2.119417, 1.747749, -0.054752, 1.820362, 1.441555, 0.179892, -1.339841,
};

kernel void inception_4d_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[32];
    for (int f = 0; f < 32; f++) {
        sum[f] = inception_4d_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 32; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 32;
            }
        }
    }

    for (int f = 0; f < 32; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4d_5x5_bias[] = {
    0.315856, 1.204594, 0.211079, 0.145787, 0.828793, -0.080078, 0.195668, 0.661155,
    -0.095767, 3.064874, 1.055195, -0.058669, 2.237655, 0.407882, 1.147679, 0.849161,
    0.180535, 0.456638, 0.680141, 1.592703, 0.246426, 0.294572, 2.471150, 0.536490,
    0.684121, 0.698192, -0.425611, 0.207909, 0.511554, 0.514343, 0.572725, 1.074196,
    0.289159, 1.448603, 0.725601, 1.082013, 2.502660, -0.020261, 0.557353, 1.737235,
    0.947262, 0.443358, 0.522400, 1.640131, 2.563501, -1.301830, 1.537335, 1.248540,
    0.312139, 0.491829, 1.606031, 1.295026, -1.444418, 1.422623, 0.893833, -0.308189,
    1.509258, 0.281304, 1.139774, 0.507085, -1.249361, 1.428569, 0.703325, 0.393887,
};

kernel void inception_4d_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_4d_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 32; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_4d_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_4d_pool_proj_bias[] = {
    0.408912, 0.373502, 0.460278, 0.621156, -0.222098, 0.695135, 0.051373, 0.201476,
    0.191152, 0.368263, -0.299430, 0.509465, 0.194315, 0.514059, -0.165912, 0.423724,
    0.506666, 0.236437, 0.496470, 1.037121, 0.341890, 1.120096, 0.448979, 1.252735,
    0.902588, -0.009458, 0.331539, 0.541538, -0.124602, 0.132716, 1.025842, 0.808539,
    -0.011825, -0.413170, 0.670940, -0.569855, 0.096140, 0.719561, 0.433416, -0.029867,
    1.346566, 0.137150, -0.603828, 0.142679, 0.060602, 0.100782, -0.392140, -0.232212,
    0.538949, 0.911865, 1.718758, 0.730730, 0.264037, 0.268271, 0.227186, -0.238239,
    0.145048, -0.444891, 0.372020, 1.354259, -0.468855, -0.412464, 0.212441, 0.494402,
};

kernel void inception_4d_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[64];
    for (int f = 0; f < 64; f++) {
        sum[f] = inception_4d_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 512; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 64; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 64;
            }
        }
    }

    for (int f = 0; f < 64; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4e_1x1_bias[] = {
    -0.082214, 0.073415, 0.486217, 0.087604, -0.085231, 0.038439, -0.450091, -0.008228,
    -0.031714, 0.096824, 0.222916, 0.199850, 0.615186, 0.265830, -0.353606, 0.635790,
    -0.289511, -0.209016, 0.397717, 0.432552, -0.315183, 0.062708, 0.583848, 0.634650,
    0.555900, -0.092218, 0.804001, 0.164179, 0.520531, 0.671972, -0.031490, -0.127366,
    0.114872, 1.303548, 0.250716, 0.397594, 0.559937, 0.655478, 0.549608, 0.295803,
    0.372027, 0.638234, 0.293459, 0.022919, 0.165332, 0.408391, 0.405552, 0.318515,
    -0.079841, 0.132061, -0.104933, 0.507206, -0.003170, 0.427740, 0.179675, 0.129378,
    0.189442, 0.232642, -0.155471, 0.396989, 0.043438, 0.700126, 0.888498, 0.197456,
    0.186124, -0.814423, 0.049818, 0.255218, 0.068611, 0.651087, 0.121102, 0.473006,
    0.473457, 0.625963, 0.045238, 0.245011, 0.521913, 0.463867, 0.759245, -0.184611,
    -0.200299, 0.544518, 0.111789, 0.770329, -0.081512, 0.781122, 0.180669, 0.150191,
    -0.285621, 0.526759, -0.310716, 0.433951, 0.477391, -0.390339, -0.193806, 0.377294,
    0.383870, 0.606279, 0.168235, -0.360785, 0.311917, 0.255595, -0.612279, 0.543435,
    0.642449, -0.185633, 0.567632, -0.039963, 0.262062, 0.312895, 0.618977, -0.138898,
    0.257150, 0.474318, -0.210010, -0.373335, 1.016141, 0.256031, 0.462872, 0.601538,
    0.743172, -0.035040, 0.248187, -0.156198, -0.552957, 0.285104, 0.086873, 0.550576,
    0.030503, -0.581008, -0.270300, 0.223492, 0.284034, 0.270474, 0.576624, 0.348230,
    0.076441, 0.512308, 0.231927, -0.296911, 1.351932, 0.374236, -0.026526, -0.341276,
    0.060983, 0.334476, -0.293376, 0.070081, 0.123560, 0.494948, 0.039484, 0.629465,
    0.520884, 0.679677, 0.733348, 0.616928, 0.318816, 0.388149, 0.441033, 0.218432,
    0.003209, 0.540614, 0.052217, -0.096875, 0.426005, -0.407300, 0.487619, 0.362404,
    -0.133966, -0.114033, 0.566599, -0.102868, 0.578628, 0.283031, 0.320870, 0.550587,
    -0.132825, -0.428681, 0.478046, 0.035855, -0.186573, 0.125250, -0.294435, 0.794220,
    0.600145, 0.356482, -0.148106, -0.275172, 0.008760, 0.323104, 0.799970, 0.557989,
    -0.024104, 0.053720, 0.569475, 0.277351, 0.693362, 0.560867, 0.016228, 0.266538,
    0.525365, -0.183905, 0.083153, 0.221649, 0.042243, 0.379643, -0.107460, -0.126040,
    0.786981, 0.289285, 0.801288, -0.121993, 0.030290, 0.119451, 0.292073, 0.528469,
    0.262065, -0.268164, -0.228955, -0.083013, -0.198609, 0.782885, 0.335929, 0.682875,
    0.550264, 0.661982, -0.306274, 0.264655, 0.395871, -0.187179, -0.256567, 0.284081,
    0.003456, 0.069266, 0.596367, 0.428096, -0.372315, 0.019699, -0.785200, 0.587119,
    0.210914, 0.223721, 0.521276, 0.450868, 0.739849, 0.690564, 0.923520, 0.698982,
    0.242901, 1.051472, 0.742515, 0.511038, 0.462907, 0.578846, 0.341511, 0.302363,
};

kernel void inception_4e_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[256];
    for (int f = 0; f < 256; f++) {
        sum[f] = inception_4e_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 528; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 256; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 256;
            }
        }
    }

    for (int f = 0; f < 256; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4e_3x3_reduce_bias[] = {
    0.039643, 0.398526, -4.030876, -0.246023, -1.616199, 1.253533, 0.636691, 1.687100,
    1.637064, 0.762514, 0.283881, 0.025687, 0.613097, 0.414965, 1.219873, 0.774339,
    0.589039, 1.924032, 1.060041, -6.872990, 1.060818, 0.116299, 1.507169, 1.752306,
    1.494286, 3.049045, 1.156090, -4.118595, 0.722727, 1.401579, 0.372112, 0.011569,
    -0.266906, -0.603980, 0.417315, 0.479653, 0.083447, -0.121381, -5.045333, 1.228810,
    0.724881, 1.271204, 1.118571, 2.450830, 1.507329, -0.026173, 0.855316, 0.076204,
    1.923819, -2.030229, -4.873106, 0.391895, -1.017131, 1.661804, 0.951038, 1.806645,
    -1.051238, -1.229134, 0.917686, 0.286210, 1.044430, 1.376562, 1.346457, 0.887845,
    1.052991, 1.473332, 0.682699, 1.207230, 1.301739, 0.022216, 1.300600, -0.026259,
    0.722937, -0.349927, 1.465480, 1.580469, 1.367625, 0.908284, 0.468461, -0.164793,
    1.864626, 0.878382, -0.904832, 0.993380, 0.734052, 0.945072, 1.081739, 1.789932,
    -2.847649, -0.360654, 0.629945, 1.664059, 0.105938, 0.705176, -0.886145, 0.678740,
    2.242966, 1.021716, -0.292543, 0.973231, 1.112031, 0.913927, 0.641771, 0.897974,
    0.887918, 0.321571, 0.690249, 1.751930, 0.610614, -2.253550, 0.647777, 0.603754,
    -3.040343, 0.421236, 1.262093, -1.460925, 1.394986, 1.533005, 0.731556, 3.124328,
    -0.328081, 0.448142, 1.345427, 1.461688, -0.781044, 2.922191, 0.698820, 2.813496,
    -0.935157, 0.010126, 1.757564, 0.462494, 1.047355, 0.563742, 2.511313, -15.751751,
    -3.509091, -0.160064, 1.366659, 1.096807, 0.066773, 1.220035, 0.328368, -0.018346,
    1.865225, 1.519652, 0.300299, 1.674391, 2.158783, 1.160421, -0.154082, -2.397900,
    -0.429089, 0.615079, 1.774812, -3.317134, 1.242851, 1.692300, 0.061790, -0.753016,
};

kernel void inception_4e_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[160];
    for (int f = 0; f < 160; f++) {
        sum[f] = inception_4e_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 528; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 160; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 160;
            }
        }
    }

    for (int f = 0; f < 160; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4e_3x3_bias[] = {
    1.786836, 0.607006, 0.080615, 0.448737, 0.619766, 0.555888, 1.008258, 0.606622,
    0.899637, 0.980626, 0.659458, 0.499371, 0.298965, 1.729579, 0.513912, 0.776237,
    0.416379, 0.716798, 1.253041, 0.771126, 0.638038, 0.877267, 0.453243, 0.495962,
    0.828417, 0.745443, 1.084139, 0.431071, -0.024313, 1.425223, 0.452815, 0.579854,
    0.546652, 1.125569, 0.944007, 0.723627, 0.728998, 1.350061, 0.639120, 0.634281,
    0.410372, 0.783235, 1.548498, 0.720592, 1.299903, 1.185829, 1.161939, 0.222362,
    0.821182, 0.426687, 0.744029, 0.436030, 0.675185, 0.664744, 0.843745, 0.525076,
    0.453328, 0.595091, 0.721245, -11.951663, 0.533862, 0.982702, 0.772784, 1.078736,
    0.582019, 0.787598, 1.133904, 1.541113, 0.897902, 0.505697, 0.875419, 0.458918,
    0.638956, 0.456745, 0.582556, 0.372358, 0.694369, 1.051062, 0.699746, 0.744173,
    0.335520, 1.043893, 1.045412, 0.831499, 0.817929, 1.752613, 0.757806, 0.969997,
    0.691905, 0.922864, 0.446142, 0.670691, 0.799811, 0.408648, -0.067799, 0.544325,
    0.880283, 0.653383, 0.122367, 1.518474, 1.246832, 0.310830, 0.476085, 0.797635,
    0.293915, 0.627447, 0.529104, 0.945393, 0.573291, 0.822943, 0.514391, 0.481809,
    1.099860, 1.218053, 1.277957, 1.005295, 0.905504, 0.566769, 1.214895, 1.223566,
    0.723077, 0.537336, 0.992437, 1.323953, 0.738378, 0.614076, 0.637042, -0.160703,
    0.310489, 1.121377, 1.320496, 0.669413, 0.537891, 0.994813, 0.941721, 0.471192,
    0.628423, 0.802893, 0.220261, 0.627637, 1.042283, 0.271214, 0.545718, 0.356459,
    0.118947, 0.512611, 1.212124, 0.429522, 1.032165, 0.535683, 0.703329, 0.545339,
    1.065303, 1.317492, 1.072684, 1.129839, 0.508612, 0.994238, 0.588082, 1.001177,
    0.308711, 0.752539, 1.203387, 1.143082, 1.106380, 0.542615, 0.452081, 0.626005,
    1.663621, 0.522891, 0.781010, 0.579969, 0.988933, 0.453970, 0.508395, 0.660471,
    0.331011, 0.703501, 0.812981, 1.004717, 0.506359, 0.712528, 0.526359, 0.419836,
    0.637704, 0.764494, 1.047078, 0.620806, 0.604659, 1.230677, 0.616403, 0.394712,
    1.095523, 1.583202, 0.152051, 0.932029, 1.180795, 0.267532, 0.809648, -0.118511,
    0.651512, 0.462377, 0.650474, 0.801574, 0.518599, 0.899222, 0.597709, 0.669289,
    -0.309887, 0.768831, 0.349654, 0.731574, 1.354383, 0.912327, 0.822010, 0.684210,
    1.027166, 0.593951, 0.366655, 0.147374, 0.298900, 0.868069, 0.665430, 0.685961,
    0.230440, 0.866800, 0.454894, 0.588164, 1.168105, 0.454429, 0.666856, 0.157108,
    1.877398, 0.786466, 0.883508, 0.876898, 0.667024, 0.617999, 0.806094, 0.672857,
    1.275174, 1.901820, 1.000385, 1.280866, 1.021131, 1.198748, 1.057374, 0.861556,
    0.646127, 0.871907, 1.199046, 0.174610, 0.895013, 0.449243, 0.290655, 0.926283,
    0.639003, 0.272501, -0.200877, 1.357461, 0.597904, 0.031720, 0.610800, 1.092143,
    0.755613, 0.373444, 0.991512, 0.564508, 1.153886, 0.638200, 1.174630, 1.092058,
    1.195185, 0.814578, 0.540399, 0.822594, 0.882342, 1.048616, 0.018788, 1.000733,
    0.833567, 2.032143, 0.590331, 0.563897, 0.734953, 0.758554, 0.525914, 1.071908,
    0.727948, 1.009486, 0.571892, 0.782339, 0.244677, 0.713360, 0.491105, 0.953286,
    1.033225, 0.320917, 0.460620, 0.130146, 0.319283, 0.819868, 0.172060, 1.387791,
    0.451362, 0.398222, 0.398711, 0.416703, 0.792689, 0.743133, 0.675088, 0.893217,
    0.245321, 0.385266, 0.726972, 0.915123, 0.754393, 1.134701, 1.101188, 1.078628,
};

kernel void inception_4e_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[320];
    for (int f = 0; f < 320; f++) {
        sum[f] = inception_4e_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 160; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 320; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 320;
            }
        }
    }

    for (int f = 0; f < 320; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4e_5x5_reduce_bias[] = {
    3.239219, 1.064924, 3.147490, 0.328032, 0.941274, 1.112661, 1.973353, 1.294532,
    -2.168442, 1.526859, 0.765975, 0.222356, 4.435249, 1.705351, 2.295964, -0.417774,
    0.701434, 0.137836, 0.417732, 0.937901, 0.597537, 2.425161, 1.193672, 4.201530,
    1.232243, 2.041968, 1.473655, 0.974401, -1.638534, 3.869189, 1.108886, 1.182687,
};

kernel void inception_4e_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[32];
    for (int f = 0; f < 32; f++) {
        sum[f] = inception_4e_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 528; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 32; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 32;
            }
        }
    }

    for (int f = 0; f < 32; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_4e_5x5_bias[] = {
    -0.001385, 0.197123, 0.392924, 0.115567, 0.163749, -0.188619, 0.033723, -0.021001,
    0.309078, 0.437946, -0.074180, 0.685904, 0.270742, 0.709172, -0.031653, 0.506220,
    0.199515, -0.047820, -0.078628, 0.279383, 0.233508, -0.206527, -0.066213, -0.155086,
    0.176165, 0.429379, -0.006761, 0.289567, 0.262552, 0.140083, 0.229703, 0.098887,
    0.129544, 0.888524, -0.146165, -0.101514, 0.591586, -0.193087, 0.344104, -0.233772,
    -0.169731, -0.041277, -0.026593, 0.116034, -0.484619, 0.248338, -0.431440, 0.321453,
    0.115203, 0.141705, 0.252750, 0.325011, 0.484529, 0.679854, 0.274254, 0.267917,
    -0.035190, -0.267043, -0.207907, 0.377555, 0.099889, -0.593464, -0.109372, 0.704184,
    0.136327, -0.642580, 0.435534, 0.297728, 0.182281, 0.468587, 1.345498, 0.324683,
    0.940826, 0.369774, -0.016795, 0.849195, -0.071828, 0.840829, -0.165924, 0.083032,
    0.134051, 0.512150, 0.066750, 0.308309, 0.341372, -0.342854, 0.376939, 0.184976,
    0.021061, 0.292885, 0.008956, 0.752620, -0.280706, 0.164580, 0.535901, 0.014635,
    0.068615, 0.502741, 0.124343, 0.488548, 0.206308, -0.363463, 0.709580, -0.150444,
    0.291943, -0.093489, -0.145388, 0.202581, 0.027888, 0.177577, 1.147515, 0.249043,
    0.245057, 0.403018, -0.222861, 0.822340, -0.056376, -0.199224, -0.733457, 0.621271,
    0.205390, 0.280019, 0.350702, 0.761033, 0.453415, 0.090908, 0.249624, 0.027591,
};

kernel void inception_4e_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_4e_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 32; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_4e_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_4e_pool_proj_bias[] = {
    0.267239, 0.399852, -0.256328, 1.488872, 0.136194, -0.417412, -0.430337, 0.111433,
    0.197740, 0.227945, 0.102139, 0.333480, 0.510868, 0.028778, -0.232694, -0.365470,
    -0.034897, 0.385721, 1.513780, 0.221273, 0.091506, -0.698911, -0.185796, 1.080314,
    -0.124176, 0.112444, -0.177877, 0.965744, 0.139487, -0.474488, 0.087276, 0.744752,
    -0.379455, 0.178218, 0.136366, 0.987856, 0.335431, -0.034745, 0.372660, 0.014117,
    0.132595, 0.576759, 0.238991, 0.457673, -0.419336, 0.087196, -0.423750, 0.271270,
    0.668222, 0.090184, 0.984956, 0.300354, 0.343134, 0.323049, -0.062579, -0.138980,
    0.049989, 0.416215, 0.272175, 0.613301, 0.325232, 1.310122, 0.067077, 1.386274,
    -0.178581, 0.195983, -0.290713, 0.089576, 0.206221, 1.119023, -0.388417, -0.052105,
    0.609482, 0.231684, 0.495018, 0.086719, -0.116869, 0.119879, 0.692015, 0.313980,
    -0.115291, -0.568233, -0.063193, 0.305150, 0.151185, 1.163833, -0.564540, 0.327243,
    -0.259090, 0.337529, 0.177941, -0.235695, -0.093436, 0.801901, 0.956732, 0.579468,
    -0.384214, -0.460221, -0.058776, 0.554824, -0.236886, 0.228125, 0.491208, -0.218881,
    -0.056130, -0.143118, -0.294563, -0.438727, 0.651287, -0.212361, -0.326103, 1.194364,
    -0.578820, -0.281481, 0.255109, -0.048590, -0.200654, -0.307707, 0.197484, 0.515068,
    0.252396, -0.283171, 0.329337, 0.144652, -0.521677, 0.072517, 0.148651, 0.280281,
};

kernel void inception_4e_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_4e_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 528; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 14 && x+dx >= 0 && x+dx < 14) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void pool4_3x3_s2_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 2;
    const int strideW = 2;
    const int padH = 0;
    const int padW = 0;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_5a_1x1_bias[] = {
    0.052877, -0.235354, 0.393708, 0.375685, -0.083300, -0.221029, 0.262642, -0.365413,
    0.090515, 0.684783, 0.356761, -0.327702, 0.691976, -0.942632, 0.290625, 0.553472,
    0.070178, -0.097247, 0.082180, 0.066787, -0.147217, 0.189248, 0.421196, -0.347826,
    -0.078414, 0.278352, 0.295674, -0.326635, 0.068849, -0.183186, 0.246848, 0.143219,
    -0.172145, 0.634132, 0.189840, 0.752333, 0.310472, 0.116825, -0.657070, 0.274229,
    0.535799, -0.035121, 0.244264, -0.254526, 0.238800, 0.062234, 0.230472, -0.407382,
    -0.123391, 0.468240, 0.420087, 0.175365, -0.537468, -0.506629, 0.124753, 0.390834,
    0.088492, 0.672451, 0.058563, -0.449544, -0.251689, 0.250178, 0.235080, 0.496538,
    0.244500, 0.094991, 0.625383, 0.273171, -0.179425, 0.390752, -0.047548, 0.192386,
    0.055434, 0.064343, 0.147526, -0.503656, 0.638272, 0.285778, 0.097239, 0.679413,
    0.615863, 0.233093, -0.074616, 0.487920, -0.394217, -0.940004, 0.682357, -0.614217,
    1.054808, -0.199816, -0.497708, 0.620002, 0.266499, 0.426328, -0.031751, 0.220516,
    0.983504, 0.568379, 0.260231, -0.389460, 0.128695, 0.137520, -0.274831, 0.164034,
    0.035387, 0.047653, 0.877748, 0.693205, 0.377056, 0.088020, -0.061063, 0.251950,
    0.206497, 0.304621, 0.707132, -0.217516, 0.138639, 0.470425, 0.309445, -0.056357,
    0.009012, 0.271899, 0.359021, -0.172571, 0.114061, 0.856011, -0.193196, 0.498539,
    0.046956, 0.165685, 0.583144, -0.162037, 0.226167, -0.675115, 0.134233, -0.293723,
    0.763223, -0.421666, 0.106697, 0.282049, 0.615376, 0.119340, 0.556933, 0.019375,
    0.438118, 0.347344, 0.252271, 0.327856, -0.457454, -0.226181, 0.388231, 0.180342,
    0.113790, 0.526864, -0.486953, 0.125905, 0.239691, -0.083301, -0.131119, 0.263938,
    0.074283, 0.204449, -0.135015, -0.448508, -0.176494, -0.205253, -0.140052, 0.111632,
    0.351729, 0.419098, 0.381747, 0.889919, 0.402266, 0.368638, -0.196162, 0.613586,
    0.166189, -0.141206, 0.167899, -0.201520, 0.097638, 0.037526, 0.354700, 0.604817,
    -0.422937, 0.227427, -0.623514, 0.865184, -0.211750, 0.376544, -0.050343, -0.028956,
    0.032757, -0.197481, 0.281745, 1.280869, -0.656407, 0.532660, 0.942866, 0.262154,
    0.679726, 0.380651, 0.190354, -0.143623, 0.115247, 0.824247, -0.231851, 0.252972,
    -0.095857, -0.054461, 0.241041, -0.339519, 0.831494, 0.070757, 0.074615, 0.496362,
    -0.344433, -0.349290, 0.012988, 0.374798, -0.393872, 0.116879, 0.084714, -0.162568,
    0.319542, 0.165138, 0.160044, 0.405383, 0.370276, 0.322565, -0.582029, -0.183869,
    0.280967, 0.544539, 0.690179, -0.117198, -0.537848, -0.936826, 0.781560, 0.206615,
    0.281822, -0.532572, -0.903142, -0.244919, 0.112981, -0.008209, 0.738323, -0.022751,
    -0.068995, 0.253337, -0.100124, -0.153285, 0.249423, 0.246189, -0.425677, 0.600320,
};

kernel void inception_5a_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[256];
    for (int f = 0; f < 256; f++) {
        sum[f] = inception_5a_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 832; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 256; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 256;
            }
        }
    }

    for (int f = 0; f < 256; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5a_3x3_reduce_bias[] = {
    0.469920, -0.112850, 0.765376, -1.791047, 0.745173, 0.915741, 1.798959, -0.199103,
    0.040949, -2.225338, 0.432869, -0.185313, 1.287956, 0.429165, 0.877358, 0.284077,
    0.684395, 0.707494, 0.575657, 1.189166, -0.120978, 0.087881, -2.456398, -2.266401,
    0.387473, 0.047508, 0.464609, 0.262696, -0.261652, 0.314511, 0.795907, 0.674131,
    0.864344, 0.907638, 0.230218, 0.475018, 1.012219, -0.219283, 0.820699, 0.423438,
    -0.105447, 0.743573, 0.915973, 0.820584, -1.130410, 1.124994, 0.317409, -1.392641,
    0.702595, 0.768032, 0.148742, -1.204661, -1.080268, 0.738154, -0.859913, -3.914418,
    0.642709, 1.344505, 0.076905, -2.078688, 0.973935, -0.280722, -0.322887, -0.552403,
    0.457218, 0.105280, -1.040278, -0.251017, -0.601268, 0.943350, 0.465865, 0.659989,
    -0.401777, 0.088739, -0.177687, -0.285563, -0.093410, 1.354485, 0.840667, -0.079113,
    -0.745081, -0.551342, 0.603386, 0.109839, 0.893421, 0.383774, -1.788446, 0.050028,
    -0.699136, 1.507107, 1.220728, 1.002226, 0.570596, 1.211798, 0.456298, 0.966606,
    -1.126802, 0.427199, 0.892887, -0.861734, -0.751230, 0.659340, 0.929918, 0.584583,
    0.345762, 1.766295, 0.276241, 1.171234, -0.167613, 0.133835, 0.523498, 0.378394,
    0.345813, 1.510755, 0.527524, 0.650443, -2.589817, 1.225953, -1.228848, 0.853915,
    0.197987, 0.777330, 1.175770, -0.238899, 0.593042, -0.431797, 0.635251, -0.602224,
    0.998496, -0.609448, 0.490065, -0.162416, -0.571673, -0.293305, -1.300543, -0.800068,
    0.608712, 0.465919, 0.566785, -0.905600, 0.781644, -2.204655, 0.548698, -0.451172,
    -1.409820, 0.749261, 1.080343, 0.990806, 1.485366, 1.054230, 0.592866, 0.831652,
    -0.259441, -0.651464, 0.448182, -1.453226, -0.254772, 0.411210, -0.795461, -0.017680,
};

kernel void inception_5a_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[160];
    for (int f = 0; f < 160; f++) {
        sum[f] = inception_5a_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 832; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 160; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 160;
            }
        }
    }

    for (int f = 0; f < 160; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5a_3x3_bias[] = {
    0.270997, 0.818566, 0.471702, 0.672682, 0.952069, 0.770653, 0.775424, 0.615402,
    0.997899, 1.763077, 0.572254, -0.498141, 0.530847, 0.131395, 0.868754, 0.518152,
    0.667526, 0.697485, 0.556840, 0.623536, 0.281242, 0.447079, 0.604949, 0.339698,
    0.713902, 1.545896, 0.703020, 0.558874, 0.321290, 0.073860, 0.007143, 1.288445,
    0.774390, -0.338597, 1.476848, 0.600647, 0.367049, 0.447098, 0.890625, -0.247935,
    0.494535, -2.579815, 0.526070, 0.420886, 0.508091, 0.100566, 0.875163, 0.516006,
    0.966956, -0.108280, 0.107513, 0.686932, 0.950553, 1.036949, 0.505984, -0.007905,
    0.922819, 0.877486, 0.782644, 0.298748, 0.655265, 0.731027, 0.645048, 0.730717,
    0.360137, 0.466175, 0.738524, 0.504938, 0.325978, 0.659147, 0.392451, 1.408642,
    0.606427, 0.561929, -0.866578, 1.106586, 0.882452, 0.929524, 0.393670, 0.298937,
    1.101628, 0.588252, 0.707464, -0.213102, 0.191438, 0.464542, 1.007773, 0.632219,
    0.517444, 0.674886, 0.432549, 0.244921, 0.240404, 1.328853, 1.397924, 0.194703,
    0.513400, 0.576105, 0.815384, 0.498600, 0.399515, 1.195651, 0.682869, 0.465154,
    0.739498, 0.515042, 0.200905, 0.963640, 0.461472, 0.423517, 0.465942, 0.072688,
    0.221320, 0.306972, 0.643120, 0.715769, 1.911244, 0.550753, 0.697538, 0.122503,
    0.298845, 1.399089, 0.851167, 0.971309, 0.359927, 0.565241, 0.195876, 1.011595,
    1.041416, 1.217818, 0.242827, 0.910932, 0.866961, 0.585815, 0.718988, 0.845255,
    0.181185, 0.612978, 0.882991, 0.582259, 0.748652, 0.197851, 0.974121, 0.485218,
    -5.516464, 0.650806, 0.442387, 0.074845, 0.251065, 0.672383, 0.093364, 0.515437,
    0.575514, 0.930844, 0.619844, 0.677730, -0.086911, 0.525437, 0.523415, -0.067198,
    0.609601, 0.511801, 0.546306, 0.593172, 0.109431, 0.733894, 0.286421, -0.001113,
    1.433545, 1.399226, 0.775188, -3.313805, 0.610573, -2.130171, 0.873763, 0.523455,
    0.490993, 0.299694, -0.901107, -1.030111, 0.903988, -9.162471, 0.629591, 0.731360,
    1.098655, 0.775867, 0.758161, 0.893174, 0.609620, 0.266777, 0.606963, 0.117110,
    0.821241, 0.717588, 0.478812, 0.442682, 0.879843, 0.351099, 0.695123, 0.605050,
    0.552684, 0.974456, 0.037352, 1.423087, 0.202458, 0.080126, -0.142256, 0.341252,
    0.173262, 0.848357, -2.055114, 0.213545, 0.677838, 0.443989, 0.921302, -1.149498,
    0.428055, 1.051608, 1.590151, 0.031949, 0.119163, 1.003418, 0.604323, 0.972981,
    0.635296, 0.730744, 0.445387, 0.061077, 0.669928, 1.241057, 0.459047, 0.562101,
    0.168066, 0.380399, 0.054691, 1.159580, 0.715785, -0.203204, 0.641396, 1.439084,
    0.816740, 0.282582, 0.281977, -2.137073, 0.883667, 0.382425, -0.007364, -0.132335,
    0.408280, 0.462436, 0.637131, 0.286680, 0.386877, 0.879082, -0.265539, 0.616706,
    -0.539206, 0.320312, 1.062241, 0.547990, 0.522178, 0.670871, 1.042222, -1.035387,
    0.711952, 0.742327, 1.012148, 0.455306, -0.142405, 0.401237, -0.410100, 0.644303,
    -0.054296, 0.982953, -0.069923, 0.229135, 0.692290, 0.035221, 0.597350, 0.221394,
    0.512588, 0.878183, 0.246082, 0.291495, -0.228223, 0.314348, 1.119914, 0.587018,
    1.553649, 0.938294, 0.751720, 0.342775, 0.050234, 0.801469, 0.721055, 0.753188,
    0.445640, 0.741498, 0.001127, 1.276020, 0.331634, 1.239567, 0.096493, 0.430013,
    0.998959, 0.696089, 0.745492, 0.579400, 0.975481, -0.609893, 0.925403, 0.740747,
    0.212416, 0.680929, 0.590900, 1.205018, 1.167317, 0.871681, 0.344643, 0.734481,
};

kernel void inception_5a_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[320];
    for (int f = 0; f < 320; f++) {
        sum[f] = inception_5a_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 160; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 320; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 320;
            }
        }
    }

    for (int f = 0; f < 320; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5a_5x5_reduce_bias[] = {
    -1.541243, 0.875177, -0.508092, 8.003083, 0.945765, -0.390146, 0.668041, 0.318639,
    1.337915, 2.044335, 3.883706, -0.648794, 1.566574, -1.766980, -0.607547, 0.099515,
    3.830976, 2.018108, 2.507716, 2.576811, 4.656754, -3.260894, -0.495644, -2.936241,
    0.417288, 2.609662, 0.054835, 5.456886, 1.858658, 3.118584, -1.149704, 0.813719,
};

kernel void inception_5a_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[32];
    for (int f = 0; f < 32; f++) {
        sum[f] = inception_5a_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 832; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 32; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 32;
            }
        }
    }

    for (int f = 0; f < 32; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5a_5x5_bias[] = {
    -0.296927, 0.031364, 0.158297, 0.710438, -0.311502, -0.204517, -0.596157, -0.411786,
    0.183185, 0.597198, 0.108718, 0.154927, 0.090721, -0.126782, 0.482170, -0.634764,
    0.106730, 0.403310, 1.930142, 0.992869, -0.010832, 0.219758, 0.014778, -0.324037,
    1.474635, 0.879069, 0.533876, 0.330796, -1.182101, -0.241682, -1.343221, 0.209860,
    0.729010, -0.277639, -0.445403, 0.164558, 0.591604, 0.314771, -0.763641, 0.607813,
    -0.170471, 0.265901, -0.216951, 0.989899, -0.297566, -0.067486, 0.129927, -0.336835,
    -1.004907, -0.489227, -0.071008, -1.946177, -1.629443, 0.398652, 0.236740, -0.319056,
    -1.082716, -0.174254, 0.145152, 1.036919, -0.047501, -0.000601, 0.227641, 0.503672,
    0.237165, -0.272202, 0.761837, 0.755022, -0.245126, 0.906507, 0.672768, 0.026103,
    0.720736, 0.719836, -0.254839, -1.188601, 0.642383, -0.215734, -0.535517, -0.287118,
    0.151541, -1.316804, 1.655232, -0.093595, 0.163967, 0.456872, 0.216277, 0.666334,
    -0.485419, -0.427309, -0.109995, 1.258893, 0.270272, 0.001014, 0.073518, 0.572987,
    0.410769, -1.705604, 0.916845, 0.345380, 0.258927, 0.505124, -2.399608, -1.370248,
    0.960868, -0.537885, 0.569097, 0.104815, -0.052638, 0.391438, 0.691806, 0.139622,
    -3.206934, 0.413743, 1.067341, 0.624883, -2.337016, -0.256682, -0.022689, -0.804094,
    -1.699604, -0.391107, -0.094343, -0.022599, 1.025634, 0.613456, 0.673105, -0.837159,
};

kernel void inception_5a_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_5a_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 32; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_5a_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_5a_pool_proj_bias[] = {
    -0.093667, 0.006494, -1.424736, -0.194855, -0.535310, 0.472974, 1.405118, 0.851779,
    0.984947, -0.052747, -0.064362, 0.744488, -0.518563, -1.161266, 0.009471, -0.233211,
    0.544687, 0.374460, 0.003792, 0.586799, 0.653007, 0.542323, -0.826684, -1.912395,
    -0.241492, -0.003919, -0.489407, 0.606331, -0.942007, -1.425425, 0.206835, 0.619017,
    -0.295689, 0.568493, -0.487246, -1.597000, 1.296173, -0.793433, 0.464626, 1.397833,
    0.652747, 0.838344, -1.127965, 0.103889, 1.388794, 0.257810, -0.178234, -0.556149,
    -0.000770, -0.027172, 0.637451, -0.057468, 0.471453, -0.789142, 1.047860, -0.213393,
    -0.018588, 0.665573, -0.167189, -0.269531, 0.236077, 0.239521, -0.021239, 0.542675,
    -0.450868, 0.695825, -1.125738, 0.861326, 0.159851, -0.806318, -1.159965, 0.081740,
    0.242671, -0.207321, -0.803174, 0.154146, 0.510837, -0.119309, 0.501868, -0.624931,
    -2.257494, -1.198235, 1.596516, 0.331312, -0.671007, -0.641019, -0.202987, 0.680110,
    -0.509876, -0.039387, -0.462102, 0.474452, 1.006984, 0.677524, -0.498119, 0.596728,
    0.466339, 0.711247, 0.179082, -0.054790, 1.280393, -0.346447, -0.545653, 0.779602,
    1.022897, 0.435969, -0.823473, 0.613336, -0.515449, 0.278779, -0.177327, 0.672456,
    0.419477, 1.276806, 1.025910, -0.516605, 0.048376, 0.620625, -0.218619, 0.407381,
    1.128336, 0.306408, -0.416330, 0.658764, 1.010634, 0.380560, -0.409817, 1.306669,
};

kernel void inception_5a_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_5a_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 832; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5b_1x1_bias[] = {
    0.788862, 1.024081, 0.755478, 0.498421, 0.652406, 0.883050, 0.730538, 0.832601,
    0.555653, 0.156133, 0.896769, 0.777289, 0.428814, 0.502820, 1.452720, 0.898161,
    0.401327, 1.177887, 0.994138, 0.707925, 0.741563, 0.776829, 1.005454, 0.807781,
    0.891103, 0.677656, 0.559689, 1.196226, 0.580442, 0.902608, 0.734204, 1.150692,
    0.842374, 0.834127, 0.362664, 0.020851, 0.952138, 0.403225, 0.518466, 0.278318,
    0.828310, 0.606788, 0.322979, 0.812836, 0.720639, 0.423058, 0.995597, 0.398472,
    0.892120, 0.639479, 0.880420, 0.293749, 0.965062, 1.269484, 0.304253, 0.548037,
    0.342492, 0.582063, 0.389667, 0.689088, 0.612668, 0.620979, 0.860976, 0.693965,
    0.502915, 0.974256, 0.846041, 0.408721, 0.179760, 0.880423, 0.839352, 0.640419,
    0.384058, 0.032947, 1.334164, 0.869573, 0.928358, 1.155331, 0.584322, 0.581817,
    0.940279, 0.499392, 0.912612, 1.128111, 0.087158, 0.532851, 0.717743, 0.965918,
    0.857965, 1.077500, 0.695920, 0.703777, 0.849889, 1.054987, 0.833790, 0.907433,
    0.324943, 0.633347, 1.185373, 0.694533, 0.547408, 0.819073, 0.978614, 0.600789,
    0.901866, 0.608673, 0.486712, 1.143341, 0.996231, 0.948430, 0.767239, 0.565547,
    0.683394, 0.307899, 1.005639, 0.937329, 0.379038, 0.558747, 0.536575, 0.929249,
    0.611285, 0.700816, 0.900150, 0.359970, 0.734262, 1.292357, 0.800617, 0.982020,
    0.539310, 0.904238, 0.837609, 0.785205, 0.699558, 0.591334, 0.662756, 1.088897,
    0.633429, 0.859780, 0.478293, 0.428408, 0.853936, 0.793001, 0.605145, 0.626256,
    0.810212, 0.985811, 0.922874, 0.899210, 1.494081, 0.688748, 0.732176, 0.692797,
    0.484368, 1.030728, 0.999012, 0.928743, 0.763697, 0.534786, 0.554867, 0.529830,
    0.483440, 0.638337, 0.677381, 0.554856, 0.416598, 0.445153, 1.341413, 0.816113,
    0.624443, 0.460159, 1.036318, 0.584871, 0.486627, 0.439289, 0.884047, 0.959301,
    0.306148, 0.899698, 0.738939, 0.606555, 0.050964, 0.794597, 0.672497, 0.370526,
    0.601419, 0.924322, 0.158159, 0.940166, 0.843025, 1.015094, 0.431435, 0.268357,
    0.886836, 0.448323, 0.620637, 0.810860, 0.706612, 0.824544, 0.810451, 0.636606,
    0.773673, 0.491637, 0.220299, 0.738383, 0.879336, 1.131335, 0.417625, 0.697699,
    0.907894, 0.983998, 0.711045, 0.594834, 0.799663, 0.930390, 0.714930, 0.787879,
    0.458467, 0.717714, 0.958199, 0.644757, 0.798416, 0.876541, 0.848685, 0.764905,
    1.256772, 0.350837, 1.276673, 0.680181, 0.510572, 1.096191, 0.455263, 0.771224,
    1.294234, 0.704389, 0.546103, 0.690076, 0.845079, 0.874568, 1.042882, 0.816334,
    0.664997, 0.629631, 1.088793, 0.695622, 0.834970, 0.839131, 0.921862, 0.967470,
    1.007359, 1.032496, 0.390834, 0.271273, 0.510153, 1.025420, 0.721252, 0.650083,
    0.775290, 1.434105, 0.907287, 0.842719, 0.171799, 1.161333, 0.556523, 0.601114,
    0.508109, 0.541089, 0.951407, 0.855261, 0.625663, 0.610357, 0.418527, 0.583494,
    0.595831, 0.564225, 0.571458, 0.984026, 0.707351, 0.672961, 1.102185, 0.972307,
    0.871465, 0.777180, 0.447661, 0.448193, 0.847414, 0.857325, 0.566464, 0.807919,
    0.828324, 0.731248, 0.400546, 0.973834, 0.914067, 0.703380, 1.292544, 0.954589,
    0.971153, 0.414975, 0.742886, 0.745862, 0.985238, 0.669171, 1.055597, 0.531557,
    0.985697, 0.615702, 0.778523, 0.610631, 0.876004, 0.476899, 0.502734, 0.952925,
    0.791946, 0.810062, 1.123751, 0.831847, 0.649703, 1.124751, 0.579857, 1.035260,
    0.603723, 1.021536, 0.413812, 0.191363, 0.280556, 0.808547, 0.729854, 0.991988,
    0.325958, 0.207080, 0.575412, 0.377753, 0.394631, 1.049801, 0.946730, 1.111032,
    0.852124, 1.046078, 0.868651, 1.249723, 1.135063, 0.481787, 0.763332, 0.778051,
    0.631387, 0.678469, 0.705291, 0.624394, 0.650605, 1.177939, 0.711604, 0.859270,
    0.999176, 0.784738, 0.473881, 0.497223, 0.743456, 0.553471, 0.541045, 0.829610,
    0.570744, 0.874228, 0.704071, 1.062733, 0.765607, 0.514997, 0.956117, 0.947216,
    1.357317, 0.258865, 0.442116, 0.804243, 0.882722, 0.835767, 0.746603, 0.589970,
    0.287372, 0.364241, 0.068404, 0.566531, 1.407816, 0.396629, 0.607847, 0.950931,
};

kernel void inception_5b_1x1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[384];
    for (int f = 0; f < 384; f++) {
        sum[f] = inception_5b_1x1_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 832; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 384; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 384;
            }
        }
    }

    for (int f = 0; f < 384; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5b_3x3_reduce_bias[] = {
    0.412481, 0.898025, 0.729611, -2.743529, 2.100342, 0.213944, 1.088009, -0.493395,
    -0.508668, 1.008666, -1.302435, 0.060698, -0.430218, 1.301922, -4.100399, 0.400995,
    -0.297820, 1.306124, -1.120253, 0.568878, 1.579929, 0.121057, 1.766505, 0.367990,
    0.338984, -0.219203, 1.061204, -1.030382, -0.189322, 0.952893, 1.522553, 0.687372,
    1.105365, 1.018102, 0.211365, -0.890327, -0.494940, -0.090381, 0.256613, 0.133406,
    1.646634, 1.771245, 2.421208, 0.228406, 1.241647, 0.712333, 1.271967, -0.077843,
    -0.013967, 0.529668, -0.538336, 1.943798, 0.852365, -0.292280, 1.181068, -0.067641,
    0.566726, 1.359450, 0.018201, 0.872432, -1.972024, 1.927414, -0.543936, 0.307579,
    -0.241993, -0.309256, 0.575466, 0.404684, 0.645317, 1.324186, -5.204278, -0.886636,
    0.596798, -0.025826, 1.237289, -0.147734, 0.527788, 0.576895, -0.403545, -2.130901,
    1.514553, -1.783806, 1.370160, -0.015544, 0.257919, -0.122567, -13.313322, 0.011718,
    -0.198780, 1.027859, 0.772258, 0.513267, 0.649031, 0.183786, -0.435432, 1.034918,
    0.310716, 0.354528, -0.413525, 0.327518, -1.608386, 1.059954, -1.193583, 0.347137,
    0.833351, -2.532184, 0.529866, 0.134742, -0.941022, -0.945256, -0.704595, 0.659013,
    -0.160219, -0.325647, 0.901758, 0.510951, -1.827018, 0.172144, 0.268037, 0.262518,
    2.356714, 0.206737, -0.351811, 0.637686, -0.637281, -0.129487, 0.430899, 0.748557,
    0.755236, 3.558767, 1.197083, 0.862175, 0.845959, 0.545131, 0.552592, 0.412806,
    -1.147625, -0.045188, 0.428631, -0.199996, 0.708195, 0.769048, 0.741060, 0.658695,
    0.319750, 0.516393, 1.513215, -0.605770, -0.452763, 0.910667, 1.263974, 1.148671,
    0.889670, -0.261422, 0.640334, 1.268458, 0.424672, -0.618985, -0.930258, -0.573420,
    -0.210002, 1.133711, 1.026685, 0.263592, 0.078220, 0.362433, 0.008592, 1.344532,
    -0.676321, 0.701368, -1.578332, -3.157346, 0.544271, 0.686560, 0.897397, 1.113382,
    0.156041, 1.219963, 0.884448, 0.727191, 1.137411, 0.976153, 0.438565, 0.070674,
    -0.710326, -0.046904, 0.321931, 1.107810, -3.057260, 0.778323, 0.269447, 0.330031,
};

kernel void inception_5b_3x3_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[192];
    for (int f = 0; f < 192; f++) {
        sum[f] = inception_5b_3x3_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 832; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 192; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 192;
            }
        }
    }

    for (int f = 0; f < 192; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5b_3x3_bias[] = {
    -0.065344, 0.876562, 0.810548, 0.676603, 0.589670, 0.140274, -0.185394, 0.631828,
    1.029024, 0.145546, 0.512758, 0.016257, 0.336617, 0.115408, 0.255621, 0.528517,
    0.657691, 0.208678, 0.967033, 0.995444, 0.348080, 1.008541, 0.545044, 0.307572,
    0.584494, -0.268437, 0.268089, -0.036224, -0.258017, 0.534840, 0.784164, 0.418773,
    0.186444, -0.064820, -0.085931, 0.463332, 0.338501, -0.068545, 0.390524, 0.897970,
    0.427423, 0.230723, 0.232380, 0.703714, 0.445841, 1.171883, 1.000282, 0.324458,
    0.892419, 0.184334, 0.251440, 0.606776, 0.352309, 0.324264, 0.000435, 0.146423,
    0.813345, 0.112670, 0.650724, 0.050151, 0.477305, 0.022843, 0.589991, 0.520608,
    0.402830, 0.456328, 0.057504, 0.658852, 0.304724, -0.059863, 0.391182, 0.217391,
    0.696060, 0.188459, 0.933660, 0.590796, 0.059167, 0.017370, -0.105443, -0.072624,
    0.185905, 0.396229, 0.156389, 0.046353, 0.293069, 0.862993, 0.850304, 0.362646,
    0.432431, 0.240927, 0.737828, 0.455617, 0.084244, 0.581530, -0.119328, -0.065765,
    0.923660, 0.329296, 0.508808, 0.830923, -0.021049, 0.467936, 0.576829, -0.158814,
    0.670620, 0.028650, -0.003142, 0.467577, 0.078408, 0.451590, 0.095495, 0.666344,
    0.233590, 1.179389, 0.908170, 0.437302, 1.343467, 0.069148, -0.038108, 0.444483,
    0.317126, 0.008750, 0.698579, -0.024192, 0.141223, 0.569020, 0.455464, 0.332929,
    0.318676, 0.467062, 0.119210, 0.987349, 0.638172, 0.885663, 0.150632, 0.325978,
    0.263237, 0.185410, -0.152026, -0.046426, 0.190342, 0.753638, 0.228085, 0.381942,
    0.088954, 0.649624, 0.316754, 0.223046, 0.224353, 0.296720, 0.315916, 1.490576,
    0.389934, 0.340075, -0.185017, 0.480350, -0.078584, -0.118332, 0.326160, -0.074494,
    1.091330, 0.278329, -0.026945, 0.276144, 0.547338, 0.330567, 0.130602, 0.211339,
    0.316142, 0.045232, -0.220244, 0.952317, 0.275190, 0.509959, -0.025983, -0.031005,
    0.659141, 0.667982, 0.824565, 0.334637, 0.119630, 0.153771, 0.054869, 0.373147,
    0.584735, 0.260371, 0.431543, 0.403923, 0.133670, 0.137581, -0.156407, 0.199643,
    0.399056, 0.100287, 0.373742, -0.084124, 0.597292, 0.024637, 0.259773, 0.324437,
    0.154300, 1.000365, 0.201063, 0.746184, 0.239274, 0.221276, 0.686876, 0.038268,
    0.901416, 0.367104, 0.021617, -0.001187, -0.102880, -0.246777, 0.444341, 0.302345,
    -0.007376, 0.624541, -0.035451, 0.377625, 0.669963, 0.501786, 0.897740, 0.588759,
    0.152283, 0.531480, 0.248236, 0.322894, 0.259642, 0.975164, 0.332447, 1.034286,
    0.559678, 0.214548, -0.056973, 0.042371, 0.023368, 0.430906, 0.349288, 0.168212,
    0.147557, 0.109236, 0.275741, 0.429328, -0.029102, 0.255006, 0.657552, -0.049815,
    0.377585, 0.750825, 0.609487, 1.026208, -0.088807, 0.116464, 0.340288, 0.749596,
    0.435764, 0.681276, 0.318347, 0.985401, 0.438838, 0.287827, 0.216597, 0.545503,
    0.320958, 0.316665, 0.416122, 0.350710, 0.717194, 0.631189, 0.068803, 0.378130,
    -0.067650, 0.616822, -0.061411, 0.322291, 0.337800, 0.479200, 0.287059, 0.329968,
    0.718071, 0.243710, 0.230069, -0.011657, 0.231979, 0.498411, 0.464734, 0.434610,
    0.267077, -0.070236, 0.589720, 0.246809, -0.008183, 0.422906, 0.306750, 0.229061,
    0.884462, 0.400881, 0.322268, 0.216408, 0.846968, 0.404250, 0.678655, 0.125453,
    0.669704, 0.120115, 0.341269, 0.982241, 0.925394, 0.033192, 0.913572, 0.456929,
    0.680203, -0.096626, -0.043054, 0.115513, 0.261883, 0.229929, 0.329628, 0.279276,
    0.414200, 0.157142, 0.702295, -0.184091, 0.107295, 0.233497, 0.517804, 0.303271,
    0.948072, 0.563863, -0.003851, 0.368593, 0.656110, -0.227202, 0.804897, 0.056985,
    0.548110, 0.600443, 0.357216, -0.170839, 0.338516, 0.114868, 0.525294, 0.274632,
    0.248051, 0.729243, 0.126052, 0.023535, 0.775535, 0.186214, 0.229439, 0.413180,
    0.730095, 0.255395, 0.285745, 0.014670, 0.413867, 0.248407, 0.286188, 0.215259,
    0.597996, 0.174783, 0.261386, 0.116761, 0.373448, 0.311473, 0.174176, 0.642297,
    0.175973, 0.257082, 0.445701, 0.277263, 0.240098, 0.015558, 0.057002, 0.315066,
    0.204439, 0.589452, -0.134533, 0.649156, 0.338906, 0.799088, 0.240468, 0.492276,
};

kernel void inception_5b_3x3_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[384];
    for (int f = 0; f < 384; f++) {
        sum[f] = inception_5b_3x3_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 192; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 384; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 384;
            }
        }
    }

    for (int f = 0; f < 384; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5b_5x5_reduce_bias[] = {
    -0.117797, 0.661428, 0.397568, 0.973152, 1.289570, 1.066936, 1.097006, 1.449225,
    1.679419, 1.584531, 0.706791, -2.004778, -3.827009, 0.658471, 0.899879, 0.528299,
    -0.287969, -1.901512, -0.802133, 0.696181, 0.799153, 1.150791, 0.029863, -2.154013,
    0.800739, 0.446959, -2.607085, 0.204182, 1.443914, -1.764248, 1.528379, 1.228268,
    1.600231, -0.077395, 1.820865, 0.084846, 2.413982, 1.102905, -0.099358, -0.229583,
    0.242128, -0.017957, 1.015241, 0.143491, 0.657904, -0.476074, -1.398365, -0.108081,
};

kernel void inception_5b_5x5_reduce_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[48];
    for (int f = 0; f < 48; f++) {
        sum[f] = inception_5b_5x5_reduce_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 832; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 48; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 48;
            }
        }
    }

    for (int f = 0; f < 48; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

constant half inception_5b_5x5_bias[] = {
    0.220851, 0.507202, 0.106063, 0.533439, -0.063495, 0.149310, 0.014977, -0.214101,
    0.431343, -0.088562, 0.122984, -0.088323, 0.263052, 0.501146, -0.519677, -0.090522,
    0.047844, 0.175974, 0.796581, -0.155206, 0.561426, 0.005527, 0.210590, 0.739507,
    0.463939, -0.048786, 0.205447, 1.010802, 0.094178, 0.647261, 0.816953, -0.367859,
    0.243876, -0.386838, 0.021179, 0.661685, 0.478521, -0.288221, -0.097065, 0.389946,
    -0.054907, 0.497222, -0.610992, 0.389898, 0.310205, 0.791220, 0.572738, 0.037413,
    0.119957, 0.599680, 1.015547, -0.072133, 0.152752, 0.131793, 0.762192, 0.295435,
    0.082133, 0.262311, 0.008568, 0.103274, 0.466865, 0.930654, 1.251638, -0.131280,
    0.201000, 0.103238, 0.225341, -0.188696, 0.861966, 0.091884, -0.109795, 0.361017,
    0.091408, 0.433341, -0.009440, 0.186100, 0.457769, -0.194401, -0.001726, 0.054256,
    0.952498, 0.535713, 1.253816, 0.439274, 0.059070, 0.087754, 0.051736, 0.330170,
    -0.261973, 1.175446, 1.132873, 0.645147, -0.015004, -0.247487, 0.199538, -0.371068,
    0.094503, -0.140278, -0.264074, 0.135870, 0.029926, -0.323708, -0.190637, 0.669513,
    0.314963, 0.013481, -0.183974, 0.959706, 0.032126, -0.039312, -0.276713, -0.492469,
    -0.388862, 0.343102, -0.029208, -0.154382, 0.035170, 0.342477, 0.657188, -0.280245,
    -0.494698, -0.063474, -0.097454, 0.441114, 1.035740, 0.131035, -0.530918, 0.353521,
};

kernel void inception_5b_5x5_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 5;
    const int kernelW = 5;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 2;
    const int padW = 2;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_5b_5x5_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 48; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void inception_5b_pool_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    max_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

constant half inception_5b_pool_proj_bias[] = {
    0.228590, 0.187272, 0.432800, 0.193124, 0.050081, -0.004243, 0.038796, 0.345775,
    0.152490, 0.373669, 0.298211, -0.065889, -0.046939, -0.041763, -0.031699, 0.589447,
    0.409318, 0.073965, 0.014303, -0.033133, 0.221461, 0.465710, -0.127827, 0.516529,
    0.225190, 0.603613, 0.048272, 0.240618, 0.268910, 0.310986, 0.086751, 0.019077,
    -0.136128, 0.167648, -0.263520, -0.276859, 0.355304, 0.016630, 0.171980, 0.126865,
    0.419028, -0.042068, 0.392206, 0.584529, 0.155860, 0.296262, 0.286288, 0.245520,
    0.172682, 0.194486, 0.270440, 0.023857, -0.058953, 0.264981, 0.409549, 0.396922,
    0.381202, -0.219635, 0.157746, 0.102057, 0.624004, 0.396912, 0.402874, 0.061040,
    -0.156500, 0.219322, 0.205493, 0.333901, -0.199313, 0.111009, 0.192381, 0.179997,
    0.005600, 0.203042, -0.012842, 0.135517, 0.155417, 0.491292, -0.002127, 0.153979,
    -0.304868, -0.002144, -0.020928, 0.135256, -0.183120, -0.118747, 0.385401, 0.193268,
    0.341017, 1.204555, 0.007097, 0.138357, 0.084773, 0.630239, -0.037148, 0.146033,
    0.207877, -0.059923, 0.262197, 0.095036, 0.211425, 0.144395, 0.427408, 0.368363,
    -0.020745, 0.017841, 0.282242, 0.020367, 0.013640, -0.028902, 0.105232, 0.025195,
    0.114921, 0.118152, 0.077991, -0.083146, -0.405419, 0.101897, 0.095883, 0.512106,
    0.268958, 0.256753, 0.266008, 0.421061, 0.654833, 0.260519, -0.072208, -0.210664,
};

kernel void inception_5b_pool_proj_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         device half* weights [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) { return; }
    const int kernelH = 1;
    const int kernelW = 1;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    const int y = -padH + kernelH/2 + int(gid.y)*strideH;
    const int x = -padW + kernelW/2 + int(gid.x)*strideW;
    float sum[128];
    for (int f = 0; f < 128; f++) {
        sum[f] = inception_5b_pool_proj_bias[f];
    }
    int i = 0;
    for (int fc = 0; fc < 832; fc++) {
        for (int dy = -kernelH/2; dy <= kernelH/2; dy++) {
            for (int dx = -kernelW/2; dx <= kernelW/2; dx++) {
                if (y+dy >= 0 && y+dy < 7 && x+dx >= 0 && x+dx < 7) {
                    half v = in.read(uint2(x + dx, y + dy), fc)[0];
                    for (int f = 0; f < 128; f++) {
                        half w = weights[i];
                        sum[f] += w * v;
                        i++;
                    }
                    continue;
                }
                i += 128;
            }
        }
    }

    for (int f = 0; f < 128; f++) {
        // Pair with a ReLU layer that goes next.
        half v = max(sum[f], 0.0f);
        out.write(v, gid, f);
    }
}

kernel void pool5_7x7_s1_0(texture2d_array<half, access::read> in [[texture(0)]],
                         texture2d_array<half, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int kernelH = 7;
    const int kernelW = 7;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 0;
    const int padW = 0;
    ave_pool(in, out, gid, kernelH, kernelW, strideH, strideW, padH, padW);
}

