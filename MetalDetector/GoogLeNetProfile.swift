//
//  GoogLeNetProfile.swift
//  MetalDetector
//
//  Created by Ivan Krasin on 10/31/15.
//  Copyright © 2015 Ivan Krasin. All rights reserved.
//

import Foundation
import Metal

public class GoogLeNetProfile {
    public static func GetThreadsPerThreadgroup() -> [String: MTLSize] {
        return [
            "conv1_7x7_s2": MTLSizeMake(8, 4, 1), // 36.1 ms vs 47.7 ms for 16x16x1
            "pool1_3x3_s2": MTLSizeMake(8, 8, 1), // 2.0 ms vs 2.3 ms for 16x16x1
            "pool1_norm1": MTLSizeMake(16, 32, 1), // 2.3 ms vs 2.5 ms for 16x16x1
            "conv2_3x3_reduce": MTLSizeMake(4, 2, 1), // 8.1 ms vs 9.8 ms for 16x16x1
            "conv2_3x3": MTLSizeMake(4, 4, 1), // 119.2 ms vs 138.9 ms for 16x16x1
            "conv2_norm2": MTLSizeMake(16, 16, 1), // 2.1 ms vs 2.1 ms for 16x16x1
            "pool2_3x3_s2": MTLSizeMake(16, 16, 1), // 3.3 ms vs 3.3 ms for 16x16x1
            "inception_3a_1x1": MTLSizeMake(2, 1, 1), // 9.2 ms vs 10.4 ms for 16x16x1
            "inception_3a_3x3_reduce": MTLSizeMake(4, 4, 1), // 12.6 ms vs 14.7 ms for 16x16x1
            "inception_3a_3x3": MTLSizeMake(2, 1, 1), // 39.3 ms vs 44.8 ms for 16x16x1
            "inception_3a_5x5_reduce": MTLSizeMake(16, 16, 1), // 1.9 ms vs 1.9 ms for 16x16x1
            "inception_3a_5x5": MTLSizeMake(4, 4, 1), // 8.9 ms vs 10.4 ms for 16x16x1
            "inception_3a_pool": MTLSizeMake(2, 4, 1), // 3.2 ms vs 3.4 ms for 16x16x1
            "inception_3a_pool_proj": MTLSizeMake(4, 4, 1), // 5.3 ms vs 6.3 ms for 16x16x1
            "inception_3b_1x1": MTLSizeMake(4, 4, 1), // 12.8 ms vs 24.1 ms for 16x16x1
            "inception_3b_3x3_reduce": MTLSizeMake(4, 4, 1), // 12.9 ms vs 18.1 ms for 16x16x1
            "inception_3b_3x3": MTLSizeMake(2, 1, 1), // 76.8 ms vs 83.1 ms for 16x16x1
            "inception_3b_5x5_reduce": MTLSizeMake(8, 8, 1), // 4.3 ms vs 4.6 ms for 16x16x1
            "inception_3b_5x5": MTLSizeMake(2, 1, 1), // 27.8 ms vs 38.3 ms for 16x16x1
            "inception_3b_pool": MTLSizeMake(16, 16, 1), // 2.6 ms vs 2.6 ms for 16x16x1
            "inception_3b_pool_proj": MTLSizeMake(1, 2, 1), // 11.5 ms vs 13.0 ms for 16x16x1
            "pool3_3x3_s2": MTLSizeMake(16, 16, 1), // 5.6 ms vs 5.6 ms for 16x16x1
            "inception_4a_1x1": MTLSizeMake(1, 2, 1), // 29.4 ms vs 42.3 ms for 16x16x1
            "inception_4a_3x3_reduce": MTLSizeMake(8, 8, 1), // 12.4 ms vs 16.9 ms for 16x16x1
            "inception_4a_3x3": MTLSizeMake(2, 1, 1), // 55.7 ms vs 60.6 ms for 16x16x1
            "inception_4a_5x5_reduce": MTLSizeMake(16, 16, 1), // 2.5 ms vs 2.5 ms for 16x16x1
            "inception_4a_5x5": MTLSizeMake(2, 2, 1), // 7.8 ms vs 12.2 ms for 16x16x1
            "inception_4a_pool": MTLSizeMake(8, 4, 1), // 5.1 ms vs 5.5 ms for 16x16x1
            "inception_4a_pool_proj": MTLSizeMake(2, 2, 1), // 14.3 ms vs 21.2 ms for 16x16x1
            "inception_4b_1x1": MTLSizeMake(1, 2, 1), // 26.4 ms vs 36.6 ms for 16x16x1
            "inception_4b_3x3_reduce": MTLSizeMake(8, 8, 1), // 16.4 ms vs 20.6 ms for 16x16x1
            "inception_4b_3x3": MTLSizeMake(1, 2, 1), // 69.3 ms vs 75.7 ms for 16x16x1
            "inception_4b_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.8 ms vs 3.0 ms for 16x16x1
            "inception_4b_5x5": MTLSizeMake(2, 2, 1), // 15.7 ms vs 22.5 ms for 16x16x1
            "inception_4b_pool": MTLSizeMake(16, 16, 1), // 5.5 ms vs 5.5 ms for 16x16x1
            "inception_4b_pool_proj": MTLSizeMake(2, 2, 1), // 14.9 ms vs 22.5 ms for 16x16x1
            "inception_4c_1x1": MTLSizeMake(2, 2, 1), // 18.9 ms vs 32.6 ms for 16x16x1
            "inception_4c_3x3_reduce": MTLSizeMake(2, 2, 1), // 18.7 ms vs 23.4 ms for 16x16x1
            "inception_4c_3x3": MTLSizeMake(2, 1, 1), // 89.8 ms vs 98.4 ms for 16x16x1
            "inception_4c_5x5_reduce": MTLSizeMake(16, 16, 1), // 2.9 ms vs 2.9 ms for 16x16x1
            "inception_4c_5x5": MTLSizeMake(2, 1, 1), // 15.4 ms vs 22.5 ms for 16x16x1
            "inception_4c_pool": MTLSizeMake(8, 8, 1), // 5.2 ms vs 5.6 ms for 16x16x1
            "inception_4c_pool_proj": MTLSizeMake(2, 2, 1), // 14.8 ms vs 22.4 ms for 16x16x1
            "inception_4d_1x1": MTLSizeMake(8, 4, 1), // 16.7 ms vs 29.9 ms for 16x16x1
            "inception_4d_3x3_reduce": MTLSizeMake(2, 2, 1), // 23.0 ms vs 26.3 ms for 16x16x1
            "inception_4d_3x3": MTLSizeMake(2, 1, 1), // 113.5 ms vs 125.8 ms for 16x16x1
            "inception_4d_5x5_reduce": MTLSizeMake(8, 8, 1), // 4.2 ms vs 6.4 ms for 16x16x1
            "inception_4d_5x5": MTLSizeMake(4, 4, 1), // 14.5 ms vs 28.9 ms for 16x16x1
            "inception_4d_pool": MTLSizeMake(8, 8, 1), // 5.3 ms vs 5.6 ms for 16x16x1
            "inception_4d_pool_proj": MTLSizeMake(2, 2, 1), // 14.8 ms vs 22.4 ms for 16x16x1
            "inception_4e_1x1": MTLSizeMake(2, 1, 1), // 42.2 ms vs 47.9 ms for 16x16x1
            "inception_4e_3x3_reduce": MTLSizeMake(2, 1, 1), // 27.2 ms vs 29.8 ms for 16x16x1
            "inception_4e_3x3": MTLSizeMake(2, 1, 1), // 139.7 ms vs 154.4 ms for 16x16x1
            "inception_4e_5x5_reduce": MTLSizeMake(8, 8, 1), // 4.3 ms vs 6.4 ms for 16x16x1
            "inception_4e_5x5": MTLSizeMake(2, 2, 1), // 27.0 ms vs 46.7 ms for 16x16x1
            "inception_4e_pool": MTLSizeMake(8, 8, 1), // 3.2 ms vs 3.4 ms for 16x16x1
            "inception_4e_pool_proj": MTLSizeMake(2, 2, 1), // 19.3 ms vs 37.1 ms for 16x16x1
            "pool4_3x3_s2": MTLSizeMake(16, 16, 1), // 4.4 ms vs 4.4 ms for 16x16x1
            "inception_5a_1x1": MTLSizeMake(8, 4, 1), // 41.4 ms vs 75.5 ms for 16x16x1
            "inception_5a_3x3_reduce": MTLSizeMake(4, 4, 1), // 22.7 ms vs 35.4 ms for 16x16x1
            "inception_5a_3x3": MTLSizeMake(4, 8, 1), // 96.8 ms vs 162.1 ms for 16x16x1
            "inception_5a_5x5_reduce": MTLSizeMake(16, 16, 1), // 6.1 ms vs 6.1 ms for 16x16x1
            "inception_5a_5x5": MTLSizeMake(4, 4, 1), // 14.4 ms vs 35.4 ms for 16x16x1
            "inception_5a_pool": MTLSizeMake(16, 16, 1), // 5.4 ms vs 5.4 ms for 16x16x1
            "inception_5a_pool_proj": MTLSizeMake(4, 4, 1), // 15.3 ms vs 36.7 ms for 16x16x1
            "inception_5b_1x1": MTLSizeMake(4, 8, 1), // 109.0 ms vs 146.2 ms for 16x16x1
            "inception_5b_3x3_reduce": MTLSizeMake(4, 4, 1), // 31.0 ms vs 50.0 ms for 16x16x1
            "inception_5b_3x3": MTLSizeMake(8, 4, 1), // 223.6 ms vs 303.6 ms for 16x16x1
            "inception_5b_5x5_reduce": MTLSizeMake(8, 8, 1), // 7.9 ms vs 9.9 ms for 16x16x1
            "inception_5b_5x5": MTLSizeMake(4, 1, 1), // 20.7 ms vs 37.9 ms for 16x16x1
            "inception_5b_pool": MTLSizeMake(16, 16, 1), // 4.1 ms vs 4.1 ms for 16x16x1
            "inception_5b_pool_proj": MTLSizeMake(3, 3, 1), // 15.5 ms vs 36.8 ms for 16x16x1
            "pool5_7x7_s1": MTLSizeMake(8, 4, 1), // 16.8 ms vs 20.9 ms for 16x16x1
        ]
    }
}
