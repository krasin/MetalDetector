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
            "conv1_7x7_s2": MTLSizeMake(8, 4, 1), // 36.7 ms vs 47.6 ms for 16x16x1
            "pool1_3x3_s2": MTLSizeMake(8, 8, 1), // 2.3 ms vs 2.5 ms for 16x16x1
            "pool1_norm1": MTLSizeMake(8, 8, 1), // 2.1 ms vs 2.3 ms for 16x16x1
            "conv2_3x3_reduce": MTLSizeMake(2, 4, 1), // 8.3 ms vs 9.9 ms for 16x16x1
            "conv2_3x3": MTLSizeMake(4, 4, 1), // 122.7 ms vs 142.7 ms for 16x16x1
            "conv2_norm2": MTLSizeMake(16, 16, 1), // 2.4 ms vs 2.4 ms for 16x16x1
            "pool2_3x3_s2": MTLSizeMake(2, 4, 1), // 3.2 ms vs 3.7 ms for 16x16x1
            "inception_3a_1x1": MTLSizeMake(2, 1, 1), // 9.1 ms vs 10.2 ms for 16x16x1
            "inception_3a_3x3_reduce": MTLSizeMake(1, 2, 1), // 12.8 ms vs 14.4 ms for 16x16x1
            "inception_3a_3x3": MTLSizeMake(4, 4, 1), // 39.6 ms vs 45.1 ms for 16x16x1
            "inception_3a_5x5_reduce": MTLSizeMake(8, 8, 1), // 1.9 ms vs 2.0 ms for 16x16x1
            "inception_3a_5x5": MTLSizeMake(2, 1, 1), // 9.3 ms vs 10.5 ms for 16x16x1
            "inception_3a_pool": MTLSizeMake(16, 16, 1), // 3.3 ms vs 3.3 ms for 16x16x1
            "inception_3a_pool_proj": MTLSizeMake(4, 8, 1), // 5.4 ms vs 6.3 ms for 16x16x1
            "inception_3b_1x1": MTLSizeMake(4, 4, 1), // 12.8 ms vs 24.4 ms for 16x16x1
            "inception_3b_3x3_reduce": MTLSizeMake(4, 4, 1), // 12.9 ms vs 16.6 ms for 16x16x1
            "inception_3b_3x3": MTLSizeMake(4, 4, 1), // 77.6 ms vs 83.3 ms for 16x16x1
            "inception_3b_5x5_reduce": MTLSizeMake(8, 8, 1), // 4.4 ms vs 4.7 ms for 16x16x1
            "inception_3b_5x5": MTLSizeMake(2, 1, 1), // 27.9 ms vs 36.9 ms for 16x16x1
            "inception_3b_pool": MTLSizeMake(16, 16, 1), // 2.6 ms vs 2.6 ms for 16x16x1
            "inception_3b_pool_proj": MTLSizeMake(4, 2, 1), // 11.7 ms vs 13.4 ms for 16x16x1
            "pool3_3x3_s2": MTLSizeMake(16, 16, 1), // 5.4 ms vs 5.4 ms for 16x16x1
            "inception_4a_1x1": MTLSizeMake(2, 1, 1), // 29.5 ms vs 40.1 ms for 16x16x1
            "inception_4a_3x3_reduce": MTLSizeMake(8, 8, 1), // 13.2 ms vs 16.9 ms for 16x16x1
            "inception_4a_3x3": MTLSizeMake(1, 2, 1), // 55.7 ms vs 60.7 ms for 16x16x1
            "inception_4a_5x5_reduce": MTLSizeMake(16, 8, 1), // 2.3 ms vs 2.5 ms for 16x16x1
            "inception_4a_5x5": MTLSizeMake(2, 2, 1), // 8.0 ms vs 12.3 ms for 16x16x1
            "inception_4a_pool": MTLSizeMake(2, 1, 1), // 5.2 ms vs 5.4 ms for 16x16x1
            "inception_4a_pool_proj": MTLSizeMake(2, 2, 1), // 14.3 ms vs 21.2 ms for 16x16x1
            "inception_4b_1x1": MTLSizeMake(1, 2, 1), // 26.3 ms vs 36.0 ms for 16x16x1
            "inception_4b_3x3_reduce": MTLSizeMake(2, 2, 1), // 16.3 ms vs 20.6 ms for 16x16x1
            "inception_4b_3x3": MTLSizeMake(2, 1, 1), // 69.4 ms vs 76.0 ms for 16x16x1
            "inception_4b_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.8 ms vs 2.9 ms for 16x16x1
            "inception_4b_5x5": MTLSizeMake(2, 1, 1), // 16.0 ms vs 22.5 ms for 16x16x1
            "inception_4b_pool": MTLSizeMake(2, 2, 1), // 5.4 ms vs 5.7 ms for 16x16x1
            "inception_4b_pool_proj": MTLSizeMake(2, 1, 1), // 15.7 ms vs 22.5 ms for 16x16x1
            "inception_4c_1x1": MTLSizeMake(2, 2, 1), // 18.8 ms vs 33.0 ms for 16x16x1
            "inception_4c_3x3_reduce": MTLSizeMake(2, 2, 1), // 18.8 ms vs 23.5 ms for 16x16x1
            "inception_4c_3x3": MTLSizeMake(1, 2, 1), // 90.2 ms vs 98.5 ms for 16x16x1
            "inception_4c_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.9 ms vs 3.0 ms for 16x16x1
            "inception_4c_5x5": MTLSizeMake(4, 4, 1), // 12.5 ms vs 22.5 ms for 16x16x1
            "inception_4c_pool": MTLSizeMake(16, 16, 1), // 5.4 ms vs 5.4 ms for 16x16x1
            "inception_4c_pool_proj": MTLSizeMake(2, 2, 1), // 14.9 ms vs 22.4 ms for 16x16x1
            "inception_4d_1x1": MTLSizeMake(8, 4, 1), // 17.3 ms vs 31.9 ms for 16x16x1
            "inception_4d_3x3_reduce": MTLSizeMake(2, 2, 1), // 23.0 ms vs 26.3 ms for 16x16x1
            "inception_4d_3x3": MTLSizeMake(2, 1, 1), // 113.5 ms vs 124.2 ms for 16x16x1
            "inception_4d_5x5_reduce": MTLSizeMake(8, 8, 1), // 4.2 ms vs 6.5 ms for 16x16x1
            "inception_4d_5x5": MTLSizeMake(4, 4, 1), // 14.2 ms vs 29.1 ms for 16x16x1
            "inception_4d_pool": MTLSizeMake(8, 8, 1), // 5.4 ms vs 5.8 ms for 16x16x1
            "inception_4d_pool_proj": MTLSizeMake(8, 4, 1), // 15.7 ms vs 22.4 ms for 16x16x1
            "inception_4e_1x1": MTLSizeMake(2, 1, 1), // 42.3 ms vs 47.7 ms for 16x16x1
            "inception_4e_3x3_reduce": MTLSizeMake(1, 2, 1), // 27.1 ms vs 29.8 ms for 16x16x1
            "inception_4e_3x3": MTLSizeMake(2, 1, 1), // 140.7 ms vs 152.8 ms for 16x16x1
            "inception_4e_5x5_reduce": MTLSizeMake(8, 8, 1), // 4.4 ms vs 6.5 ms for 16x16x1
            "inception_4e_5x5": MTLSizeMake(2, 2, 1), // 27.4 ms vs 44.0 ms for 16x16x1
            "inception_4e_pool": MTLSizeMake(16, 16, 1), // 3.4 ms vs 3.4 ms for 16x16x1
            "inception_4e_pool_proj": MTLSizeMake(2, 2, 1), // 19.7 ms vs 38.5 ms for 16x16x1
            "pool4_3x3_s2": MTLSizeMake(16, 16, 1), // 4.5 ms vs 4.5 ms for 16x16x1
            "inception_5a_1x1": MTLSizeMake(8, 4, 1), // 41.4 ms vs 75.6 ms for 16x16x1
            "inception_5a_3x3_reduce": MTLSizeMake(4, 4, 1), // 23.9 ms vs 35.1 ms for 16x16x1
            "inception_5a_3x3": MTLSizeMake(8, 4, 1), // 99.9 ms vs 164.0 ms for 16x16x1
            "inception_5a_5x5_reduce": MTLSizeMake(8, 8, 1), // 5.3 ms vs 6.2 ms for 16x16x1
            "inception_5a_5x5": MTLSizeMake(3, 3, 1), // 14.3 ms vs 35.7 ms for 16x16x1
            "inception_5a_pool": MTLSizeMake(16, 16, 1), // 5.9 ms vs 5.9 ms for 16x16x1
            "inception_5a_pool_proj": MTLSizeMake(3, 3, 1), // 15.6 ms vs 35.3 ms for 16x16x1
            "inception_5b_1x1": MTLSizeMake(4, 8, 1), // 110.1 ms vs 147.3 ms for 16x16x1
            "inception_5b_3x3_reduce": MTLSizeMake(8, 4, 1), // 31.3 ms vs 50.2 ms for 16x16x1
            "inception_5b_3x3": MTLSizeMake(8, 4, 1), // 226.6 ms vs 304.8 ms for 16x16x1
            "inception_5b_5x5_reduce": MTLSizeMake(8, 8, 1), // 7.8 ms vs 9.7 ms for 16x16x1
            "inception_5b_5x5": MTLSizeMake(3, 4, 1), // 20.8 ms vs 38.3 ms for 16x16x1
            "inception_5b_pool": MTLSizeMake(16, 16, 1), // 4.3 ms vs 4.3 ms for 16x16x1
            "inception_5b_pool_proj": MTLSizeMake(3, 3, 1), // 15.4 ms vs 34.8 ms for 16x16x1
            "pool5_7x7_s1": MTLSizeMake(16, 16, 1), // 18.1 ms vs 18.1 ms for 16x16x1
        ]
    }
}
