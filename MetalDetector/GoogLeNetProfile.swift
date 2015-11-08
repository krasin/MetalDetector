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
            "conv1_7x7_s2": MTLSizeMake(4, 1, 1), // 15.8 ms vs 27.5 ms for 16x16x1
            "pool1_3x3_s2": MTLSizeMake(16, 16, 1), // 2.0 ms vs 2.0 ms for 16x16x1
            "pool1_norm1": MTLSizeMake(3, 4, 1), // 2.2 ms vs 2.3 ms for 16x16x1
            "conv2_3x3_reduce": MTLSizeMake(4, 1, 1), // 5.3 ms vs 5.9 ms for 16x16x1
            "conv2_3x3": MTLSizeMake(4, 1, 1), // 59.4 ms vs 66.5 ms for 16x16x1
            "conv2_norm2": MTLSizeMake(16, 16, 1), // 2.2 ms vs 2.2 ms for 16x16x1
            "pool2_3x3_s2": MTLSizeMake(2, 4, 1), // 3.2 ms vs 3.6 ms for 16x16x1
            "inception_3a_1x1": MTLSizeMake(4, 4, 1), // 5.6 ms vs 6.1 ms for 16x16x1
            "inception_3a_3x3_reduce": MTLSizeMake(4, 4, 1), // 7.3 ms vs 8.2 ms for 16x16x1
            "inception_3a_3x3": MTLSizeMake(3, 1, 1), // 16.8 ms vs 34.9 ms for 16x16x1
            "inception_3a_5x5_reduce": MTLSizeMake(8, 8, 1), // 1.9 ms vs 2.0 ms for 16x16x1
            "inception_3a_5x5": MTLSizeMake(2, 4, 1), // 4.6 ms vs 6.0 ms for 16x16x1
            "inception_3a_pool": MTLSizeMake(2, 4, 1), // 3.2 ms vs 3.5 ms for 16x16x1
            "inception_3a_pool_proj": MTLSizeMake(2, 4, 1), // 3.2 ms vs 4.2 ms for 16x16x1
            "inception_3b_1x1": MTLSizeMake(4, 3, 1), // 10.3 ms vs 12.4 ms for 16x16x1
            "inception_3b_3x3_reduce": MTLSizeMake(4, 2, 1), // 10.3 ms vs 12.2 ms for 16x16x1
            "inception_3b_3x3": MTLSizeMake(4, 8, 1), // 29.9 ms vs 36.5 ms for 16x16x1
            "inception_3b_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.6 ms vs 2.9 ms for 16x16x1
            "inception_3b_5x5": MTLSizeMake(4, 4, 1), // 15.9 ms vs 30.5 ms for 16x16x1
            "inception_3b_pool": MTLSizeMake(16, 16, 1), // 2.8 ms vs 2.8 ms for 16x16x1
            "inception_3b_pool_proj": MTLSizeMake(16, 8, 1), // 7.2 ms vs 7.8 ms for 16x16x1
            "pool3_3x3_s2": MTLSizeMake(1, 2, 1), // 5.3 ms vs 5.6 ms for 16x16x1
            "inception_4a_1x1": MTLSizeMake(1, 1, 1), // 18.0 ms vs 28.2 ms for 16x16x1
            "inception_4a_3x3_reduce": MTLSizeMake(8, 8, 1), // 11.8 ms vs 12.7 ms for 16x16x1
            "inception_4a_3x3": MTLSizeMake(3, 1, 1), // 25.3 ms vs 34.9 ms for 16x16x1
            "inception_4a_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.2 ms vs 2.5 ms for 16x16x1
            "inception_4a_5x5": MTLSizeMake(2, 1, 1), // 6.2 ms vs 8.0 ms for 16x16x1
            "inception_4a_pool": MTLSizeMake(8, 4, 1), // 4.9 ms vs 5.3 ms for 16x16x1
            "inception_4a_pool_proj": MTLSizeMake(2, 1, 1), // 9.2 ms vs 10.8 ms for 16x16x1
            "inception_4b_1x1": MTLSizeMake(1, 1, 1), // 17.3 ms vs 25.4 ms for 16x16x1
            "inception_4b_3x3_reduce": MTLSizeMake(16, 16, 1), // 14.0 ms vs 14.0 ms for 16x16x1
            "inception_4b_3x3": MTLSizeMake(4, 8, 1), // 32.0 ms vs 36.0 ms for 16x16x1
            "inception_4b_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.8 ms vs 3.2 ms for 16x16x1
            "inception_4b_5x5": MTLSizeMake(2, 1, 1), // 10.7 ms vs 13.6 ms for 16x16x1
            "inception_4b_pool": MTLSizeMake(16, 16, 1), // 5.5 ms vs 5.5 ms for 16x16x1
            "inception_4b_pool_proj": MTLSizeMake(8, 8, 1), // 9.6 ms vs 11.5 ms for 16x16x1
            "inception_4c_1x1": MTLSizeMake(2, 4, 1), // 17.4 ms vs 20.5 ms for 16x16x1
            "inception_4c_3x3_reduce": MTLSizeMake(8, 4, 1), // 17.2 ms vs 21.7 ms for 16x16x1
            "inception_4c_3x3": MTLSizeMake(3, 1, 1), // 43.5 ms vs 46.5 ms for 16x16x1
            "inception_4c_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.7 ms vs 3.3 ms for 16x16x1
            "inception_4c_5x5": MTLSizeMake(2, 1, 1), // 10.7 ms vs 14.0 ms for 16x16x1
            "inception_4c_pool": MTLSizeMake(16, 16, 1), // 5.5 ms vs 5.5 ms for 16x16x1
            "inception_4c_pool_proj": MTLSizeMake(8, 16, 1), // 9.6 ms vs 11.5 ms for 16x16x1
            "inception_4d_1x1": MTLSizeMake(1, 1, 1), // 16.0 ms vs 19.1 ms for 16x16x1
            "inception_4d_3x3_reduce": MTLSizeMake(8, 16, 1), // 17.2 ms vs 23.0 ms for 16x16x1
            "inception_4d_3x3": MTLSizeMake(8, 8, 1), // 59.0 ms vs 59.0 ms for 16x16x1
            "inception_4d_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.9 ms vs 4.0 ms for 16x16x1
            "inception_4d_5x5": MTLSizeMake(3, 1, 1), // 13.6 ms vs 17.8 ms for 16x16x1
            "inception_4d_pool": MTLSizeMake(8, 8, 1), // 5.3 ms vs 5.6 ms for 16x16x1
            "inception_4d_pool_proj": MTLSizeMake(4, 4, 1), // 9.7 ms vs 11.4 ms for 16x16x1
            "inception_4e_1x1": MTLSizeMake(1, 2, 1), // 18.4 ms vs 32.0 ms for 16x16x1
            "inception_4e_3x3_reduce": MTLSizeMake(16, 16, 1), // 14.7 ms vs 14.7 ms for 16x16x1
            "inception_4e_3x3": MTLSizeMake(3, 1, 1), // 64.5 ms vs 71.5 ms for 16x16x1
            "inception_4e_5x5_reduce": MTLSizeMake(8, 8, 1), // 2.9 ms vs 3.9 ms for 16x16x1
            "inception_4e_5x5": MTLSizeMake(2, 2, 1), // 17.5 ms vs 34.4 ms for 16x16x1
            "inception_4e_pool": MTLSizeMake(16, 16, 1), // 4.1 ms vs 4.1 ms for 16x16x1
            "inception_4e_pool_proj": MTLSizeMake(1, 1, 1), // 17.9 ms vs 21.5 ms for 16x16x1
            "pool4_3x3_s2": MTLSizeMake(3, 3, 1), // 7.0 ms vs 7.8 ms for 16x16x1
            "inception_5a_1x1": MTLSizeMake(8, 4, 1), // 26.9 ms vs 38.6 ms for 16x16x1
            "inception_5a_3x3_reduce": MTLSizeMake(16, 16, 1), // 16.3 ms vs 16.3 ms for 16x16x1
            "inception_5a_3x3": MTLSizeMake(8, 4, 1), // 60.5 ms vs 64.1 ms for 16x16x1
            "inception_5a_5x5_reduce": MTLSizeMake(16, 16, 1), // 3.5 ms vs 3.5 ms for 16x16x1
            "inception_5a_5x5": MTLSizeMake(4, 4, 1), // 15.2 ms vs 26.4 ms for 16x16x1
            "inception_5a_pool": MTLSizeMake(3, 1, 1), // 6.5 ms vs 7.2 ms for 16x16x1
            "inception_5a_pool_proj": MTLSizeMake(3, 3, 1), // 17.8 ms vs 25.4 ms for 16x16x1
            "inception_5b_1x1": MTLSizeMake(8, 8, 1), // 46.9 ms vs 46.9 ms for 16x16x1
            "inception_5b_3x3_reduce": MTLSizeMake(4, 4, 1), // 18.4 ms vs 20.4 ms for 16x16x1
            "inception_5b_3x3": MTLSizeMake(2, 4, 1), // 90.0 ms vs 96.4 ms for 16x16x1
            "inception_5b_5x5_reduce": MTLSizeMake(16, 16, 1), // 5.9 ms vs 5.9 ms for 16x16x1
            "inception_5b_5x5": MTLSizeMake(3, 3, 1), // 17.1 ms vs 31.2 ms for 16x16x1
            "inception_5b_pool": MTLSizeMake(16, 16, 1), // 5.0 ms vs 5.0 ms for 16x16x1
            "inception_5b_pool_proj": MTLSizeMake(2, 1, 1), // 17.5 ms vs 25.2 ms for 16x16x1
            "pool5_7x7_s1": MTLSizeMake(4, 4, 1), // 16.8 ms vs 22.3 ms for 16x16x1
        ]
    }
}
