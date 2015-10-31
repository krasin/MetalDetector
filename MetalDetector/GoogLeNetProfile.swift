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
            "conv1_7x7_s2": MTLSizeMake(8, 16, 1), // 29.1 ms vs 46.4 ms for 16x16x1
            "conv2_3x3_reduce": MTLSizeMake(4, 8, 1), // 8.1 ms vs 9.4 ms for 16x16x1
            "conv2_3x3": MTLSizeMake(8, 4, 1), // 84.4 ms vs 93.1 ms for 16x16x1
            "pool2_3x3_s2": MTLSizeMake(4, 2, 1), // 3.2 ms vs 3.9 ms for 16x16x1
            "inception_3a_1x1": MTLSizeMake(4, 4, 1), // 11.5 ms vs 12.2 ms for 16x16x1
            "inception_3a_3x3_reduce": MTLSizeMake(1, 2, 1), // 13.1 ms vs 17.1 ms for 16x16x1
            "inception_3a_3x3": MTLSizeMake(4, 1, 1), // 38.4 ms vs 46.0 ms for 16x16x1
            "inception_3a_pool": MTLSizeMake(16, 8, 1), // 3.2 ms vs 3.5 ms for 16x16x1
            "inception_3a_pool_proj": MTLSizeMake(4, 4, 1), // 6.7 ms vs 7.6 ms for 16x16x1
            "inception_3b_1x1": MTLSizeMake(4, 4, 1), // 15.8 ms vs 28.1 ms for 16x16x1
            "inception_3b_3x3_reduce": MTLSizeMake(4, 4, 1), // 15.7 ms vs 18.6 ms for 16x16x1
            "inception_3b_3x3": MTLSizeMake(4, 2, 1), // 77.1 ms vs 88.8 ms for 16x16x1
            "inception_3b_5x5_reduce": MTLSizeMake(8, 8, 1), // 5.1 ms vs 5.4 ms for 16x16x1
            "inception_3b_5x5": MTLSizeMake(4, 8, 1), // 24.2 ms vs 36.4 ms for 16x16x1
            "inception_3b_pool_proj": MTLSizeMake(2, 1, 1), // 12.6 ms vs 16.1 ms for 16x16x1
            "inception_4a_1x1": MTLSizeMake(8, 16, 1), // 45.7 ms vs 53.0 ms for 16x16x1
            "inception_4a_3x3": MTLSizeMake(4, 3, 1), // 51.1 ms vs 64.4 ms for 16x16x1
            "inception_4a_5x5": MTLSizeMake(1, 1, 1), // 12.3 ms vs 13.9 ms for 16x16x1
            "inception_4a_pool": MTLSizeMake(8, 4, 1), // 5.2 ms vs 5.6 ms for 16x16x1
            "inception_4a_pool_proj": MTLSizeMake(4, 4, 1), // 17.3 ms vs 28.8 ms for 16x16x1
            "inception_4b_1x1": MTLSizeMake(2, 1, 1), // 38.2 ms vs 43.4 ms for 16x16x1
            "inception_4b_3x3": MTLSizeMake(1, 2, 1), // 65.1 ms vs 78.5 ms for 16x16x1
            "inception_4b_5x5": MTLSizeMake(4, 4, 1), // 17.0 ms vs 25.3 ms for 16x16x1
            "inception_4b_pool": MTLSizeMake(8, 8, 1), // 5.4 ms vs 5.8 ms for 16x16x1
            "inception_4b_pool_proj": MTLSizeMake(4, 8, 1), // 17.5 ms vs 29.6 ms for 16x16x1
            "inception_4c_3x3": MTLSizeMake(1, 2, 1), // 85.0 ms vs 113.5 ms for 16x16x1
            "inception_4c_5x5": MTLSizeMake(4, 4, 1), // 16.7 ms vs 25.6 ms for 16x16x1
            "inception_4c_pool_proj": MTLSizeMake(4, 4, 1), // 17.8 ms vs 29.5 ms for 16x16x1
            "inception_4d_3x3": MTLSizeMake(2, 1, 1), // 109.9 ms vs 143.5 ms for 16x16x1
            "inception_4d_5x5": MTLSizeMake(4, 4, 1), // 16.2 ms vs 28.4 ms for 16x16x1
            "inception_4d_pool_proj": MTLSizeMake(4, 4, 1), // 17.6 ms vs 29.4 ms for 16x16x1
            "inception_4e_3x3": MTLSizeMake(1, 2, 1), // 133.7 ms vs 178.3 ms for 16x16x1
            "inception_4e_5x5": MTLSizeMake(4, 3, 1), // 28.6 ms vs 39.5 ms for 16x16x1
            "inception_4e_pool_proj": MTLSizeMake(8, 16, 1), // 33.8 ms vs 42.6 ms for 16x16x1
            "inception_5a_3x3_reduce": MTLSizeMake(3, 3, 1), // 62.3 ms vs 65.8 ms for 16x16x1
            "inception_5a_3x3": MTLSizeMake(3, 1, 1), // 112.5 ms vs 127.0 ms for 16x16x1
            "inception_5a_5x5": MTLSizeMake(4, 3, 1), // 25.9 ms vs 30.2 ms for 16x16x1
            "inception_5a_pool_proj": MTLSizeMake(3, 1, 1), // 50.2 ms vs 56.8 ms for 16x16x1
            "inception_5b_1x1": MTLSizeMake(3, 3, 1), // 147.0 ms vs 155.6 ms for 16x16x1
            "inception_5b_3x3_reduce": MTLSizeMake(3, 3, 1), // 74.4 ms vs 78.4 ms for 16x16x1
            "inception_5b_3x3": MTLSizeMake(3, 1, 1), // 161.7 ms vs 183.8 ms for 16x16x1
            "inception_5b_5x5": MTLSizeMake(4, 3, 1), // 38.4 ms vs 42.8 ms for 16x16x1
            "inception_5b_pool_proj": MTLSizeMake(3, 1, 1), // 50.0 ms vs 55.7 ms for 16x16x1
        ]
    }
}
