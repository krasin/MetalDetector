//
//  MetalDetectorTests.swift
//  MetalDetectorTests
//
//  Created by Ivan Krasin on 10/1/15.
//  Copyright © 2015 Ivan Krasin. All rights reserved.
//

import XCTest
import Metal
import MetalKit
import MetalPerformanceShaders
import UIKit
import MetalDetector


class MetalDetectorTests: XCTestCase {
    var engine: Engine?
    var net: Net?
    var cat: MTLTexture?

    override func setUp() {
        engine = Engine()
        net = Net(engine: engine!, config: GoogLeNetConfig(),
            threadsPerThreadgroup: GoogLeNetProfile.GetThreadsPerThreadgroup())

        cat = engine!.GetResourceAsMetalTexture("cat.png")
        XCTAssert(cat != nil)
        super.setUp()
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func argMax(arr : [Float]) -> (Int, Float) {
        var maxv : Float = arr[0]
        var idx : Int = 0
        for i in 0...arr.count-1 {
            if arr[i] > maxv {
                maxv = arr[i]
                idx = i
            }
        }
        return (idx, maxv)
    }

    func checkMetrics(name: String, L1: Float, err1: Float, L2: Float, err2: Float) {
        let realL1 = engine!.L1(net!.blobs[name]!)
        let realL2 = engine!.L2(net!.blobs[name]!)
        // let realErr1 = abs(L1-realL1)/L1
        // let realErr2 = abs(L2-realL2)/L2
        // print("checkMetrics(\"\(name)\", L1: \(L1), err1: \(realErr1*1.1), L2: \(L2), err2: \(realErr2*1.1))")
        XCTAssertEqualWithAccuracy(realL1, L1, accuracy: L1 * err1)
        XCTAssertEqualWithAccuracy(realL2, L2, accuracy: L2 * err2)
    }

    func testGoogleNetOnCat() {
        var ans = net!.forward(cat!)
        // HACK: find the answer
        for i in 1...5 {
            let (idx, p) = argMax(ans)
            ans[idx] = 0
            print("\(i). \(net!.labels[idx]) - \(p)")
        }
        checkMetrics("data", L1: 8.81542e+06, err1: 0.000380084, L2: 6.46363e+08, err2: 0.000457996)
        checkMetrics("conv1_7x7_s2", L1: 2.6894e+07, err1: 0.00110491, L2: 7.38918e+09, err2: 0.00214398)
        checkMetrics("pool1_3x3_s2", L1: 1.57066e+07, err1: 0.0011964, L2: 5.23878e+09, err2: 0.00229482)
        checkMetrics("pool1_norm1", L1: 6.76352e+06, err1: 0.000786839, L2: 4.70019e+08, err2: 0.00130362)
        checkMetrics("conv2_3x3_reduce", L1: 4.60704e+06, err1: 0.00133386, L2: 3.65177e+08, err2: 0.00260084)
        checkMetrics("conv2_3x3", L1: 6.19039e+06, err1: 0.00214362, L2: 6.51582e+08, err2: 0.00421105)
        checkMetrics("conv2_norm2", L1: 4.97939e+06, err1: 0.00188194, L2: 3.50694e+08, err2: 0.00333839)
        checkMetrics("pool2_3x3_s2", L1: 3.98302e+06, err1: 0.00189123, L2: 3.00735e+08, err2: 0.00331265)
        checkMetrics("inception_3a_1x1", L1: 1.3767e+06, err1: 0.00249991, L2: 1.33713e+08, err2: 0.00468566)
        checkMetrics("inception_3a_3x3_reduce", L1: 1.7166e+06, err1: 0.00258115, L2: 1.41277e+08, err2: 0.00487)
        checkMetrics("inception_3a_3x3", L1: 2.32059e+06, err1: 0.00329051, L2: 2.87686e+08, err2: 0.00639601)
        checkMetrics("inception_3a_5x5_reduce", L1: 418858.0, err1: 0.00230989, L2: 3.75757e+07, err2: 0.00444219)
        checkMetrics("inception_3a_5x5", L1: 697954.0, err1: 0.00300993, L2: 8.09328e+07, err2: 0.005881)
        checkMetrics("inception_3a_pool", L1: 8.83716e+06, err1: 0.00180339, L2: 7.66142e+08, err2: 0.0031494)
        checkMetrics("inception_3a_pool_proj", L1: 460720.0, err1: 0.0012534, L2: 3.96342e+07, err2: 0.00322521)
        checkMetrics("inception_3a_output", L1: 4.85596e+06, err1: 0.00283361, L2: 5.41964e+08, err2: 0.00565978)
        checkMetrics("inception_3b_output", L1: 2.07071e+06, err1: 0.00432286, L2: 1.73487e+08, err2: 0.00834659)
        checkMetrics("inception_4a_output", L1: 803036.0, err1: 0.00577003, L2: 7.60488e+07, err2: 0.0112075)
        checkMetrics("inception_4b_output", L1: 1.17206e+06, err1: 0.00654475, L2: 7.70931e+07, err2: 0.0132287)
        checkMetrics("inception_4c_output", L1: 907352.0, err1: 0.0074547, L2: 6.21977e+07, err2: 0.0151556)
        checkMetrics("inception_4d_output", L1: 464166.0, err1: 0.00755387, L2: 2.76898e+07, err2: 0.0160231)
        checkMetrics("pool5_7x7_s1", L1: 584.409912, err1: 0.002, L2: 1351.037598, err2: 0.007)

        self.measureBlock {
            for _ in 1...1 {
                self.net!.forward(self.cat!)
            }
        }
    }

    func testLargeConvolution() {
        for layer in net!.layers {
            if layer.shards == 0 || layer.bottoms.count > 1 {
                continue
            }
            net!.ProfileLayer(layer.name)
        }
    }
}
