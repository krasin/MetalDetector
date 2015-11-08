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
        // let realErr1 = abs(L1-realL1)/L1
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
        checkMetrics("data", L1: 8.81542e+06, err1: 0.000380708, L2: 6.46363e+08, err2: 0.000457561)
        checkMetrics("conv1_7x7_s2", L1: 2.6894e+07, err1: 0.00239215, L2: 7.38918e+09, err2: 0.00470541)
        checkMetrics("pool1_3x3_s2", L1: 1.57066e+07, err1: 0.00256157, L2: 5.23878e+09, err2: 0.00493635)
        checkMetrics("pool1_norm1", L1: 6.76352e+06, err1: 0.00113765, L2: 4.70019e+08, err2: 0.00197756)
        checkMetrics("conv2_3x3_reduce", L1: 4.60704e+06, err1: 0.0030698, L2: 3.65177e+08, err2: 0.00591738)
        checkMetrics("conv2_3x3", L1: 6.19039e+06, err1: 0.00479172, L2: 6.51582e+08, err2: 0.00959848)
        checkMetrics("conv2_norm2", L1: 4.97939e+06, err1: 0.00369263, L2: 3.50694e+08, err2: 0.00659015)
        checkMetrics("pool2_3x3_s2", L1: 3.98302e+06, err1: 0.00350697, L2: 3.00735e+08, err2: 0.00623133)
        checkMetrics("inception_3a_1x1", L1: 1.3767e+06, err1: 0.00471357, L2: 1.33713e+08, err2: 0.00898961)
        checkMetrics("inception_3a_3x3_reduce", L1: 1.7166e+06, err1: 0.00479199, L2: 1.41277e+08, err2: 0.00907613)
        checkMetrics("inception_3a_3x3", L1: 2.32059e+06, err1: 0.00638276, L2: 2.87686e+08, err2: 0.0125438)
        checkMetrics("inception_3a_5x5_reduce", L1: 418858.0, err1: 0.00444778, L2: 3.75757e+07, err2: 0.00869691)
        checkMetrics("inception_3a_5x5", L1: 697954.0, err1: 0.0063399, L2: 8.09328e+07, err2: 0.0124509)
        checkMetrics("inception_3a_pool", L1: 8.83716e+06, err1: 0.00327816, L2: 7.66142e+08, err2: 0.00578735)
        checkMetrics("inception_3a_pool_proj", L1: 460720.0, err1: 0.00190647, L2: 3.96342e+07, err2: 0.00575425)
        checkMetrics("inception_3a_output", L1: 4.85596e+06, err1: 0.00547751, L2: 5.41964e+08, err2: 0.0111515)
        checkMetrics("inception_3b_output", L1: 2.07071e+06, err1: 0.00836278, L2: 1.73487e+08, err2: 0.0161684)
        checkMetrics("inception_4a_output", L1: 803036.0, err1: 0.0111183, L2: 7.60488e+07, err2: 0.0219395)
        checkMetrics("inception_4b_output", L1: 1.17206e+06, err1: 0.0128661, L2: 7.70931e+07, err2: 0.0259412)
        checkMetrics("inception_4c_output", L1: 907352.0, err1: 0.0146419, L2: 6.21977e+07, err2: 0.0296205)
        checkMetrics("inception_4d_output", L1: 464166.0, err1: 0.0144568, L2: 2.76898e+07, err2: 0.0308716)
        checkMetrics("pool5_7x7_s1", L1: 584.41, err1: 0.00304463, L2: 1351.04, err2: 0.0102279)

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
