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

    override func setUp() {
        engine = Engine()
        net = Net(engine: engine!, config: GoogLeNetConfig())
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
        let realErr1 = abs(L1-realL1)/L1
        let realErr2 = abs(L2-realL2)/L2
        print("\(name): L1: \(realL1), err1: \(realErr1), L2: \(realL2), err2: \(realErr2)")
        XCTAssertEqualWithAccuracy(realL1, L1, accuracy: L1 * err1)
        XCTAssertEqualWithAccuracy(realL2, L2, accuracy: L2 * err2)
    }

    func testGoogleNetOnCat() {
        let input = engine!.GetResourceAsMetalTexture("cat.png")
        XCTAssert(input != nil)
        if input == nil {
            return
        }
        XCTAssert(input != nil)

        var ans = net!.forward(input!)
        // HACK: find the answer
        for i in 1...5 {
            let (idx, p) = argMax(ans)
            ans[idx] = 0
            print("\(i). \(net!.labels[idx]) - \(p)")
        }
        checkMetrics("data", L1: 8815415, err1: 6E-5, L2: 646363264, err2: 2E-4)
        checkMetrics("conv1_7x7_s2", L1: 26894028, err1: 5E-3, L2: 7389178368, err2: 1E-2)
        checkMetrics("pool1_3x3_s2", L1: 15706585, err1: 5E-3, L2: 5238782464, err2: 1E-2)
        checkMetrics("pool1_norm1", L1: 6763517, err1: 5E-3, L2: 470019456, err2: 1E-2)
        checkMetrics("conv2_3x3_reduce", L1: 4607045, err1: 5E-3, L2: 365177216, err2: 1E-2)
        checkMetrics("conv2_3x3", L1: 6190394, err1: 1E-2, L2: 651581568, err2: 1E-2)
        checkMetrics("conv2_norm2", L1: 4979390.500000, err1: 1E-2, L2: 350693696, err2: 1E-2)
        checkMetrics("pool2_3x3_s2", L1: 3983016, err1: 1E-2, L2: 300735104, err2: 1E-2)
        checkMetrics("inception_3a_1x1", L1: 1376699, err1: 1E-2, L2: 133712880, err2: 1E-2)
        checkMetrics("inception_3a_3x3_reduce", L1: 1716600.250000, err1: 1E-2, L2: 141276912, err2: 2E-2)
        checkMetrics("inception_3a_3x3", L1: 2320588.250000, err1: 2E-2, L2: 287686272, err2: 4E-2)
        checkMetrics("inception_3a_5x5_reduce", L1: 418858.343750, err1: 1E-2, L2: 37575728, err2: 1E-2)
        checkMetrics("inception_3a_5x5", L1: 697953.750000, err1: 1E-2, L2: 80932752, err2: 1E-2)
        checkMetrics("inception_3a_pool", L1: 8837156, err1: 1E-2, L2: 766142080, err2: 1E-2)
        checkMetrics("inception_3a_pool_proj", L1: 460720.125000, err1: 2E-2, L2: 39634208, err2: 4E-2)
        checkMetrics("inception_3a_output", L1: 4855965, err1: 1E-2, L2: 541964032, err2: 2E-2)
        checkMetrics("inception_3b_output", L1: 2070707.625000, err1: 2E-2, L2: 173486784, err2: 4E-2)
        checkMetrics("inception_4a_output", L1: 803036.125000, err1: 4E-2, L2: 76048768, err2: 4E-2)
        checkMetrics("inception_4b_output", L1: 1172061, err1: 4E-2, L2: 77093112, err2: 4E-2)
        checkMetrics("inception_4c_output", L1: 907352.500000, err1: 0.032, L2: 62197672, err2: 0.04)
        checkMetrics("inception_4d_output", L1: 464165.843750, err1: 0.045, L2: 27689768, err2: 0.07)

        /*print("Pool1_3x3_s2: ")
        var buf = Array<Float>(count: 56 * 56, repeatedValue: 0)
        net!.blobs["pool1_3x3_s2"]!.getBytes(&buf, bytesPerRow: 56 * 4, bytesPerImage: 56 * 56 * 4,
            fromRegion: MTLRegionMake2D(0, 0, 56, 56), mipmapLevel: 0, slice: 0)
        for j in 0...6 {
            for i in 0...16 {
                print("\(buf[j*56+i])  ", terminator:"")
            }
            print("")
        }
        print("")
        print("")*/

        /*print("data: h: \(net!.data.height) w: \(net!.data.width) textureType: \(net!.data.textureType.rawValue) pixelFormat: \(net!.data.pixelFormat.rawValue)")
        for c in 0...2 {
            print("Data, c=\(c)")
            var buf0 = Array<Float>(count: 224 * 224 * 4, repeatedValue: 0)
            net!.data.getBytes(&buf0, bytesPerRow: 224 * 4, bytesPerImage: 224*224*4,
                fromRegion: MTLRegionMake2D(0, 0, 224, 224), mipmapLevel: 0, slice: c)
            for j in 0...6 {
                for i in 0...6 {
                    print("\(buf0[j*224+i])  ", terminator:"")
                }
                print("")
            }
            print("")
            print("")
        }

        print("Sample8x8 for data: ")
        var sample = engine!.Sample8x8(net!.data)
        for j in 0...7 {
            for i in 0...7 {
                print("\(sample[j*8+i])  ", terminator:"")
            }
            print("")
        }
        print("")
        print("")*/

        /*var buf = Array<Float>(count: 112 * 112 * 4, repeatedValue: 0)
        net!.conv1_7x7_s2.getBytes(&buf, bytesPerRow: 112*4, bytesPerImage: 112*112*4,
            fromRegion: MTLRegionMake2D(0, 0, 112, 112), mipmapLevel: 0, slice: 31)

        print("buf[341]=\(buf[341])")
        for j in 100...106 {
            for i in 100...106 {
                print("\(buf[j*112+i])  ", terminator:"")
            }
            print("")
        }*/

        self.measureBlock {
            for _ in 1...1 {
                self.net!.forward(input!)
            }
        }
    }

}
